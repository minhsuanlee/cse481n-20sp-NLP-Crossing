import torch
import numpy as np
from collections import defaultdict
import pandas as pd
from transformers import *
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn, optim
import os
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import f1_score, confusion_matrix

PRETRAINED_MODEL = 'roberta-base'
BIN_NAME = 'roberta-base.bin'
config = AutoConfig.from_pretrained(PRETRAINED_MODEL)
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
roberta_model = AutoModel.from_pretrained(PRETRAINED_MODEL, config=config)

MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 10

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('train_imdb.csv')
print(df.head())


class IMDBDataset(Dataset):
  def __init__(self, reviews, targets, tokenizer, max_len):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.reviews)

  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

df_train, df_test = train_test_split(
  df,
  test_size=0.1,
  random_state=RANDOM_SEED
)

print("data_shapes", df_train.shape, df_test.shape)

def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = IMDBDataset(
    reviews=df.review.to_numpy(),
    targets=df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)


class SentimentClassifier(nn.Module):
  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.roberta = AutoModel.from_pretrained(PRETRAINED_MODEL, config=config)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.roberta.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.roberta(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)

class_names = ['pos', 'neg']

classifyModel = SentimentClassifier(len(class_names))
classifyModel = classifyModel.to(device)


optimizer = AdamW(classifyModel.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
  model = model.train()

  losses = []
  correct_predictions = 0

  all_targets = None
  all_preds = None

  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    # print(preds)
    loss = loss_fn(outputs, targets)

    # correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    if all_targets is None:
      all_targets = targets
    else:
      all_targets = torch.cat((all_targets, targets))
    
    if all_preds is None:
      all_preds = preds
    else:
      all_preds = torch.cat((all_preds, preds))

  # return correct_predictions.double() / n_examples, np.mean(losses)
  return f1_score(all_targets.cpu(), all_preds.cpu()), np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0
  all_preds = None
  all_targets = None

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      # correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

      if all_targets is None:
        all_targets = targets
      else:
        all_targets = torch.cat((all_targets, targets))
        
      if all_preds is None:
        all_preds = preds
      else:
        all_preds = torch.cat((all_preds, preds))


  print(confusion_matrix(all_targets.cpu(), all_preds.cpu()))

  # return correct_predictions.double() / n_examples, np.mean(losses)
  return f1_score(all_targets.cpu(), all_preds.cpu()), np.mean(losses)


history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):

  print(f'Epoch {epoch + 1}/{EPOCHS}')
  print('-' * 10)

  train_acc, train_loss = train_epoch(classifyModel, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train))

  print(f'Train loss {train_loss} accuracy {train_acc}')

  val_acc, val_loss = eval_model(
    classifyModel,
    test_data_loader,
    loss_fn,
    device,
    len(df_test)
  )

  print(f'Val   loss {val_loss} F1_score {val_acc}')
  print()

  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)

  if val_acc > best_accuracy:
    torch.save(classifyModel.state_dict(), BIN_NAME)
    best_accuracy = val_acc

