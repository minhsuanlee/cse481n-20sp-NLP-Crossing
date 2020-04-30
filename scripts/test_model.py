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
BIN_NAME = '../roberta-base.bin'

config = AutoConfig.from_pretrained(PRETRAINED_MODEL)
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
roberta_model = AutoModel.from_pretrained(PRETRAINED_MODEL, config=config)

MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 10

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('test_imdb.csv')
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

test_data_loader = create_data_loader(df, tokenizer, MAX_LEN, BATCH_SIZE)
loss_fn = nn.CrossEntropyLoss().to(device)
class_names= ['pos', 'neg']

classifyModel = SentimentClassifier(len(class_names))
classifyModel = classifyModel.to(device)


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


print('Testing...')
classifyModel.load_state_dict(torch.load(BIN_NAME))
classifyModel = classifyModel.to(device)
test_acc, test_loss = eval_model(classifyModel, test_data_loader,
                                 loss_fn, device, len(df))

print(f'Test   loss {test_loss} F1_score {test_acc}')
