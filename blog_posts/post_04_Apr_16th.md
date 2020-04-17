# Blog Post 4 - Due April 16th 

### Team Name: NLP Crossing

### Team Members: Sam Lee, Shuheng Liu, Robin Yang

__GitHub URL: https://github.com/minhsuanlee/cse481n-20sp-NLP-Crossing__

#### Strawman/baseline Approach:
- Forked and cloned [Bert from Google research](https://github.com/google-research/bert)
- Ran SQuAD1.1 using a pre-trained model (BERT-Base, Cased: 12-layer, 768-hidden, 12-heads , 110M parameters)
- Using the model above, we were able to reach the expected performance of:
  ```{"exact_match": 80.88930936613056, "f1": 88.4823666931611}```

#### Evaluation framework:
- For the strawman/baseline approach, we used [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py) included in [Bert from Google research](https://github.com/google-research/bert) to evaluate our BERT's predictions on the SQuAD1.1 data
- Select domain-specific tasks from _Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks_, which should be done in the next week
- Evaluate the performances of different versions of BERT/RoBERTa using the same evaluation methods as described in the paper above