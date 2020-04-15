# Blog Post 3 - Due April 14th

### Team Name: NLP Crossing

### Team Members: Sam Lee, Shuheng Liu, Robin Yang

__GitHub URL: https://github.com/minhsuanlee/cse481n-20sp-NLP-Crossing__

##### Project Name: 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; RoBERTa For Everyone
##### Project Objectives
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; We would like to explore the performance of fine-tuned RoBERTa on a domain comparing to a RoBERTa pretrained on that domain.
##### Motivation
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; RoBERTa is one of the state-of-the-art, best-performing language model that people have ever developed. However, as it was pre-trained on data such as reddit posts, news, Wikipedia articles, etc., it can only be used for general purposes. For general public without great computing resources, it is almost implausible to pre-train RoBERTa in a specific domain unless it has already been pre-trained by some institution. Our group aims to evaluate the performance of a RoBERTa pre-trained in a certain domain, and strives to design a way to fine-tune the general-purpose RoBERTa with data from that domain to achieve comparable performance.
##### Minimal Action Plan
1. 
##### Methodology
##### Stretch Goals
##### Evaluation Plan
##### Available Resources
   - [Google Research BERT](https://github.com/google-research/bert)
   - [FAIR Original RoBERTa](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md)
   - [AI2 pre-trained RoBERTa](https://huggingface.co/allenai)
##### Related Work
   - [BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
   - [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf)
   - Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks

rep w/ bert -> rep w/ roberta -> rep w/ domained roberta -> rep w/ fine-tuned roberta -> compare domained and fine-tuned