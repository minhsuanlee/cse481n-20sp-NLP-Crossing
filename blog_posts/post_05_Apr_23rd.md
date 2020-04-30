# Blog Post 5 - Due April 23rd 

### Team Name: NLP Crossing

### Team Members: Sam Lee, Shuheng Liu, Robin Yang

__GitHub URL: https://github.com/minhsuanlee/cse481n-20sp-NLP-Crossing__


As a clarification, the main objective of our project is to use pretrained models from AllenNLP to replicate some experiments from the paper _Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks_, and if there is time after the replication, we aim to fine-tune the general purpose RoBERTa to improve its performance on specific domains without going through the pretraining process since we do not have the necessary computing power.

For this blog post, we were supposed to write about multiple strawman/baseline approaches that we did, our evaluation framework, and some error analysis. However, as we are trying to replicate some experiments, our focus was mainly to decide on the tasks that we would like to replicate.

We have chosen our two experiments to be using [IMDB](http://ai.stanford.edu/~amaas/data/sentiment/index.html) and and [SCIERC](http://nlp.cs.washington.edu/sciIE/) datasets. For IMDB, we have four pretrained models on Huggingface, and we arbitrarily selected `dsp_roberta_base_tapt_imdb_20000` as our pretrained model. The IMDB dataset contains 25000 movie reviews of categories "posotive" and "negative". We have applied the pretrained model on this dataset but got really poor performance. The output differs each time so we assume there is some randomness in the model. However, this is more likely to be a result of our incorrect implementation. For SCIERC, we are still trying to figure out the general purpose and usage of this dataset.

Since we were unable to find specific instructions on how to use the pretrained models or any scripts using them, and none of us were familiar with the models provided on Hugging Face, we need to either find existing scripts for running experiments and/or understand how to use these models quickly.