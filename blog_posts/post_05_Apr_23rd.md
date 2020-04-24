# Blog Post 5 - Due April 23rd 

### Team Name: NLP Crossing

### Team Members: Sam Lee, Shuheng Liu, Robin Yang

__GitHub URL: https://github.com/minhsuanlee/cse481n-20sp-NLP-Crossing__

We have chosen our two baselines to be using [IMDB](http://ai.stanford.edu/~amaas/data/sentiment/index.html)
and and SCIERC datasets. For IMDB,
we have four pretrained models on Huggingface, and we arbitrarily selected
`dsp_roberta_base_tapt_imdb_20000` as our pretrained model. The IMDB dataset contains
25000 movie reviews of categories "posotive" and "negative". We have applied the
pretrained model on this dataset but got really poor performance. The output
differs each time so we assume there is some randomness in the model. However, 
this is more likely to be a result of our incorrect implementation. We are still
in the process of learning to use Huggingface and all the pretrained models.