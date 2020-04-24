# Blog Post 5 - Due April 23rd 

### Team Name: NLP Crossing

### Team Members: Sam Lee, Shuheng Liu, Robin Yang

__GitHub URL: https://github.com/minhsuanlee/cse481n-20sp-NLP-Crossing__



For this blog post, we were supposed to write about multiple strawman/baseline
approaches that we did, our evaluation framework, and some error analysis.
However, as we are trying to replicate the experiements
from _Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks_,
there is not too many strawman/baseline approaches that we can do and
therefore we mainly focused on preparation of the replication.

Later we have chosen our two baselines to be using [IMDB](http://ai.stanford.edu/~amaas/data/sentiment/index.html)
and and SCIERC datasets. For IMDB,
we have four pretrained models on Huggingface, and we arbitrarily selected
`dsp_roberta_base_tapt_imdb_20000` as our pretrained model. The IMDB dataset contains
25000 movie reviews of categories "posotive" and "negative". We have applied the
pretrained model on this dataset but got really poor performance. The output
differs each time so we assume there is some randomness in the model. However, 
this is more likely to be a result of our incorrect implementation. We are still
in the process of learning to use Huggingface and all the pretrained models.

Here is a list of what we completed and what we need to do recently:


__What we did:__


__What need to be done:__
