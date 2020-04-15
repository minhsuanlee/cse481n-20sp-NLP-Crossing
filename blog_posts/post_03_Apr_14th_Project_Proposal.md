# Blog Post 3 (Project Proposal) - Due April 14th

### Team Name: NLP Crossing

### Team Members: Sam Lee, Shuheng Liu, Robin Yang

__GitHub URL: https://github.com/minhsuanlee/cse481n-20sp-NLP-Crossing__

##### Project Name: 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Evolution of BERT

##### Project Objectives
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; We would like to take two domain-specific tasks from _Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks_ and replicate the experiments on BERT, RoBERTa and RoBERTa's that were pre-trained on these specific domains, and compare and analyze the results. Eventually, we hope to design methond that will be able to fine-tune RoBERTa to achieve comparable results as the RoBERTa's that were pre-trained on these domains.

##### Motivation
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; RoBERTa is one of the state-of-the-art, best-performing language model that was ever developed. However, as it was pre-trained on data such as reddit posts, news, Wikipedia articles, etc., it can only be used for general purposes. For general public without great computing resources, it is almost implausible to pre-train RoBERTa in a specific domain unless it has already been pre-trained by some institution. Our group strives to design a way to fine-tune the general-purpose RoBERTa to achieve comparable performance as the domain-specific RoBERTa's.

##### Minimal Action Plan
1. Same replication study using vanilla BERT as (2)
2. Replication study using vanilla RoBERTa as well as two pre-trained, domain-specific RoBERTa on tasks from related domains

##### Stretch Goals
1. Fine-tune vanilla RoBERTa and perform the same study in order to compare with pre-trained, domain-specific versions of RoBERTa
2. Provide generalizable techniques that can be applied to vanilla RoBERTa and improve model performance on domain-specific tasks without the pre-training process on those specific domains

##### Methodology
1. Apply models pre-trained on different datasets to several tasks in selected domains
2. Compare performances between models
3. Fine-tune the vanilla model by adding extra layers and training on datasets in the selected domains
4. Evaluate the performance of fine-tuned models

##### Evaluation Plan
1. Use the same evaluation techniques used with pre-trained, domain-specific RoBERTa
2. Compare performances and do error analysis across different variations of BERT/RoBERTa

##### Available Resources
   - [Google Research BERT](https://github.com/google-research/bert)
   - [FAIR Vanilla RoBERTa](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md)
   - [AI2 pre-trained domain-specific RoBERTa](https://huggingface.co/allenai)
##### Related Work
   - [BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
   - [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf)
   - Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks