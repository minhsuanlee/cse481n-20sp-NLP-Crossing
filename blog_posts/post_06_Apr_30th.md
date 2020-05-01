# Blog Post 6 - Due April 30th 

### Team Name: NLP Crossing

### Team Members: Sam Lee, Shuheng Liu, Robin Yang

__GitHub URL: https://github.com/minhsuanlee/cse481n-20sp-NLP-Crossing__

#### Explanation of Approach

To start our replication study, we selected the sentiment analysis task under the domain "review" using the IMDB database. Even though this is not the main focus of the paper, we decided to make this our first advanced solution attempt as this seems to be more manageable, and since our ultimate goal (find a way to fine-tune RoBERTa in lieu of domain-specific pre-training) deviates from what is proposed in the paper, we think this should be a reasonable starting point.

As we were unable to find the code base used for the paper at first, we had to write our own scripts to train and test the models. We made use of the models uploaded to Hugging Face by Allen Institute for AI (https://huggingface.co/allenai) as well as the original RoBERTa. Here are the list of models that we used:

- Original RoBERTa: [RoBERTa](https://huggingface.co/roberta-base)
- DAPT (Domain Adaptive Pre-Training): [allenai/reviews_roberta_base](https://huggingface.co/allenai/reviews_roberta_base) 
- TAPT (Task Adaptive Pre-Training): [allenai/dsp_roberta_base_tapt_imdb_70000](https://huggingface.co/allenai/dsp_roberta_base_tapt_imdb_70000)
- DAPT + TAPT: [allenai/dsp_roberta_base_dapt_reviews_tapt_imdb_70000](https://huggingface.co/allenai/dsp_roberta_base_dapt_reviews_tapt_imdb_70000)

We also found a tutorial of how to do sentiment analysis with models on Hugging Face here: 

[Sentiment Analysis with BERT and Transformers by Hugging Face using PyTorch and Python](https://www.curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/)

In addition, we also converted the IMDB dataset from `.txt` files to `.csv` files so that it is compatible with the code that we wrote. The `.csv` files are separated by train (which in our code was further separated to 90% train and 10% dev) and test. Each `.csv` file contains 25000 reviews from IMDB and are divided into positive reviews (score >= 7 out of 10) and negative reviews (score <= 4 out of 10). The IMDB dataset can be found here:

[Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/index.html)

The scripts that we used to train and test the models are stored in `cse481n-20sp-NLP-Crossing/scripts/` and the log files are in `cse481n-20sp-NLP-Crossing/logs/` where the logs when training the models are in `train_dev/` whereas those when testing the models are in `test/`. 

#### Reporting Results
Here are the results that we got as compared to the results recorded in the original paper:

|                 | _Don't Stop Pre-Training_ | NLP Crossing |
| --------------- | :-----------------------: | :----------: |
| __RoBERTa__     |    94.5<sub>0.1</sub>*    |     94.0     |
| __DAPT__        |    95.1<sub>0.1</sub>     |     94.9     |
| __TAPT__        |    95.1<sub>0.2</sub>     |     95.4     |
| __DAPT + TAPT__ |    95.2<sub>0.1</sub>     |     95.2     |

*Note: Reported results are tested macro _F_<sub>1</sub>, with standard deviations as subscripts.

For each model, we also generated a confusion matrix demonstrating the predictions of our models:

__RoBERTa:__

|              | Predicted Positive | Predicted Negative |
| ------------ | :----------------: | :----------------: |
| __Positive__ |   11397 (91.18%)   |    1103 (8.82%)    |
| __Negative__ |    440 (3.52%)     |   12060 (96.48%)   |

__DAPT:__

|              | Predicted Positive | Predicted Negative |
| ------------ | :----------------: | :----------------: |
| __Positive__ |   11812 (94.50%)   |    688 (5.50%)     |
| __Negative__ |    598 (4.78%)     |   11902 (95.22%)   |

__TAPT:__

|              | Predicted Positive | Predicted Negative |
| ------------ | :----------------: | :----------------: |
| __Positive__ |   11802 (94.42%)   |    698 (5.58%)     |
| __Negative__ |    471 (3.77%)     |   12029 (96.23%)   |

__DAPT + TAPT:__

|              | Predicted Positive | Predicted Negative |
| ------------ | :----------------: | :----------------: |
| __Positive__ |   11625 (93.00%)   |    875 (7.00%)     |
| __Negative__ |    352 (2.82%)     |   12148 (97.18%)   |

#### Failure Modes

- Notice RoBERTa and DAPT F1 lower than expected
- Data: size (80% 20%)
- Different code base (dropout)

One of the problem that we encountered when we first started was that the IMDB training dataset was divided into positive, negative and unsupervised. The unsupervised data was used for unsupervised learning that we did not realize. Adding the unsupervised data, which we treated as a category other than positive or negative, significantly lowered our models performance to 67% accuracy (our evaluation metrics then was the accuracy of the prediction rather than the _F_<sub>1</sub> score).

In the results shown in the previous section, one can observe that even though we got similar results using TAPT and DAPT + TAPT models, the _F_<sub>1</sub> score of the DAPT model was lower than the expected results by 2 standard deviations, and that of the RoBERTa model was lower by 5 standard deviations. Aside from the possibility that the original paper, reporting results using five different seeds, may not cover some anomalies, here are some possible causes that account for the differences in results.

One reason for the difference in results might be attributed to the difference in data division. Since originally we were unable to locate the code base related to the paper that we are performing replication study on, we had to come up with our own metric to divide the dataset into training, validation, and test sets. We ended up using training : validation= 90% : 10% instead of training : validation= 80% : 20% that was documented in the code base that we found later. Additionally, we randomly divided the full dataset into the sets described above, which was likely to be different from the division done in the code base.

There are some hyperparameters that we used in our settings that are different from the ones used in the original code base. Although we used the same number of epochs as the paper did, the dropout rate of the final linear layer was different - we used a dropout rate of 0.3 where as the original code base used 0.1. The dropout rate may have an effect on our convergence rate. The model may converge slower because of a different dropout rate.

#### Next Step

We wrote our own code at the beginning as we could not find the code based used in the paper. However, Tal pointed out that the code base can be found [here](https://github.com/allenai/dont-stop-pretraining). Despite being able to practice coding with RoBERTa-base pretrained models and transformers as well as validating the results from the original paper using a different code base, it is also crucial to perform a replication using the original code base so as to confirm the consistency between the results published and the code base uploaded. Therefore, the first step that we are going to take next is to run the original training and testing on these four models with the configurations used in the original code base.

Another step that we will take is to replicate the results from a different domain. Our current choice is the SciERC database and since we now have the original code base, it would not take too long to replicate the experiments.

The ultimate goal of our project, which we first identified as the stretched goal, is to find a way to fine-tune the original RoBERTa to achieve comparable performances as these DAPT and TAPT models for those without the computational resources to pre-train it on specific domains are able to utilize. Therefore, after our replication of the experiments mentioned above (as the process is getting easier since we have access to the original code base now), we aim to start brainstorming ideas and start trying to implement them for fine-tuning RoBERTa. 