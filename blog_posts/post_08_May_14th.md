# Blog Post 8 - Due May 14th

### Team Name: NLP Crossing

### Team Members: Sam Lee, Shuheng Liu, Robin Yang

#### Explanation of Approach

In the second advanced attempt, we continued our replication studies of the paper _Don't Stop Pretraining_ by including three more replications from two domains that were not covered in the last blog posts.

Prior to this attempt, we have run one experiment in the domains of reviews (IMDB) and one in CS (SciERC). However, in the previous attempt, due to the GPU memory limitation, we were not able to fully replicate the IMDB task with the paper's original parameter. Even though by changing the batch size we were able to generate some insightful results, the authors of the original paper gave us some suggestions on how to use a larger batch size to fit on the GPU, along with other matters.

In this attempt, we ran the tasks of AGNews (domain: news), HyperPartisan (domain: news) and ChemProt (domain: biomedical). The reason to run two tasks in the domain of news was that HyperPartisan, the first task that we ran, has a high standard deviation such that our results are drastically different from the paper's results. However, the authors of the paper explained that this was unsurprising due to the fact that this is a small dataset (515 labelled training data, 65 dev and test data each, comparing to IMDB which has 20000 labelled training data, 5000 dev data and 25000 test data), which resulted in wide range of _F<sub>1</sub>_ scores that they obtained, which covers the results of our experiments.

Using a batch size of 16 on the IMDB dataset was not possible previously due to GPU memory limitation, but now, following the author's suggestion, we have learned about gradient accumulation. Gradient accumulation is basically to make optimzer step after several batches. For example, stepping after two batches of size 8 gives us a effective batch size of 16.  Using gradient accumulation enables us to train with a batch size of 16. We can then compare the performance difference between the two batch sizes.

#### Results Report

As we contacted the authors of the paper, we were also informed that the paper that we were looking at was outdated and the newest version of the paper include some updates of the reported results. Therefore, we include the tables in the previous blog posts with updated results from the paper.

Previously we noted that our experiments got higher scores than the results on the paper, and we thought this was due to the difference in batch size. As we compare our results to the updated version of paper, we realize that our results are comparable to the ones reported in the paper, and batch size may not have a significant effect on the models' performances as we suspected.

<center>

##### IMDB

|                 | _Don't Stop Pretraining_ | Replication  Study(bsz=8) | Replication Study (bsz=16) |
| :-------------: | :----------------------: | :-----------------------: | :------------------------: |
|   __RoBERTa__   |    95.0<sub>0.2</sub>    |           94.7            |            94.5            |
|    __DAPT__     |    95.4<sub>0.1</sub>    |           94.8            |            95.0            |
|    __TAPT__     |    95.5<sub>0.1</sub>    |           95.5            |            95.7            |
| __DAPT + TAPT__ |    95.6<sub>0.1</sub>    |           95.7            |            95.3            |

##### SciERC

|                 | _Don't Stop Pretraining_ | Replication Study (bsz=8) | Replication  Study (bsz=16) |
| :-------------: | :----------------------: | :-----------------------: | :-------------------------: |
|   __RoBERTa__   |    77.3<sub>1.9</sub>    |           78.1            |            80.7             |
|    __DAPT__     |    80.8<sub>1.5</sub>    |           79.0            |            82.6             |
|    __TAPT__     |    79.3<sub>1.5</sub>    |           82.5            |            80.2             |
| __DAPT + TAPT__ |    81.3<sub>1.8</sub>    |           84.8            |            80.9             |

##### Biomedical (ChemProt)

|                 | _Don't Stop Pretraining_ | Replication  Study |
| :-------------: | :----------------------: | :----------------: |
|   __RoBERTa__   |    81.9<sub>1.0</sub>    |        82.3        |
|    __DAPT__     |    84.2<sub>0.2</sub>    |        84.4        |
|    __TAPT__     |    82.6<sub>0.4</sub>    |        82.8        |
| __DAPT + TAPT__ |    84.4<sub>0.4</sub>    |        84.1        |

##### News (AGNews)

|                 | _Don't Stop Pretraining_ | Replication Study |
| :-------------: | :----------------------: | :---------------: |
|   __RoBERTa__   |    93.9<sub>0.2</sub>    |       93.3        |
|    __DAPT__     |    93.9<sub>0.2</sub>    |       93.6        |
|    __TAPT__     |    94.5<sub>0.1</sub>    |       94.3        |
| __DAPT + TAPT__ |    94.6<sub>0.1</sub>    |       95.2        |

##### News (HyperPartisan)

|                 | _Don't Stop Pretraining_ | Replication Study (bsz=8) |
| :-------------: | :----------------------: | :-----------------------: |
|   __RoBERTa__   |    86.6<sub>0.9</sub>    |           85.5            |
|    __DAPT__     |    88.2<sub>5.9</sub>    |           72.6*           |
|    __TAPT__     |    90.4<sub>5.2</sub>    |           87.5            |
| __DAPT + TAPT__ |    90.0<sub>6.6</sub>    |           95.2            |

</center>

\*Even though this result seems to be too low comparing to the paper's results, as we explained in the Explanation of Approach section, this dataset is very small which results in a large variance across the performance scores. After running it for twice more, we obtained the results of 90.4 and 96.3, which are more comparable to the results reported in the paper.


#### Failure Modes

Generally we would consider this replication study successful at this point, and there are only a few things that may result in failure of our experiments. 

One thing is that we did not re-run the HyperPartisan task after implementing gradient accumulation which enables us to use the original batch size. This might be one of the reasons that our experiments yield slightly different results aside from the nature of this dataset. 

Another reason that could be used to explain the discrepancy between our results and the paper's is the number of experiments that we ran. In the paper, they have mentioned that the results which they reported are the average across multiple runs, while ours are the results after only one run. If time permits, we might be able to re-run the experiments to get a range of values and report them in the same way (mean + standard deviation) as the paper did.


#### Next Step

Since now we have successfully replicated the results from the two remaining domains (i.e. Biomedical and News) in the paper, we are ready to tackle the next major goal, which is to investigate the effects of __fine-tuning__ original RoBERTa. Specifically, due to the time constrain, we are planning on first fine-tuning RoBERTa by adding a layer or more to the network with its parameters randomly initialized and training on only the newly included layer(s), while keeping other parameters frozen. We hope that by doing so, the fine-tuned model may have a chance to achieve similar performance as the model that was pretrained on the IMDB dataset. We chose to compare against the IMDB dataset mainly because we wrote our own code to replicate the result for this particular dataset in the beginning of the project and it would be much easier to include extra layer(s) with full control over it.

If we are able to get convincing result from fine-tuning original RoBERTa, comparing to the pre-trained model on the IMDB dataset, we will be able to move on to the remaining three domains, applying the same fine-tuned model. Then, we will carry out error analysis as well as comapring results between fine-tuned RoBERTa and specific-domain pre-trained models.