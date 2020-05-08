# Blog Post 7 - Due May 7th

### Team Name: NLP Crossing

### Team Members: Sam Lee, Shuheng Liu, Robin Yang

__GitHub URL: https://github.com/minhsuanlee/cse481n-20sp-NLP-Crossing__

#### Advancing Solution Attempt & Additional Experiments

In the previous attempt, as we were not able to locate the original code base that the paper _Don't Stop Pre-Training_ used, we had to develop our own code base to find out the _F<sub>1</sub>_ score using four models (RoBERTa, DAPT, TAPT and DAPT + TAPT) on the IMDB task.

As we have acquired the original code base, which was forked [here](https://github.com/ShuhengL/dont-stop-pretraining), we were able to re-run the IMDB tasks using the same models, as well as running an additional task in the domain of CS using the database SciERC.

While running the IMDB task, we realized that we could not use the batch size (16) that was used in the paper due to memory constraint of the GPU that we were using. Therefore, we had to lower the batch size to 8 when running the experiments. While this might have slow down the training process, we were able to obtain comparable and even better performance. Here are the results that we obtained, including the previous results using our own code base:

<center>

|                 | _Don't Stop Pre-Training_ | NLP Crossing Code Base | Replication  Study |
| :-------------: | :-----------------------: | :--------------------: | :----------------: |
|   __RoBERTa__   |    94.5<sub>0.1</sub>     |          94.0          |        94.7        |
|    __DAPT__     |    95.1<sub>0.1</sub>     |          94.9          |        94.8        |
|    __TAPT__     |    95.1<sub>0.2</sub>     |          95.4          |        95.5        |
| __DAPT + TAPT__ |    95.2<sub>0.1</sub>     |          95.2          |        95.7        |

</center>

By using the paper's code base, we obtained results higher than one standard deviation in all models except for DAPT, which was 0.3% lower than the result reported on the paper.

Additionally, we tried to run the SciERC task, a relation classification task, in the domain of CS. We first ran it with bsz (short for batch size) = 8. Since this is a relatively smaller dataset (3219 as opposed to IMDB which contains 20000 in the training set), we were able to adjust the batch size to 16, the one that the paper used, to do the replication study. Here are the results that we obtained for both studies:

<center>

|                 | _Don't Stop Pre-Training_ | Replication Study (bsz=8) | Replication  Study (bsz=16) |
| :-------------: | :-----------------------: | :-----------------------: | :-------------------------: |
|   __RoBERTa__   |    78.9<sub>1.6</sub>     |           78.1            |            80.7             |
|    __DAPT__     |    79.8<sub>1.2</sub>     |           79.0            |            82.6             |
|    __TAPT__     |    79.6<sub>1.5</sub>     |           82.5            |            80.2             |
| __DAPT + TAPT__ |    80.2<sub>1.5</sub>     |           84.8            |            80.9             |

</center>

On the left most column, we can observe that all the results fall into 1 standard deviation of the reported results on the paper, but when we used a smaller batch size, even though we obtained slightly lower results using RoBERTa and DAPT (still within 1 standard deviation), the TAPT and DAPT + TAPT models performed exceptionally well as compared to the original results.

All the results reported above still follow the same pattern as the paper's results, with pre-trained models performing better than RoBERTa and DAPT + TAPT performing better than DAPT or TAPT alone.

One reminder when using the paper's code base is that one needs to install the cuda 9.2 when setting up the environment. The [installation](https://github.com/ShuhengL/dont-stop-pretraining#installation) page did not specify that cuda needs to be installed, and we found that the code base is only compatible with cuda 9.



#### Error Analysis

Due to memory limitations we were only able to use a batch size of 8 on the IMDB dataset. We have noticed that if we use the smaller batch size in some experiments, especially expirements that envolve TAPT, we generally get higher F1 scores than reported in the paper. The scores we got have exceeded the standard deviations reported.

On SciERC dataset, we were able to use the original batch size used in the paper. We then tried both batch sizes and compared the performance. We again got better results in pre-trained models involving TAPT.

One possible reason for the better performance is that training with a larger batch size converges faster and is more stable, but may not generalize very well, whereas a smaller batch size introduces better generalization ability. Because we stop based on the performance on the validation dataset, if the model has a good generalization ability, we will be more likely to get a similar result on the test dataset to the validation dataset.

#### Next Action Plan

Since we have gotten convincing results by replicating the results from two domains (i.e. Reviews and CS) in the paper: _Don't Stop Pretraining_, we are now moving onto the remaining two domains (i.e. Biomedical and News). The plan is to replicate the results for four tasks, each from a different domain documented in the paper. We will continue to try out smaller batch sizes (i.e. 8) on the remaining domains, since it seems to increase performances based on our earlier findings. If time permits, we will include standard deviation of _F<sub>1</sub>_ from all four tasks, similar to what the paper provides. 

Our next __major__ goal is to investigate the effects of __fine-tuning__ original RoBERTa with the hope to find relatively universal methods across various domains (in our case, 4 domains) such that it would increase the performance of RoBERTa on specific domains that it was not originally pretrained on.

Furthermore, we are planning on carrying out error analysis as well as comparing results between fine-tuned RoBERTa and models that were pre-trained on specialized domains, once we make more progress in fine-tuning RoBERTa.
