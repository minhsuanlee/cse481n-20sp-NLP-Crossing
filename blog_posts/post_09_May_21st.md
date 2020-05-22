# Blog Post 9 - Due May 21st

### Team Name: NLP Crossing

### Team Members: Sam Lee, Shuheng Liu, Robin Yang

#### Advancing Solution Attempt

One advancement that we made in this week is regarding the discrepancy when running the HyperPartisan task. In the original experiment, as it was also confirmed by the author of the paper _Don't Stop Pre-training_, we concluded that the large difference between our results and the paper's results was due to the fact that the HyperPartisan dataset is relatively small, which causes the results to have a high standard deviation, and all our results were in the range of results obtained by the original authors of the paper. As the original results was presented in the form of mean<sub>sd</sub>, which was calculated by running experiments at "around 6 points over 5 seeds", we ran the experiments for five times for each pretrained model, each with a different random seed.

We tried to add one additional 1d convolutional layer at the end of the pre-trained model, then followed by a linear layer for prediction. We freeze the pre-trained model and train only on the two added layers. The model does not seem to be able to predict properly. We think that convolutional layers may be not very useful in a sentiment analysis model. The second attempt we tried was to add two linear layers at the end instead of 1. Initially the first layer had output dimension of `hidden` and the second layer had output dimension of `2` We hope to improve performance on a domain with simple training. The training took a long time so we decided to reduce the output dimension of the first layer to `hidden/2`. We currently have only finished first two epochs, the _F<sub>1</sub>_ scores on validation dataset are 69.4% and 79.9% respectively.

#### Additional Experiments

Here are the results of the HyperPartisan tasks:
<center>

##### Table 1: News (HyperPartisan)

|                 | _Don't Stop Pretraining_ |  Replication Study  |
| :-------------: | :----------------------: | :-----------------: |
|   __RoBERTa__   |    86.6<sub>0.9</sub>    | 87.2<sub>4.8</sub>  |
|    __DAPT__     |    88.2<sub>5.9</sub>    | 84.4<sub>10.6</sub> |
|    __TAPT__     |    90.4<sub>5.2</sub>    | 80.8<sub>8.6</sub>  |
| __DAPT + TAPT__ |    90.0<sub>6.6</sub>    | 86.7<sub>5.4</sub>  |

##### Table 2: Results Breakdown

| Exp. no. | RoBERTa | DAPT  | TAPT  | DAPT + TAPT |
| :------: | :-----: | :---: | :---: | :---------: |
|    1     |  86.8   | 93.6  | 92.1  |    85.3     |
|    2     |  91.8   | 92.1  | 68.9  |    87.0     |
|    3     |  91.9   | 69.7  | 79.5  |    90.1     |
|    4     |  78.7   | 93.5  | 74.7  |    77.6     |
|    5     |  87.0   | 73.3  | 88.9  |    93.7     |

</center>

The results in Table 1 are presented in the format of mean<sub>sd</sub>, and Table 2 records the test _F<sub>1</sub>_ for each experiment that we ran. As we can see, the results for this task, most likely due to the size of the dataset, are fairly inconsistent and deviate from the mean by a large extent. We believe that the discrepancy that we observed from the last time can therefore be justified.

#### Additional Error Analysis

The original authors of the paper ran 6 points over 5 random seeds for the HyperPartisan News and reported their results. As tables shown above, for each model we ran 5 experiemnts with random seeds and calculated the mean and standard deviation of the _F<sub>1</sub>_ scores. Similar to the results reported by the authors of the paper, we also found that the standard deviations remained relatively large due to limited amount of data for the task. Our standard deviations, in comparsion, are generally larger than the ones reported in the paper, probably because we have not run enough number of trials.

#### Next Action Plan

We are ready to investigate further about the effects of __fine-tuning__ original RoBERTa. We attempt to fine-tune RoBERTa by adding a layer or more and train on only the newly included layer(s), while keeping other parameters frozen.

We expect two additional linear layers to improve the performance. However, compared to the original RoBERTa model, two linear layers may not be significant. We could add more layers, but it also means more training time. We would like to find a balance between the number of layers, parameter in the layers, and the training time.

Besides linear layers, we are also interested in adding attention layers. Attention seems to be a powerful tool that could further improve the model performance. We expect this appraoch to be time-consuming on both writing and running the code.