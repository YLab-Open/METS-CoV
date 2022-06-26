# METS-CoV: A Dataset of Medical Entity and Targeted Sentiment on COVID-19 Related Tweets

This repository includes the dataset and benchmark of the paper:

**METS-CoV: A Dataset of Medical Entity and Targeted Sentiment on COVID-19 Related Tweets (Submitted to NeurIPS 2022 Track on Datasets and Benchmarks).**

**Authors**: Peilin Zhou, Zeqiang Wang, Dading Chong, Zhijiang Guo, Yining Hua, Zichang Su, Zhiyang Teng, Jiageng Wu, Jie Yang


## Abstract:
The COVID-19 pandemic continues to bring up various topics discussed or debated on social media. In order to explore the impact of pandemics on people's lives, it is crucial to understand the public's concerns and attitudes towards pandemic-related entities (e.g., drugs, vaccines) on social media. However, models trained on existing named entity recognition (NER) or targeted sentiment analysis (TSA) datasets have limited ability to understand COVID-19-related social media texts because these datasets are not designed or annotated from a medical perspective. In this paper, we release METS-CoV, a dataset containing medical entities and targeted sentiments from COVID-19 related tweets. METS-CoV contains 10,000 tweets with 7 types of entities, including 4 medical entity types (_Disease_, _Drug_, _Symptom_, and _Vaccine_) and 3 general entity types (_Person_, _Location_, and _Organization_). To further investigate tweet users' attitudes toward specific entities, 4 types of entities (_Person_, _Organization_, _Drug_, and _Vaccine_) are selected and annotated with user sentiments, resulting in a targeted sentiment dataset with 9,101 entities (in 5,278 tweets). To the best of our knowledge, METS-CoV is the first dataset to collect medical entities and corresponding sentiments of COVID-19 related tweets. We benchmark the performance of classical machine learning models and state-of-the-art deep learning models on NER and TSA tasks with extensive experiments. Results show that this dataset has vast room for improvement for both NER and TSA tasks. With rich annotations and comprehensive benchmark results, we believe METS-CoV is a fundamental resource for building better medical social media understanding tools and facilitating computational social science research, especially on epidemiological topics. Our data, annotation guidelines, benchmark models, and source code are publicly available [link](https://github.com/YLab-Open/METS-CoV) to ensure reproducibility. 

![avatar](figs/tweet_annotation_sample.png)

## Dataset

加时间信息。
Twitter users express their attitudes toward medical entities (e.g., drugs and vaccines) through their Twitter posts.
In this work, we introduce METS-CoV, the first dataset to include medical entities and targeted sentiments on COVID-19-related tweets. 

### Data Statistics
The figure below shows the distribution of tweet lengths in METS-CoV. Most tweets have lengths between 20 and 60 tokens (split with white spaces). 

![avatar](figs/data_len_distribution.png)

The table below shows the statistics of the METS-CoV-NER dataset. 10,000 tweets contain 19,057 entities in total, resulting in 1.91 entities per tweet in this dataset. We observe that _Symptom_ entities have the highest frequency. This is an expected result of pre-filtering tweets using symptom-related keywords (Some matched symptom keywords are not real symptom entities in tweets due to the ambiguity of words, that is why the total number of Symptom entities is less than 10,000). Other than _symptoms_, all other 6 entity types have relatively balanced proportions. 

| **Number**       | **Train** | **Dev** | **Test** |
|------------------|-----------|---------|----------|
| **Tweets**       | 7,000     | 1,500   | 1,500    |
| **Tokens**       | 285k      | 59k     | 60k      |
| **All Entities** | 13,570    | 2,749   | 2,738    |
| **Person**       | 2,253     | 472     | 487      |
| **Location**     | 1,371     | 294     | 279      |
| **Organization** | 1,934     | 396     | 381      |
| **Disease**      | 1,555     | 301     | 258      |
| **Drug**         | 1,077     | 262     | 227      |
| **Symptom**      | 4,223     | 806     | 869      |
| **Vaccine**      | 1,157     | 218     | 237      |

Table below presents the statistical information of the METS-CoV-TSA dataset, where we observe that neutral sentiment accounts for the highest proportion across all 4 target entities. For the _drug_ entities, users have significantly more positive sentiments than negative sentiments, whereas for the _vaccine_ entities, users have similar positive and negative sentiments.

| Number       |     | Train | Dev | Test |
|--------------|-----|-------|-----|------|
| Person       | POS | 260   | 64  | 58   |
|              | NEU | 1293  | 256 | 240  |
|              | NEG | 700   | 152 | 189  |
| Organization | POS | 126   | 24  | 31   |
|              | NEU | 1346  | 284 | 251  |
|              | NEG | 462   | 88  | 99   |
| Drug         | POS | 234   | 85  | 64   |
|              | NEU | 730   | 147 | 142  |
|              | NEG | 113   | 30  | 21   |
| Vaccine      | POS | 112   | 25  | 20   |
|              | NEU | 913   | 173 | 183  |
|              | NEG | 132   | 20  | 34   |
### Data Access
Following Twitter's automation rules and data security policy, we can not directly provide the original tweets to dataset consumer. 
Therefore, the Tweet IDs and corresponding annotations are released to the public. 
Based on the Tweet IDs provided by us, the dataset consumers could download the original tweets freely via official Twitter API by themselves.

Here we also provide the script `download_tweets.py` to help users to download tweets using Tweet IDs. (Before downloading, users need to apply for a Twitter developer account first.)

We perform a train-dev-test splitting of our dataset with a ratio of 70:15:15.

Training, validation and test data are available in `dataset` folder, which are saved in a CSV format.

| tweet_id            | labels                                                                                                                                                                                                                                                                                                                           |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1423080345067565057 |  "[{'type': 'Organization', 'start_index': 5, 'end_index': 6, 'value': 'NextDoor', 'sentiment': 'Negative'}, {'type': 'Drug', 'start_index': 13, 'end_index': 15, 'value': 'MLM vitamin', 'sentiment': 'Negative'}]"
| 1278592113878409216 |  "[{'type': 'Disease', 'start_index': 3, 'end_index': 5, 'value': 'sleep apnea', 'sentiment': 'null'}]"

### Annotation Guidelines

Our annotation guidelines are customized to understand medical-related entities and sentiments better. NER guidelines include rules such as "when the manufacturer name refers to a vaccine in the tweet context, the name should be annotated as a vaccine rather than an organization". Sentiment guidelines include rules such as "when irony appears in an annotation entity, its sentiment should be annotated from the standpoint of the person who posted the tweet". 
More details can be found in our guidelines (see `annotation guidelines/`).  Both Chinese version and English version are released/



## Model Benchmarking 

### Preprocessing
After downloading the tweets, users need to preprocess the original tweets using following script:
```console
❱❱❱ python preprocess.py --input_path=xxx.csv --output_path=xxx.csv
```
* --input_path: the path of input csv file, the column name of the tweets should be 'full_text'.
* --output_path: the path to save preprocessed csv file

### NER
To reproduce the results of CRF model, users could use the [sklearn-crfsuite](https://sklearn-crfsuite.readthedocs.io/en/latest/).

All the experiments of NN-based and PLM-based NER models are conducted using [NCRF++]().
The config files of all these models are available in `NCRFpp_configs/` folder.

Users could check the documentation of NCRF++ to learn how to use these configs to reproduce the  benchmark results.

The link to the pretrained GloVe vectors is here:
* **[`GloVe: Global Vectors for Word Representation`](https://github.com/stanfordnlp/GloVe)**

The links to the PLM models are here:

* **[`bert-base-uncased`](https://huggingface.co/bert-base-uncased)**
* **[`bert-base-cased`](https://huggingface.co/bert-base-cased)**
* **[`bert-large-uncased`](https://huggingface.co/bert-large-uncased)**
* **[`bert-large-uncased`](https://huggingface.co/bert-large-uncased)**
* **[`RoBERTa-base`](https://huggingface.co/roberta-base)**
* **[`RoBERTa-large`](https://huggingface.co/roberta-large)**
* **[`BART-base`](https://huggingface.co/facebook/bart-base)**
* **[`BART-large`](https://huggingface.co/facebook/bart-large)**
* **[`BERTweet-covid19-base-uncased`](https://huggingface.co/vinai/bertweet-covid19-base-uncased)**
* **[`BERTweet-covid19-base-cased`](https://huggingface.co/vinai/bertweet-covid19-base-cased)**
* **[`COVID-TWITTER-BERT`](https://huggingface.co/digitalepidemiologylab/covid-twitter-bert-v2)**

#### Leaderboard
We reported the results, namely mean ± std, based on experiments on 5 different random seeds: 22, 32, 42, 52 and 62. Mean is calculated by averaging the performances of different seeds. Std denotes the standard error, which is calculated by dividing the standard deviation of the mean value by the square root of the number of seeds.

Note: * means uncased model.

 **Results (F1 value ± std)** | **Person** | **Location** | **Organization** | **Disease** | **Drug**   | **Symptom** | **Vaccine** | **Overall** 
------------------------------|------------|--------------|------------------|-------------|------------|-------------|-------------|-------------
 CRF                          | 64.43±1.59 | 76.37±0.62   | 54.64±2.08       | 73.61±0.44  | 77.34±1.60 | 74.05±0.56  | 84.85±0.82  | 71.58±0.54  
 WLSTM                        | 72.05±0.79 | 79.82±0.61   | 60.79±0.77       | 73.52±1.26  | 79.63±1.36 | 76.72±0.83  | 86.03±0.84  | 75.02±0.36  
 WLSTM + CCNN                 | 80.63±0.62 | 81.47±0.89   | 61.30±0.91       | 74.52±0.75  | 80.46±0.28 | 76.63±0.91  | 85.91±1.17  | 76.78±0.29  
 WLSTM + CLSTM                | 81.16±0.29 | 81.37±0.48   | 62.28±1.41       | 74.80±1.49  | 79.50±1.01 | 76.70±0.25  | 85.59±1.46  | 76.91±0.22  
 WLSTM + CRF                  | 72.41±0.44 | 79.25±0.59   | 62.39±0.97       | 74.89±1.28  | 79.60±0.71 | 79.14±0.51  | 88.72±0.62  | 76.38±0.22  
 WLSTM + CCNN + CRF           | 81.38±0.44 | 82.15±0.44   | 62.79±0.91       | 76.12±0.76  | 80.41±0.58 | 78.12±0.51  | 89.11±0.36  | 78.10±0.19  
 WLSTM + CLSTM + CRF          | 77.49±1.67 | 81.26±1.19   | 63.21±0.93       | 75.61±0.76  | 81.27±0.65 | 79.14±0.53  | 87.85±0.43  | 77.63±0.40  
 BERT-base*                   | 86.99±1.27 | 84.47±0.84   | 71.01±0.52       | 76.18±1.49  | 84.78±1.02 | 80.26±0.43  | 89.83±0.56  | 81.49±0.36  
 BERT-base                    | 86.71±0.73 | 84.09±1.67   | 71.90±0.91       | 76.93±1.11  | 84.54±1.05 | 79.70±1.22  | 89.48±0.99  | 81.34±0.41  
 BERT-large*                  | 87.47±1.22 | 84.43±1.05   | 71.21±0.59       | 76.34±0.96  | 85.53±1.52 | 81.44±0.18  | 89.33±1.25  | 81.98±0.30  
 BERT-large                   | 88.25±0.52 | 84.63±1.38   | 73.30±1.38       | 76.52±1.11  | 86.05±1.06 | 80.12±0.55  | 89.16±1.58  | 82.05±0.24  
 RoBERTa-base                 | 85.58±0.73 | 85.46±0.86   | 72.21±1.11       | 76.49±1.63  | 85.38±1.15 | 79.81±0.67  | 89.89±0.28  | 81.43±0.20  
 RoBERTa-large                | 86.79±0.44 | 85.85±2.12   | 73.78±0.72       | 76.84±0.57  | 86.79±0.78 | 81.32±0.67  | 90.42±1.12  | 82.55±0.27  
 BART-base                    | 84.24±0.59 | 82.85±0.71   | 70.60±1.46       | 75.01±1.80  | 83.39±1.03 | 79.03±0.48  | 90.22±1.58  | 80.17±0.46  
 BART-large                   | 81.60±4.93 | 80.04±4.74   | 64.66±8.86       | 71.24±1.90  | 80.61±2.90 | 74.27±4.45  | 81.21±6.20  | 75.56±5.04  
 BERTweet-covid19-base*       | 91.63±0.79 | 85.79±0.75   | 77.07±0.51       | 77.09±1.61  | 83.57±0.65 | 81.16±0.45  | 91.16±1.54  | 83.63±0.36  
 BERTweet-covid19-base        | 91.50±0.81 | 86.26±0.97   | 76.47±0.46       | 77.80±0.57  | 84.16±1.22 | 80.89±0.52  | 89.98±1.04  | 83.49±0.18  
 COVID-TWITTER-BERT*          | 91.29±0.42 | 85.68±0.92   | 76.27±0.64       | 77.48±0.81  | 86.35±0.96 | 81.85±0.53  | 90.44±0.94  | 83.88±0.20  

### TSA

To reproduce the results of LSTM, TD-LSTM, MemNet, IAN, MGAN, TNET-LF, ASGCN, AEN and LCF,  users could use [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch).

To reproduce the results of BERT-SPC, depGCN, kumaGCN and dotGCN, please use the code of [pro-absa](https://github.com/zeeeyang/pro-absa).



#### Leaderboard
We reported the results, namely mean ± std, based on experiments on 5 different random seeds: 22, 32, 42, 52 and 62. Mean is calculated by averaging the performances of different seeds. Std denotes the standard error, which is calculated by dividing the standard deviation of the mean value by the square root of the number of seeds.

|                              | Person              |                     | Organization   |                     | Drug                |                     | Vaccine             |                     | Overall             |                      |
|------------------------------|---------------------|---------------------|----------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|----------------------|
| Model                        | Acc                 | F1                  | Acc            | F1                  | Acc                 | F1                  | Acc                 | F1                  | Acc                 | F1                   |
| SVM                          | 50.88               | 46.43               | 56.79          | 40.75               | 47.25               | 32.85               | 66.06               | 44.1                | 55.17               | 42.96                |
| LSTM                         | 57.35±1.85          | 42.06±3.69          | 69.14±2.68     | 45.06±2.41          | 57.69±3.50          | 38.55±0.78          | 72.02±4.09          | 38.40±3.40          | 64.00±1.74          | 44.11±0.73           |
| TD-LSTM                      | 60.00±1.31          | 47.46±2.39          | **70.37±1.06** | 43.87±2.72          | 62.64±1.82          | 40.54±3.94          | **79.36±2.92** | 42.20±2.37          | **67.58±0.59** | 51.66±2.24           |
| MemNet                       | **61.18±1.19** | 49.07±2.01          | 67.59±1.68     | 42.85±1.81          | **64.84±4.34** | **46.84±3.68** | **79.36±1.44** | 40.98±1.68          | 67.48±1.47          | 49.11±1.34           |
| IAN                          | 58.82±0.82          | 40.80±1.00          | 66.67±1.09     | 40.14±2.12          | 62.64±3.14          | 34.57±6.49          | **79.36±1.65** | 37.02±2.48          | 66.07±0.91          | 42.36±2.65           |
| MGAN                         | 59.71±1.99          | 41.30±1.93          | 67.90±1.35     | 41.22±1.47          | 59.34±3.98          | 32.16±7.06          | 77.52±2.88          | 31.39±3.42          | 65.79±0.67          | 42.51±0.99           |
| TNet-LF                      | 60.29±2.31          | **54.33±2.96** | 66.98±2.53     | **56.16±3.64** | 50.00±4.19          | 36.98±2.91          | 66.97±2.12          | **43.35±2.89** | 61.94±1.15          | **54.45±1.82**  |
| ASGCN                        | 60.88±1.72          | 50.08±2.05          | 67.59±2.00     | 46.82±5.32          | 62.64±2.10          | 44.63±3.94          | 75.69±1.39          | 42.74±5.31          | 66.26±1.07          | 49.90±2.70           |
| AEN                          | 60.88±1.88          | 53.93±2.98          | 70.37±2.67     | **60.60±2.21** | 51.65±7.18          | 42.41±4.33          | 69.27±4.06          | 45.06±3.86          | 63.91±2.37          | 54.27±0.65           |
| LCF                          | 63.53±2.56          | 55.15±2.74          | 72.22±2.01     | 56.28±2.32          | 59.34±3.97          | 50.53±2.86          | 75.23±3.47          | 39.62±5.21          | 67.86±1.18          | 56.64±2.47           |
| BERT-SPC                     | 63.82±1.65          | 56.89±3.15          | 73.15±0.70     | 52.79±3.93          | 64.29±1.33          | 48.62±2.66          | 77.52±1.86          | 41.90±4.90          | 69.55±0.36          | 56.19±1.56           |
| depGCN                       | **65.00±1.51** | **58.71±2.33** | **75.00±1.20** | 59.47±1.03          | 64.29±1.71          | 52.54±0.96          | 77.06±2.39          | **48.30±3.73** | **70.39±0.99** | **58.56±1.13**  |
| kumaGCN                      | 63.53±2.01          | 56.94±3.15          | 71.91±1.43     | 57.16±1.17          | 64.29±1.61          | 45.69±1.39          | **79.82±1.66** | 47.50±3.86          | 69.55±1.31          | 55.70±2.15           |
| dotGCN                       | 62.94±1.27          | 57.25±1.71          | 72.22±1.36     | 56.63±2.02          | **66.48±1.17** | **55.88±2.31** | 78.44±1.85          | 44.78±3.61          | 69.55±0.69          | 57.95±0.65           |
| BERT-SPC(COVID-TWIITER-BERT) | 67.35±1.87          | 61.37±1.97          | **78.70±0.90** | **65.57±2.40** | 68.13±2.24          | 49.05±3.83          | 76.61±3.05          | 50.37±5.05          | 72.84±0.73          | 62.16±2.19           |
| depGCNC(COVID-TWIITER-BERT)  | **75.29±2.29** | **70.43±3.10** | **78.70±1.20** | 62.16±4.47          | 75.82±1.27          | 67.20±4.00          | 75.23±1.97          | 54.80±2.05          | 76.41±0.51          | **69.91±1.53**  |
| kumaGCNC(COVID-TWIITER-BERT) | 75.00±1.93          | 70.12±3.70          | 77.47±1.23     | 62.62±3.62          | 74.18±1.95          | 63.07±4.56          | **81.65±2.00** | **57.18±1.14** | **76.97±1.23** | 69.31±1.52           |
| dotGCNC(COVID-TWIITER-BERT)  | 70.88±3.48          | 66.88±5.09          | **78.70±1.89** | 62.03±4.61          | **79.12±2.04** | **68.04±3.73** | 79.36±2.64          | 54.63±2.90          | 76.41±1.62          | 69.07±3.59           |

## Contact

If you have any questions about our dataset or benchmark results, please feel free to contact us!
(Peilin Zhou, zhoupalin@gmail.com; Jie Yang, jieynlp@gmail.com)
