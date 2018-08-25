# Comparative Studies of Detecting Abusive Language on Twitter

Python 3.6.0 implementation of "Comparative Studies of Detecting Abusive Language on Twitter" accepted in EMNLP 2018 Workshop on Abusive Language Online (ALW2).
This paper conducts a comparative study of various learning models on [Hate and Abusive Speech on Twitter](https://github.com/ENCASEH2020/hatespeech-twitter) dataset, which has great potential in training deep models with its significant size.

## Requirements
The following script will install required Python packages.

```
pip3 install -r requirements.txt
```

## Data Preprocessing
Data preprocessing consists of three steps:
1. Download the data. The dataset's each row is consisted of a tweet ID and its annotation. 
2. Crawl tweets based on tweet ID. Note that **you might not be able to crawl full data of tweets with tweet IDs**, mainly because the tweet is deleted or the user account has been suspended.
3. Preprocess raw text data. This step includes text preprocessing (e.g. handling hashtags and URLs), generating vocabulary dictionaries (e.g. word to index), and converting each tweet message into numpy object with its feature representations.


