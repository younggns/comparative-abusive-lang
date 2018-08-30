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
1. Download the source data (*Hate and Abusive Speech on Twitter*). The dataset's each row is consisted of a tweet ID and its annotation. 
2. Crawl tweets based on tweet ID. Note that **you might not be able to crawl full data of tweets with tweet IDs**, mainly because the tweet is deleted or the user account has been suspended. Please fill in your Twitter app credentials from line 57 in the [data_preprocess.py](./data_preprocess.py) script.
3. Preprocess raw text data---basic text preprocessing (e.g. handling hashtags and URLs) and splitting the data into 10 randomly divided folds.

Run the following script with specified parameters:

```
python3 data_preprocess.py
	--download_source (Download the source file from the Github repository if True.; Type: boolean; default: True)
	--crawl_tweets (Crawl tweets with tweet IDs. This procedure also creates a Python pickle file of the data; Type: boolean; default: True)
	--feature_level (Feature representation level. Either 'word' or 'char'; Type: char; default: 'word')
```

## Training
### Traditional Machine Learning Models
In order to report the baselines of feature engineering based machine learning models, we experimented with Naïve Bayes, Logistic Regression, Support Vector Machine, Random Forests, and Gradient Boosted Trees. All models were implemented using Scikit-learn packages.

Go to the [model](./model) directory and run the following script with specified parameters.

### Neural Network Models

this step includes generating vocabulary dictionaries (e.g. word to index), and converting each tweet message into numpy object with its feature representations for neural network models.

--use_pretrained_embedding ('yes' if you want to use GloVe embedding as in this paper. 'no' will use random word embeddings; default: 'yes')
