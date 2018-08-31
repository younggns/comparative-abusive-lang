*Still working with code distribution. Please contact me (younggnse@gmail.com) if you have any questions.*
# Comparative Studies of Detecting Abusive Language on Twitter

Python 3.6.0 implementation of ["Comparative Studies of Detecting Abusive Language on Twitter"](http://arxiv.org/abs/1808.10245) accepted in EMNLP 2018 Workshop on Abusive Language Online (ALW2).
This paper conducts a comparative study of various learning models on [Hate and Abusive Speech on Twitter](https://github.com/ENCASEH2020/hatespeech-twitter) dataset, which has great potential in training deep models with its significant size.

## Requirements
The following script will install required Python packages.

```
pip3 install -r requirements.txt
```

## Data Preprocessing
Data preprocessing consists of three steps:
1. Download the source data (*Hate and Abusive Speech on Twitter*). The dataset's each row is consisted of a tweet ID and its annotation. 
2. Crawl tweets based on tweet ID. Note that **you might not be able to crawl full data of tweets with tweet IDs**, mainly because the tweet is deleted or the user account has been suspended. Please fill in your Twitter app credentials from line 57 in the [./data_preprocess.py](./data_preprocess.py) script.
3. Preprocess raw text data---basic text preprocessing (e.g. handling hashtags and URLs) and splitting the data into 10 randomly divided folds.

Run the following script with specified parameters:

```
python3 data_preprocess.py
	--download_source (Download the source file from the Github repository if True; Type: boolean; default: True)
	--crawl_tweets (Crawl tweets with tweet IDs. This procedure also creates a Python pickle file of the data; Type: boolean; default: True)
	--feature_level (Feature representation level. Either 'word' or 'char'; Type: char; default: 'word')
```

## Training
### Traditional Machine Learning Models
In order to report the baselines of feature engineering based machine learning models, we experimented with Naïve Bayes, Logistic Regression, Support Vector Machine, Random Forests, and Gradient Boosted Trees. All models were implemented using Scikit-learn packages.

In this paper, we tune n-gram ranges and maximum length of features as hyperparameters. Please modify the code [./model/train_ml.py](./model/train_ml.py) in order to try different settings (e.g. loss function, learning rate, etc.)

Go to the [./model](./model) directory and run the following script with specified parameters.

```
python3 train_ml.py
	--feature_level (Feature representation level. Either 'word' or 'char'; Type: char; default: 'word')
	--ngram_range (Range of n-grams for training. E.g. '1,3' for using 1-gram to 3-grams; Type: char)
	--max_features (Maximum length of features that you want to use. Features are selected based on TF-IDF normalized values; Type: int)
```

## Reference
Please refer the following paper in order to use the code as part of any publications.

```
@article{lee2018comparative,
	title={Comparative Studies of Detecting Abusive Language on Twitter},
	author={Lee, Younghun and Yoon, Seunghyun and Jung, Kyomin},
	journal={arXiv preprint arXiv:1808.10245},
	year={2018}
}
```
