import os
import sys
import numpy as np
import argparse
import pickle
import re

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

import warnings
from sklearn.metrics import classification_report

import pandas as pd
import functools

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import helper
import time

def train(feature_level, n_gram_tuple, max_feature_length, classifier, output_path, k):
	splits = ["train", "valid", "test"]

	path = os.path.dirname(os.path.abspath(__file__))
	with open(path+"/../data/data_splits.pkl", "rb") as f:
		_data = pickle.load(f)

	for index in range(k):
		print("Fold number:",str(index))

		_text_data = _data["text_data"][index]
		_label_data = _data["label_data"][index]

		text_data = {key:[] for key in splits}
		label_data = {key:[] for key in splits}
		for split in splits:
			for text in _text_data[split]:
				text_data[split].append(text)

			for label in _label_data[split]:
				label_index = label.tolist().index(1) # convert 1-hot encoded label vector into an integer value.
				label_data[split].append(label_index)

		if classifier == 'NB':
			text_clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(n_gram_tuple[0],n_gram_tuple[1]), analyzer=feature_level, max_features=max_feature_length)),
							('clf', MultinomialNB())])

		elif classifier == 'LR':
			text_clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(n_gram_tuple[0],n_gram_tuple[1]), analyzer=feature_level, max_features=max_feature_length)),
							('clf', LogisticRegression(multi_class="multinomial", solver="lbfgs"))])

		elif classifier == 'SVM':
			text_clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(n_gram_tuple[0],n_gram_tuple[1]), analyzer=feature_level, max_features=max_feature_length)),
							('clf', SGDClassifier(loss='log', penalty='l2'))])

		elif classifier == 'RF':
			text_clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(n_gram_tuple[0],n_gram_tuple[1]), analyzer=feature_level, max_features=max_feature_length)),
							('clf', RandomForestClassifier())]) 

		elif classifier == 'GBT':
			text_clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(n_gram_tuple[0],n_gram_tuple[1]), analyzer=feature_level, max_features=max_feature_length)),
							('clf', GradientBoostingClassifier(learning_rate=1, max_depth=1))])
		else:
			print("Invalid classifier input.")
			exit()

		print("Fitting the data....")
		text_clf.fit(text_data["train"], label_data["train"])
		print("Output prediction on test data....")
		pred_scores = text_clf.predict_proba(text_data["test"])

		with open(output_path+"/pred_output_"+str(index)+".pkl", "wb") as f:
			print("Writing prediction output pickle files....")
			pickle.dump({"pred_scores": pred_scores, "labels": label_data["test"]}, f)

	print("Done training.")

def report_average(report_list, labels):

	output_report_list = list()
	for report in report_list:
		splitted = [' '.join(x.split()) for x in report.split('\n\n')]
		header = [x for x in splitted[0].split(' ')]
		data = np.array(splitted[1].split(' ')).reshape(-1, len(header) + 1)
		masked_data = np.array([[l,'0','0','0','0'] for l in labels])
		for i, label in enumerate(labels):
			if label not in data:
				data = np.insert(data, i, masked_data[i], axis=0)
		data = np.delete(data, 0, 1).astype(float)
		# avg_total = np.array([x for x in splitted[2].split(' ')][3:]).astype(float).reshape(-1, len(header))
		avg_total = np.array([x for x in splitted[2].split('weighted avg ')[1].split(' ')]).astype(float).reshape(-1, len(header))
		df = pd.DataFrame(np.concatenate((data, avg_total)), columns=header)
		output_report_list.append(df)
	res = functools.reduce(lambda x, y: x.add(y, fill_value=0), output_report_list) / len(output_report_list)
	metric_labels = labels + ['avg / total']
	return res.rename(index={res.index[idx]: metric_labels[idx] for idx in range(len(res.index))})

def evaluate(feature_level, classifier, output_path, k, labels):
	path = os.path.dirname(os.path.abspath(__file__))
	report_list = []

	for index in range(k):

		with open(output_path+"/pred_output_"+str(index)+".pkl", "rb") as f:
			_data = pickle.load(f)

		preds, target = _data["pred_scores"], _data["labels"]
		pred_np = np.argmax(preds, axis=1)

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			_report = classification_report(target, pred_np, digits=4, target_names=labels)
		report_list.append(_report)

	tot_report = report_average(report_list, labels)
	print(tot_report)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--feature_level", type=helper.str2feature, default='word')
	parser.add_argument("--clf", type=helper.str2clfs, required=True)
	parser.add_argument("--num_splits", type=int, default=10)

	args = vars(parser.parse_args())
	
	ngram_range_str = input("{:25}".format("ngram range (n1,n2):"))
	ngram_range = helper.str2ngrams(ngram_range_str)
	max_features = input("{:25}".format("max number of features:"))

	path = os.path.dirname(os.path.abspath(__file__))
	model_name = args['clf']+'_'+args["feature_level"]
	output_path = path + "/../data/output/"+model_name+"/"+str(int(time.time()))
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	data_path = os.path.dirname(os.path.abspath(__file__)) + "/../data/"
	with open(data_path+"crawled_data.pkl", "rb") as f:
		id2entities = pickle.load(f)

	labels = list(set([entity[0] for entity in id2entities.values()]))

	train(args["feature_level"], ngram_range, int(max_features), args["clf"], output_path, args["num_splits"])
	print("Evaluating the test results...")
	evaluate(args["feature_level"], args["clf"], output_path, args["num_splits"], labels)
