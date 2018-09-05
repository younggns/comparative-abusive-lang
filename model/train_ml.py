import os
import sys
import numpy as np
import argparse
import pickle

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import helper

def train(feature_level, n_gram_tuple, max_feature_length, classifier):
	splits = ["train", "valid", "test"]

	path = os.path.dirname(os.path.abspath(__file__))
	with open(path+"/../data/preprocessed/"+feature_level+"/data_splits.pkl", "rb") as f:
		_data = pickle.load(f)

	for index in range(10):
		output_path = path + "/../data/output/"+feature_level+"/"+classifier+"/"+str(index)

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

		if not os.path.exists(output_path):
			os.makedirs(output_path)

		with open(output_path+"/pred_output.pkl", "wb") as f:
			print("Fold number:",str(index),"/ Writing prediction output pickle files to",output_path)
			pickle.dump({"pred_scores": pred_scores, "labels": label_data["test"]}, f)

	print("Done train/test.")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--feature_level", type=helper.str2feature, default='word')
	parser.add_argument("--ngram_range", type=helper.str2ngrams, required=True)
	parser.add_argument("--max_features", type=int, required=True)
	parser.add_argument("--clf", type=helper.str2clfs, required=True)

	args = vars(parser.parse_args())

	train(args["feature_level"], args["ngram_range"], args["max_features"], args["clf"])
