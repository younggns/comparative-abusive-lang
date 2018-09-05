import os
import sys
import numpy as np
import argparse
import pickle

import warnings
from sklearn.metrics import classification_report

import pandas as pd
import functools

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import helper

################################################
labels = ["normal", "spam", "hateful", "abusive"]
################################################

def report_average(report_list):
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
		avg_total = np.array([x for x in splitted[2].split(' ')][3:]).astype(float).reshape(-1, len(header))
		df = pd.DataFrame(np.concatenate((data, avg_total)), columns=header)
		output_report_list.append(df)
	res = functools.reduce(lambda x, y: x.add(y, fill_value=0), output_report_list) / len(output_report_list)
	metric_labels = labels + ['avg / total']
	return res.rename(index={res.index[idx]: metric_labels[idx] for idx in range(len(res.index))})

def evaluate(feature_level, classifier):
	path = os.path.dirname(os.path.abspath(__file__))
	report_list = []

	for index in range(10):
		output_path = path + "/../data/output/"+feature_level+"/"+classifier+"/"+str(index)

		with open(output_path+"/pred_output.pkl", "rb") as f:
			_data = pickle.load(f)

		preds, target = _data["pred_scores"], _data["labels"]
		pred_np = np.argmax(preds, axis=1)

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			_report = classification_report(target, pred_np, digits=4, target_names=labels)
		report_list.append(_report)

	tot_report = report_average(report_list)
	print(tot_report)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--feature_level", type=helper.str2feature, default='word')
	parser.add_argument("--clf", type=helper.str2clfs, required=True)

	args = vars(parser.parse_args())

	evaluate(args["feature_level"], args["clf"])