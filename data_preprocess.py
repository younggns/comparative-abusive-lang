import os
import pickle
import argparse
# For text preprocessing
import re
import numpy as np
from nltk.tokenize import TweetTokenizer
from wordsegment import segment, load

# K-fold splits
from sklearn.model_selection import train_test_split, KFold

################################################################################
############################## Text Preprocessing ##############################
################################################################################
def text_preprocess(text, tknzr):
	FLAGS = re.MULTILINE | re.DOTALL
	# Different regex parts for smiley faces
	eyes = r"[8:=;]"
	nose = r"['`\-]?"

	# function so code less repetitive
	def re_sub(pattern, repl):
		return re.sub(pattern, repl, text, flags=FLAGS)

	text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
	text = re_sub(r"/"," / ")
	text = re_sub(r"@\w+", "<user>")
	text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
	text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
	text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
	text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
	text = re_sub(r"<3","<heart>")
	text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
	text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
	text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
	text = re_sub(r"#\S+", lambda hashtag: " ".join(segment(hashtag.group()[1:]))) # segment hastags

	tokens = tknzr.tokenize(text.lower())
	return " ".join(tokens)

def concat_data():
	path = os.path.dirname(os.path.abspath(__file__)) + "/data/"
	with open(path+"crawled_data.pkl", "rb") as f:
		id2entities = pickle.load(f)

	########## Lookup Tables ##########
	labels = list(set([entity[0] for entity in id2entities.values()]))
	num_classes = len(labels)

	label_lookup = np.zeros((num_classes,num_classes),int)
	np.fill_diagonal(label_lookup, 1)
	###################################

	text_data, context_data, label_data = [], [], []
	label_dict = {}
	for i, label in enumerate(labels):
		label_dict[label] = i

	load()
	tknzr = TweetTokenizer(reduce_len=True, preserve_case=False, strip_handles=False)
	print("Preprocessing tweets.....")
	for _id in id2entities:
		if id2entities[_id][0] in label_dict.keys():
			text_data.append(text_preprocess(id2entities[_id][1], tknzr))
			context_data.append(text_preprocess(id2entities[_id][2], tknzr))

			label_data.append(label_lookup[ label_dict[id2entities[_id][0]] ])

	assert len(text_data) == len(context_data) == len(label_data)

	return text_data, context_data, label_data

################################################################################
############################## K-fold Data Split ###############################
################################################################################
def kfold_splits(text_data, context_data, label_data, k):
	kfold_text, kfold_context, kfold_label = [], [], []
	for i in range(k):
		_text_data = {"train": {}, "valid": {}, "test": {}}
		_context_data = {"train": {}, "valid": {}, "test": {}}
		_label_data = {"train": {}, "valid": {}, "test": {}}
		kfold_text.append(_text_data)
		kfold_context.append(_context_data)
		kfold_label.append(_label_data)

	kf = KFold(n_splits=k, shuffle=True, random_state=0)
	kfold_index = 0
	for rest_index, test_index in kf.split(text_data):

		train_index, valid_index, _, _ = train_test_split(rest_index, np.zeros_like(rest_index), test_size=0.05)

		kfold_text[kfold_index]["train"] = [text_data[index] for index in train_index]
		kfold_text[kfold_index]["test"] = [text_data[index] for index in test_index]
		kfold_text[kfold_index]["valid"] = [text_data[index] for index in valid_index]

		kfold_context[kfold_index]["train"] = [context_data[index] for index in train_index]
		kfold_context[kfold_index]["test"] = [context_data[index] for index in test_index]
		kfold_context[kfold_index]["valid"] = [context_data[index] for index in valid_index]

		kfold_label[kfold_index]["train"] = [label_data[index] for index in train_index]
		kfold_label[kfold_index]["test"] = [label_data[index] for index in test_index]
		kfold_label[kfold_index]["valid"] = [label_data[index] for index in valid_index]

		assert len(kfold_text[kfold_index]["train"]) == len(kfold_context[kfold_index]["train"]) == len(kfold_label[kfold_index]["train"])
		assert len(kfold_text[kfold_index]["valid"]) == len(kfold_context[kfold_index]["valid"]) == len(kfold_label[kfold_index]["valid"])
		assert len(kfold_text[kfold_index]["test"]) == len(kfold_context[kfold_index]["test"]) == len(kfold_label[kfold_index]["test"])

		train_length = len(kfold_text[kfold_index]["train"])
		valid_length = len(kfold_text[kfold_index]["valid"])
		test_length = len(kfold_text[kfold_index]["test"])

		kfold_index += 1

	print("Input Data Splitted: %s (train) / %s (valid) / %s (test)" % (train_length, valid_length, test_length))
	
	return kfold_text, kfold_context, kfold_label

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument("--num_splits", type=int, default=10)
	args = vars(parser.parse_args())

	_text, _ctxt, _label = concat_data()

	print("Splitting data into", args['num_splits'], "folds.....")
	_text_split, _ctxt_split, _label_split = kfold_splits(_text, _ctxt, _label, int(args['num_splits']))

	path = os.path.dirname(os.path.abspath(__file__)) + "/data/"
	if not os.path.exists(path):
		os.makedirs(path)
	with open(path+"data_splits.pkl", "wb") as f:
		print("Creating pickle files for each split to /data/data_splits.pkl")
		pickle.dump({"text_data": _text_split, "context_data": _ctxt_split, "label_data": _label_split}, f)
