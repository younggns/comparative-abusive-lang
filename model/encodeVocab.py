import os
import sys
import argparse
import time
from datetime import datetime
import pickle
import numpy as np

from collections import Counter

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import helper
###################################################################################
############################## Word Vocab Generation ##############################
###################################################################################
def tokens2id(tokens, word2id, max_len):
	idx = []
	if tokens == ['']:
		for _ in range(max_len):
			idx.append(word2id["PAD"])
	else:
		for t in tokens[:max_len]:
			if t not in word2id.keys():
				idx.append(word2id["UNK"])
			else:
				idx.append(word2id[t])
		padding_needed = max_len - len(idx) if max_len > len(idx) else 0
		for _ in range(padding_needed):
			idx.append(word2id["PAD"])
	assert len(idx) == max_len
	return idx

def idx_representations_of_text(text_list, context_list, word2id, max_len, k):
	idx_data_list, splits = [], ["train", "valid", "test"]
	for i in range(k):
		idx_data = {}
		for split in splits:
			idx_data[split] = []
			for index, tokens in enumerate(text_list[i][split]):
				ctxt_idx = tokens2id(context_list[i][split][index], word2id, max_len)
				text_idx = tokens2id(text_list[i][split][index], word2id, max_len)
				idx_data[split].append(ctxt_idx + text_idx)
		idx_data_list.append(idx_data)
	return idx_data_list
	
def genWordVocab(text_data, context_data, k):
	splits = ["train", "valid", "test"]
	##### Convert tab-separated tweets into list of tokens #####
	text_word_list, ctxt_word_list = [], []
	for i in range(k):
		_text, _ctxt = {}, {}
		for split in splits:
			text_tokens = [tweet.rstrip().split() for tweet in text_data[i][split]]
			ctxt_tokens = [tweet.rstrip().split() for tweet in context_data[i][split]]
			_text[split] = text_tokens
			_ctxt[split] = ctxt_tokens
		text_word_list.append(_text)
		ctxt_word_list.append(_ctxt)

	##### Create Vocab (id2word, word2id) #####
	vocab = Counter()

	max_len = 0
	for i in range(k):
		for split in splits:
			for tokens in text_word_list[i][split]:
				vocab.update(tokens)
				if max_len < len(tokens):
					max_len = len(tokens)
			for tokens in ctxt_word_list[i][split]:
				vocab.update(tokens)
				if max_len < len(tokens):
					max_len = len(tokens)
	# max_len = 100 if max_len > 100 else max_len
	max_len = 100

	count = 0
	for word in vocab.keys():
		if vocab[word] >= 2: # only count the word with frequencies of at least 2
			count += 1
	vocab_size = count

	_vocab = ["PAD", "UNK"] # add special vocab for PAD and UNK
	for word, _ in vocab.most_common(vocab_size):
		_vocab.append(word)

	# create dictionaries
	id2word = {}
	word2id = {}
	for i, word in enumerate(_vocab):
		id2word[i] = word
		word2id[word] = i

	# Representing words in tweets into their indices
	ctxt_text_idx = idx_representations_of_text(text_word_list, ctxt_word_list, word2id, max_len, k)

	return id2word, word2id, ctxt_text_idx

def genWordEmbeddings(ues_glove, path):

	embedding_dim = 300
	with open(path+"/vocab.pkl", "rb") as f:
		vocab = pickle.load(f)
		vocab_size = len(vocab["word2id"].keys())
		print("Vocabulary loaded with %s words" % vocab_size)

	glove_embedding_matrix = np.zeros((vocab_size, embedding_dim))

	if ues_glove:
		glove = {}
		print("Loading pre-trained embedding.")
		f = open(path+"/../../glove.840B.300d.txt")
		for line in f:
			values = line.split()
			word = ' '.join(values[:-embedding_dim])
			coefs = np.asarray(values[-embedding_dim:], dtype='float32')
			glove[word] = coefs

		for word in vocab["word2id"]:
			if word == "PAD":
				glove_embedding_matrix[vocab["word2id"][word]] = np.zeros(embedding_dim)
			elif word in glove:
				glove_embedding_matrix[vocab["word2id"][word]] = glove[word]
			else:
				glove_embedding_matrix[vocab["word2id"][word]] = np.random.normal(0,0.01,embedding_dim)

		np.save(path+"/embedding.npy", glove_embedding_matrix)
	
	else:
		print("Generating random embedding.")
		for word in vocab["word2id"]:
			if word == "PAD":
				glove_embedding_matrix[vocab["word2id"][word]] = np.zeros(embedding_dim)
			else:
				glove_embedding_matrix[vocab["word2id"][word]] = np.random.normal(0,0.01,embedding_dim)
		np.save(path+"/embedding.npy", glove_embedding_matrix)

###################################################################################
############################## Char Vocab Generation ##############################
###################################################################################
def charEmbedding(text, max_len=140):
	FEATURES = list("abcdefghijklmnopqrstuvwxyz0123456789 \u2014,;.!?:\u201c\u201d’/|_@#$%ˆ&*~‘+-=<>()[]{}")
	_text = ""
	for c in list(text):
		if c in FEATURES:
			_text += c
	tokens = list(_text)
	vector = -np.ones(max_len)
	
	for i, t in enumerate(tokens):
		if i < max_len:
			try:
				j = FEATURES.index(t)
			except ValueError:
				j = -1
				print("value error")
			vector[i] = j

	return vector

def genCharVocab(text_data, context_data, k):
	_text_data, _ctxt_data = [], []
	splits = ["train", "valid", "test"]
	for index in range(k):
		_text = {"train":[], "valid":[], "test":[]}
		_ctxt = {"train":[], "valid":[], "test":[]}
		_text_data.append(_text)
		_ctxt_data.append(_ctxt)
	for index in range(k):
		for split in splits:
			for text in text_data[index][split]:
				_vector = charEmbedding(text)
				_text_data[index][split].append(_vector)
			for context in context_data[index][split]:
				_vector = charEmbedding(context)
				_ctxt_data[index][split].append(_vector)

	return _text_data, _ctxt_data

#######################################################################################
############################## Generating Numpy of Input ##############################
#######################################################################################
def save_kfold_npy(data_list, name, path, k):
	file_format = path+"/%s/%s_%s.npy"
	for i in range(k):
		fold_path = path+"/"+str(i)
		if not os.path.exists(fold_path):
			os.makedirs(fold_path)
		for key in data_list[i].keys():
			file_name = file_format % (str(i), name, key)
			array = np.array(data_list[i][key])
			np.save(file_name, array)
			# print("Saved in %s. %s" % (file_name, str(array.shape)))

def processWordVocab(target_path, use_glove, k):
	print("Processing word vocab....")
	data_path = os.path.dirname(os.path.abspath(__file__)) + "/../data"
	with open(data_path+"/data_splits.pkl", "rb") as f:
		_data = pickle.load(f)
	_text_split, _ctxt_split, _label_split = _data["text_data"], _data["context_data"], _data["label_data"]
	id2word, word2id, ctxt_text_idx = genWordVocab(_text_split, _ctxt_split, k)
	
	save_kfold_npy(ctxt_text_idx, "CtxtText_InputText", target_path, k)
	save_kfold_npy(_label_split, "Label", target_path, k)

	with open(target_path+"/vocab.pkl", "wb") as f:
		pickle.dump({"word2id": word2id, "id2word":id2word}, f)
	print("Done creating vocabulary pickles")

	genWordEmbeddings(use_glove, target_path)

def processCharVocab(target_path, k):
	print("Processing char vocab....")
	data_path = os.path.dirname(os.path.abspath(__file__)) + "/../data"
	with open(data_path+"/data_splits.pkl", "rb") as f:
		_data = pickle.load(f)
	_text_split, _ctxt_split, _label_split = _data["text_data"], _data["context_data"], _data["label_data"]
	text_data, ctxt_data = genCharVocab(_text_split, _ctxt_split, k)
	save_kfold_npy(text_data, "Char_InputText", target_path, k)
	save_kfold_npy(ctxt_data, "Char_CtxtText", target_path, k)
	save_kfold_npy(_label_split, "Label", target_path, k)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--use_glove", type=helper.str2bool, default=True)
	parser.add_argument("--num_splits", type=int, default=10)
	args = vars(parser.parse_args())

	target_path = os.path.dirname(os.path.abspath(__file__)) + "/../data/target"
	processWordVocab(target_path, args["use_glove"], args["num_splits"])
	processCharVocab(target_path, args["num_splits"])

