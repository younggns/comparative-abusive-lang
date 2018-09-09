import os
import sys
import argparse
import time
from datetime import datetime
import pickle
import numpy as np
import models_cnn

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
		if not os.path.exists("./../glove.840B.300d.txt"):
			print("\nDownloading pre-trained GloVe embedding.....")
			os.system("curl -LJO http://nlp.stanford.edu/data/glove.840B.300d.zip")
			print("Downloaded.\n")
			print("Unzipping...")
			os.system("unzip glove.840B.300d.zip")
			print("Done.\n")

		glove = {}
		print("Loading pre-trained embedding.")
		f = open("./../glove.840B.300d.txt")
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
			os.mkdir(fold_path)
		for key in data_list[i].keys():
			file_name = file_format % (str(i), key, name)
			array = np.array(data_list[i][key])
			np.save(file_name, array)
			print("Saved in %s. %s" % (file_name, str(array.shape)))

def modelSpecCNN(is_hybrid):
	print("\n#############################\n#### Model Specification ####\n#############################\n")
	batch_size = int(input("{:30}".format("Batch size:")))
	num_epochs = int(input("{:30}".format("Number of epoch:")))
	if is_hybrid:
		filter_sizes_str_word = input("{:30}".format("Word Filter sizes (e.g. 1,2,3):"))
		num_filters_word = int(input("{:30}".format("Number of word filters:")))
		filter_sizes_str_char = input("{:30}".format("Char Filter sizes (e.g. 1,2,3):"))
		num_filters_char = int(input("{:30}".format("Number of char filters:")))
	else:
		filter_sizes_str = input("{:30}".format("Filter sizes (e.g. 1,2,3):"))
		num_filters = int(input("{:30}".format("Number of filters:")))
	train_embedding_str = input("{:30}".format("Train embedding? (True/False):"))
	lr = float(input("{:30}".format("Learning rate:")))

	if is_hybrid:
		filter_sizes_word = [int(elem) for elem in filter_sizes_str_word.split(',')]
		filter_sizes_char = [int(elem) for elem in filter_sizes_str_char.split(',')]
	else:
		filter_sizes = [int(elem) for elem in filter_sizes_str.split(',')]
	train_embedding = helper.str2bool(train_embedding_str)

	if is_hybrid:
		return batch_size, num_epochs, filter_sizes_word, num_filters_word, filter_sizes_char, num_filters_char, train_embedding, lr
	else:
		return batch_size, num_epochs, filter_sizes, num_filters, train_embedding, lr

def modelSpecRNN():
	print("\n#############################\n#### Model Specification ####\n#############################\n")
	batch_size = input("{:30}".format("Batch size:"))
	encoder_size = input("{:30}".format("Encoder size:"))
	num_layer = input("{:30}".format("Number of layers:"))
	hidden_dim = input("{:30}".format("Hidden layer dimension:"))
	num_train_steps = input("{:30}".format("Number of train steps:"))
	lr = input("{:30}".format("Learning rate:"))
	dr = input("{:30}".format("Dropout probability:"))
	is_save_str = input("{:30}".format("Save best result? (True/False):"))
	attn_str = input("{:30}".format("Apply attention? (True/False):"))
	ltc_str = input("{:30}".format("Apply LTC? (True/False):"))
	graph_prefix = input("{:30}".format("Graph prefix:"))

	is_save = helper.str2bool(is_save_str)
	attn = helper.str2bool(attn_str)
	ltc = helper.str2bool(ltc_str)

	return batch_size, encoder_size, num_layer, hidden_dim, lr, num_train_steps, is_save, dr, attn, ltc, graph_prefix

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--feature_level", type=helper.str2feature, default='word')
	parser.add_argument("--clf", type=helper.str2clfs, required=True)

	args = vars(parser.parse_args())

	if args["clf"] not in ["CNN", "RNN"]:
		print("Invalid classifier input.")
		exit()

	process_vocab_str = input("{:30}".format("Process vocab? (True/False):"))
	process_vocab = helper.str2bool(process_vocab_str)

	path = os.path.dirname(os.path.abspath(__file__)) + "/../data/preprocessed/" + args["feature_level"]
	if args["feature_level"] == "word":
		use_glove_str = input("{:30}".format("Use GloVe? (True/False):"))
		use_glove = helper.str2bool(use_glove_str)
		if process_vocab:
			with open(path+"/data_splits.pkl", "rb") as f:
				_data = pickle.load(f)
			_text_split, _ctxt_split, _label_split = _data["text_data"], _data["context_data"], _data["label_data"]
			id2word, word2id, ctxt_text_idx = genWordVocab(_text_split, _ctxt_split, 10)
			save_kfold_npy(ctxt_text_idx, "CtxtText_InputText", path, 10)
			save_kfold_npy(_label_split, "Label", path, 10)

			with open(path+"/vocab.pkl", "wb") as f:
				pickle.dump({"word2id": word2id, "id2word":id2word}, f)
			print("Done creating vocabulary pickles")

			genWordEmbeddings(use_glove, path)

		if args["clf"] == "CNN":
			is_hybrid_str = input("{:30}".format("Hybrid CNN? (True/False):"))
			is_hybrid = helper.str2bool(is_hybrid_str)
			
			if is_hybrid:
				batch_size, num_epochs, filter_sizes_word, num_filters_word, filter_sizes_char, num_filters_char, train_embedding, lr = modelSpecCNN(is_hybrid)
				for index in range(10):
					models_cnn.train_hybrid_cnn(use_glove, index, batch_size, num_epochs, filter_sizes_word, num_filters_word, filter_sizes_char, num_filters_char, train_embedding, lr)
				models_cnn.test_hybrid_cnn(use_glove, 10)
			else:
				use_ctxt_str = input("{:30}".format("Use context tweet? (True/False):"))
				use_ctxt = helper.str2bool(use_ctxt_str)
				batch_size, num_epochs, filter_sizes, num_filters, train_embedding, lr = modelSpecCNN(is_hybrid)
				for index in range(10):
					models_cnn.train_word_cnn(use_glove, use_ctxt, index, batch_size, num_epochs, filter_sizes, num_filters, train_embedding, lr)
				models_cnn.test_word_cnn(use_glove, use_ctxt, 10)
		else:
			for index in range(10):
				data_path = path + "/" +str(index) + "/"
				batch_size, encoder_size, num_layer, hidden_dim, lr, num_train_steps, is_save, dr, attn, ltc, graph_prefix = modelSpecRNN()
				
				use_glove_rnn = 1 if use_glove==True else 0
				is_save_rnn = 1 if is_save==True else 0
				attn_rnn = 1 if attn==True else 0
				ltc_rnn = 1 if ltc==True else 0

				os.system("python3 models_rnn.py --batch_size "+batch_size+" --encoder_size "+encoder_size+" --num_layer "+num_layer+" --hidden_dim "+hidden_dim+" --lr "+lr+" --num_train_steps "+num_train_steps+" --is_save "+is_save_rnn+" --dr "+dr+" --use_glove "+use_glove_rnn+" --attn "+attn_rnn+" --ltc "+ltc_rnn+" --graph_prefix "+graph_prefix+" --data_path "+data_path)

	else: # character models
		if process_vocab:
			with open(path+"/data_splits.pkl", "rb") as f:
				_data = pickle.load(f)
			_text_split, _ctxt_split, _label_split = _data["text_data"], _data["context_data"], _data["label_data"]
			text_data, ctxt_data = genCharVocab(_text_split, _ctxt_split, 10)
			save_kfold_npy(text_data, "InputText", path, 10)
			save_kfold_npy(ctxt_data, "CtxtText", path, 10)
			save_kfold_npy(_label_split, "Label", path, 10)

		if args["clf"] == "CNN":
			is_hybrid_str = input("{:30}".format("Hybrid CNN? (True/False):"))
			is_hybrid = helper.str2bool(is_hybrid_str)
			if is_hybrid:
				use_glove_str = input("{:30}".format("Use GloVe? (True/False):"))
				use_glove = helper.str2bool(use_glove_str)
				batch_size, num_epochs, filter_sizes_word, num_filters_word, filter_sizes_char, num_filters_char, train_embedding, lr = modelSpecCNN(is_hybrid)
				for index in range(10):
					models_cnn.train_hybrid_cnn(use_glove, index, batch_size, num_epochs, filter_sizes_word, num_filters_word, filter_sizes_char, num_filters_char, train_embedding, lr)
				models_cnn.test_hybrid_cnn(use_glove, index)
			else:
				batch_size, num_epochs, filter_sizes, num_filters, train_embedding, lr = modelSpecCNN(is_hybrid)
				for index in range(10):
					models_cnn.train_char_cnn(index, batch_size, num_epochs, filter_sizes, num_filters, train_embedding, lr)
				models_cnn.test_char_cnn(10)
		# else:
		# 	os.system("python3 models_rnn.py ")

