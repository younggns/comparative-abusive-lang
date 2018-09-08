import os
import sys
import argparse
import time
import pickle
import numpy as np
import warnings
import pandas as pd
import functools

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, Callback
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import classification_report

from layers.cnn import WordCNN, WordCNN_Ctxt, CharCNN, HybridCNN

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import helper

class ClassificationReport(Callback):
	def __init__(self, model, x_eval, y_eval, labels):
		self.model = model
		self.x_eval = x_eval
		self.truth = np.argmax(y_eval, axis=1)
		self.labels = labels

	def on_epoch_end(self, epoch, logs={}):
		print("Generating Classification Report:")
		preds = np.argmax(self.model.predict(self.x_eval, verbose=1), axis=1)
		print("\n%s\n" % classification_report(self.truth, preds, target_names=self.labels))

def vector2matrix(text_vector, max_len=140, N_DIM=70):
	matrix = np.zeros((max_len, N_DIM))
	for i, index_elem in enumerate(text_vector):
		row = np.zeros(N_DIM)
		if int(index_elem) != -1:
			row[int(index_elem)] = 1
		matrix[i] = row
	return matrix

def report_average(report_list):
	labels = ["normal", "spam", "hateful", "abusive"]
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

def load_word_npys(path, splits, index):
	text_data = {"train":None, "valid":None}
	ctxt_data = {"train":None, "valid":None}
	label_data = {"train":None, "valid":None}

	file_format = path+"/"+str(index)+"/%s_%s.npy"
	for split in splits: # only save train and valid data
		file_name = file_format % (split, "CtxtText_InputText")
		_CtxtText = np.load(file_name)
		ctxt_data[split] = _CtxtText[:,:100]
		text_data[split] = _CtxtText[:,100:]

		file_name = file_format % (split, "Label")
		label_data[split] = np.load(file_name)

	return text_data, ctxt_data, label_data

def load_char_npys(path, splits, index):
	text_data = {"train":None, "valid":None}
	label_data = {"train":None, "valid":None}

	file_format = path+"/"+str(index)+"/%s_%s.npy"
	for split in splits: # only save train and valid data
		file_name = file_format % (split, "InputText")
		_Text = np.load(file_name)
		text_data[split] = []
		for _text_vector in _Text:
			text_data[split].append(vector2matrix(_text_vector))
		text_data[split] = np.asarray(text_data[split])

		file_name = file_format % (split, "Label")
		label_data[split] = np.load(file_name)

	return text_data, label_data

def train_word_cnn(use_glove, use_ctxt, kfold_index, batch_size, num_epochs, filter_sizes, num_filters, train_embedding, lr):
	labels = ["normal", "spam", "hateful", "abusive"]
	splits = ["train", "valid", "test"]
	model_name = "CNN_WORD"
	if use_ctxt:
		model_name += "_CTXT"
	if use_glove:
		model_name += "_GLOVE"
	
	path = os.path.dirname(os.path.abspath(__file__))
	dataPath_root = path + "/../data/preprocessed/word"
	
	with open(dataPath_root+"/vocab.pkl", "rb") as f:
		vocab = pickle.load(f)
		vocab_size = len(vocab["word2id"].keys())
		print("vocabulary loaded with %s words" % vocab_size)

	embedding_matrix = np.load(dataPath_root+"/embedding.npy")
	assert embedding_matrix.shape[0] == vocab_size
	print("loaded embedding table")

	K.set_learning_phase(1)

	text_data, ctxt_data, label_data = load_word_npys(dataPath_root, splits[:2], kfold_index)

	sequence_length = text_data["train"].shape[1]
	print("sequence_length: %s" % sequence_length)

	log_path = path + "/../data/output/word/"+model_name+"/"+str(kfold_index)

	# define keras training procedure
	tb_callback = TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=True, write_images=True)

	ckpt_callback = ModelCheckpoint(log_path + "/weights.{epoch:02d}.hdf5",
									monitor='val_acc', save_best_only=True,
									save_weights_only=False, mode='max', verbose=1)

	early_stop_callback = EarlyStopping(monitor='val_acc', min_delta=0, patience=1, verbose=0, mode='max')

	if use_ctxt:
		model = WordCNN_Ctxt(sequence_length=sequence_length, n_classes=len(labels), vocab_size=vocab_size, 
		filter_sizes=filter_sizes, num_filters=num_filters, learning_rate=lr, 
		embedding_size=300, embedding_matrix= embedding_matrix, train_embedding=train_embedding)

		clf_report_callback = ClassificationReport(model.model, [text_data["valid"],ctxt_data["valid"]], label_data["valid"], labels)

		model.model.fit(x=[text_data["train"],ctxt_data["train"]], y=label_data["train"], batch_size=batch_size, verbose=2, epochs=num_epochs, 
			callbacks=[tb_callback, clf_report_callback, early_stop_callback, ckpt_callback], 
			validation_data=([text_data["valid"],ctxt_data["valid"]], label_data["valid"]))

	else:
		model = WordCNN(sequence_length=sequence_length, n_classes=len(labels), vocab_size=vocab_size, 
		filter_sizes=filter_sizes, num_filters=num_filters, learning_rate=lr, 
		embedding_size=300, embedding_matrix= embedding_matrix, train_embedding=train_embedding)

		clf_report_callback = ClassificationReport(model.model, text_data["valid"], label_data["valid"], labels)

		model.model.fit(x=text_data["train"], y=label_data["train"], batch_size=batch_size, verbose=2, epochs=num_epochs, 
			callbacks=[tb_callback, clf_report_callback, early_stop_callback, ckpt_callback], 
			validation_data=(text_data["valid"], label_data["valid"]))

	print("Training Finished")

def train_char_cnn(kfold_index, batch_size, num_epochs, filter_sizes, num_filters, train_embedding, lr):
	labels = ["normal", "spam", "hateful", "abusive"]
	splits = ["train", "valid", "test"]
	model_name = "CNN_CHAR"
	
	path = os.path.dirname(os.path.abspath(__file__))
	dataPath_root = path + "/../data/preprocessed/char"

	K.set_learning_phase(1)

	text_data, label_data = load_char_npys(dataPath_root, splits[:2], kfold_index)

	char_len = text_data["train"].shape[1]
	char_embed_dim = text_data["train"].shape[2]
	print("max_char_length: %s" % char_len)
	print("char_dim: %s" % char_embed_dim)

	log_path = path + "/../data/output/char/"+model_name+"/"+str(kfold_index)

	# define keras training procedure
	tb_callback = TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=True, write_images=True)

	ckpt_callback = ModelCheckpoint(log_path + "/weights.{epoch:02d}.hdf5",
									monitor='val_acc', save_best_only=True,
									save_weights_only=False, mode='max', verbose=1)

	early_stop_callback = EarlyStopping(monitor='val_acc', min_delta=0, patience=1, verbose=0, mode='max')

	model = CharCNN(char_len=char_len, char_embed_dim=char_embed_dim, n_classes=len(labels),
			filter_sizes=filter_sizes, num_filters=num_filters, learning_rate=lr)

	clf_report_callback = ClassificationReport(model.model, text_data["valid"], label_data["valid"], labels)

	model.model.fit(x=text_data["train"], y=label_data["train"], batch_size=batch_size, verbose=2, epochs=num_epochs, 
		callbacks=[tb_callback, clf_report_callback, early_stop_callback, ckpt_callback], 
		validation_data=(text_data["valid"], label_data["valid"]))

	print("Training Finished")

def train_hybrid_cnn(use_glove, kfold_index, batch_size, num_epochs, filter_sizes_word, num_filters_word, filter_sizes_char, num_filters_char, train_embedding, lr):
	labels = ["normal", "spam", "hateful", "abusive"]
	splits = ["train", "valid", "test"]
	model_name = "CNN_HYBRID"
	if use_glove:
		model_name += "_GLOVE"
	
	path = os.path.dirname(os.path.abspath(__file__))
	dataPath_root_word = path + "/../data/preprocessed/word"
	dataPath_root_char = path + "/../data/preprocessed/char"
	
	with open(dataPath_root_word+"/vocab.pkl", "rb") as f:
		vocab = pickle.load(f)
		vocab_size = len(vocab["word2id"].keys())
		print("vocabulary loaded with %s words" % vocab_size)

	embedding_matrix = np.load(dataPath_root_word+"/embedding.npy")
	assert embedding_matrix.shape[0] == vocab_size
	print("loaded embedding table")

	K.set_learning_phase(1)

	text_data_word, _, label_data = load_word_npys(dataPath_root_word, splits[:2], kfold_index)
	text_data_char, _ = load_char_npys(dataPath_root_char, splits[:2], kfold_index)

	sequence_length = text_data_word["train"].shape[1]
	char_len = text_data_char["train"].shape[1]
	char_embed_dim = text_data_char["train"].shape[2]
	print("max_char_length: %s" % char_len)
	print("char_dim: %s" % char_embed_dim)
	print("word_sequence_length: %s" % sequence_length)

	log_path = path + "/../data/output/hybrid/"+model_name+"/"+str(kfold_index)

	# define keras training procedure
	tb_callback = TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=True, write_images=True)

	ckpt_callback = ModelCheckpoint(log_path + "/weights.{epoch:02d}.hdf5",
									monitor='val_acc', save_best_only=True,
									save_weights_only=False, mode='max', verbose=1)

	early_stop_callback = EarlyStopping(monitor='val_acc', min_delta=0, patience=1, verbose=0, mode='max')



	model = HybridCNN(n_classes=len(labels),
			char_len=char_len, char_embed_dim=char_embed_dim, char_filter_sizes=filter_sizes_char, char_num_filters=num_filters_char,
			word_sequence_len=sequence_length, word_vocab_size=vocab_size, word_filter_sizes=filter_sizes_word, word_num_filters=num_filters_word, 
			word_embedding_dim=300, embedding_matrix=embedding_matrix, train_embedding=train_embedding,
			learning_rate=lr)

	clf_report_callback = ClassificationReport(model.model, [text_data_word["valid"],text_data_char["valid"]], label_data["valid"], labels)

	model.model.fit(x=[text_data_word["train"],text_data_char["train"]], y=label_data["train"], batch_size=batch_size, verbose=2, epochs=num_epochs, 
		callbacks=[tb_callback, clf_report_callback, early_stop_callback, ckpt_callback], 
		validation_data=([text_data_word["valid"],text_data_char["valid"]], label_data["valid"]))

	print("Training Finished")

def test_word_cnn(use_glove, use_ctxt, k):
	labels = ["normal", "spam", "hateful", "abusive"]
	text_data_list, ctxt_data_list, label_data_list = [], [], []

	model_name = "CNN_WORD"
	if use_ctxt:
		model_name += "_CTXT"
	if use_glove:
		model_name += "_GLOVE"

	path = os.path.dirname(os.path.abspath(__file__))
	dataPath_root = path + "/../data/preprocessed/word"

	text_data_list, ctxt_data_list, label_data_list = [], [], []

	file_format = dataPath_root + "/%s/%s_%s.npy"
	for index in range(k):
		file_name = file_format % (str(index), "test", "CtxtText_InputText")
		_CtxtText = np.load(file_name)
		ctxt_data_list.append(_CtxtText[:,:100])
		text_data_list.append(_CtxtText[:,100:])
		
		file_name = file_format % (str(index), "test", "Label")
		label_data_list.append(np.load(file_name))

	report_list = []
	for index in range(k):
		X_orig = text_data_list[index]
		X_ctxt = ctxt_data_list[index]
		y = label_data_list[index]

		log_path = path + "/../data/output/word/"+model_name+"/"+str(index)
		dir_list = os.listdir(log_path)

		# Find maximum epoch num for each weight output
		max_epoch_num = 0
		for file in dir_list:
			if file.startswith('weights'):
				epoch_num = int(file.split('.')[1])
				if epoch_num > max_epoch_num:
					max_epoch_num = epoch_num

		model = load_model(log_path+"/weights."+str(max_epoch_num).zfill(2)+".hdf5")

		if use_ctxt:
			preds = model.predict([X_orig, X_ctxt], batch_size=128)
		else:
			preds = model.predict(X_orig, batch_size=128)

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			_report = classification_report(np.argmax(y, axis=1), np.argmax(preds, axis=1), digits=4, target_names=labels)
		report_list.append(_report)

	tot_report = report_average(report_list)
	print(tot_report)

def test_char_cnn(k):
	labels = ["normal", "spam", "hateful", "abusive"]
	model_name = "CNN_CHAR"

	path = os.path.dirname(os.path.abspath(__file__))
	dataPath_root = path + "/../data/preprocessed/char"

	file_format = dataPath_root + "/%s/%s_%s.npy"
	report_list = []
	for index in range(k):
		text_data_list = []
		file_name = file_format % (str(index), "test", "InputText")

		_text_data = np.load(file_name)
		for text_vector in _text_data:
			text_data_list.append(vector2matrix(text_vector))
		text_data = np.asarray(text_data_list)
		
		file_name = file_format % (str(index), "test", "Label")
		label_data = np.load(file_name)

		X_orig = text_data
		y = label_data

		log_path = path + "/../data/output/char/CNN_CHAR/"+str(index)
		dir_list = os.listdir(log_path)

		# Find maximum epoch num for each weight output
		max_epoch_num = 0
		for file in dir_list:
			if file.startswith('weights'):
				epoch_num = int(file.split('.')[1])
				if epoch_num > max_epoch_num:
					max_epoch_num = epoch_num

		model = load_model(log_path+"/weights."+str(max_epoch_num).zfill(2)+".hdf5")

		preds = model.predict(X_orig, batch_size=128)

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			_report = classification_report(np.argmax(y, axis=1), np.argmax(preds, axis=1), digits=4, target_names=labels)
		report_list.append(_report)

	tot_report = report_average(report_list)
	print(tot_report)

def test_hybrid_cnn(use_glove, k):
	labels = ["normal", "spam", "hateful", "abusive"]
	# word_text_data_list, char_text_data_list, label_data_list = [], [], []
	char_text_data_list = []

	model_name = "CNN_HYBRID"
	if use_glove:
		model_name += "_GLOVE"

	path = os.path.dirname(os.path.abspath(__file__))
	dataPath_root_word = path + "/../data/preprocessed/word"
	dataPath_root_char = path + "/../data/preprocessed/char"

	word_file_format = dataPath_root_word + "/%s/%s_%s.npy"
	char_file_format = dataPath_root_char + "/%s/%s_%s.npy"
	report_list = []
	for index in range(k):
		file_name = char_file_format % (str(index), "test", "InputText")
		_char_text_data = np.load(file_name)
	
		for text_vector in _char_text_data:
			char_text_data_list.append(vector2matrix(text_vector))
		char_text_data = np.asarray(char_text_data_list)

		file_name = word_file_format % (str(index), "test", "CtxtText_InputText")
		word_text_data = np.load(file_name)[:,100:]
		
		file_name = word_file_format % (str(index), "test", "Label")
		label_data = np.load(file_name)

		X_word = word_text_data
		X_char = char_text_data
		y = label_data

		log_path = path + "/../data/output/hybrid/"+model_name+"/"+str(index)
		dir_list = os.listdir(log_path)

		# Find maximum epoch num for each weight output
		max_epoch_num = 0
		for file in dir_list:
			if file.startswith('weights'):
				epoch_num = int(file.split('.')[1])
				if epoch_num > max_epoch_num:
					max_epoch_num = epoch_num

		model = load_model(log_path+"/weights."+str(max_epoch_num).zfill(2)+".hdf5")

		preds = model.predict([X_word, X_char], batch_size=128)

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			_report = classification_report(np.argmax(y, axis=1), np.argmax(preds, axis=1), digits=4, target_names=labels)
		report_list.append(_report)

	tot_report = report_average(report_list)
	print(tot_report)



