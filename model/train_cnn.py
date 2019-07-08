import os
import sys
import argparse
import time
import pickle
import numpy as np
import layers.CNN_models

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import helper

def modelSpecCNN(is_hybrid):
	print("\n#############################\n#### Model Specification ####\n#############################\n")
	batch_size = int(input("{:30}".format("Batch size:")))
	num_epochs = int(input("{:30}".format("Number of epoch:")))
	train_embedding_str = input("{:30}".format("Train embedding? (True/False):"))
	train_embedding = helper.str2bool(train_embedding_str)
	lr = float(input("{:30}".format("Learning rate:")))

	if is_hybrid:
		filter_sizes_str_word = input("{:30}".format("Word Filter sizes (e.g. 1,2,3):"))
		num_filters_word = int(input("{:30}".format("Number of word filters:")))
		filter_sizes_str_char = input("{:30}".format("Char Filter sizes (e.g. 1,2,3):"))
		num_filters_char = int(input("{:30}".format("Number of char filters:")))

		filter_sizes_word = [int(elem) for elem in filter_sizes_str_word.split(',')]
		filter_sizes_char = [int(elem) for elem in filter_sizes_str_char.split(',')]

		return batch_size, num_epochs, filter_sizes_word, num_filters_word, filter_sizes_char, num_filters_char, train_embedding, lr

	else:
		filter_sizes_str = input("{:30}".format("Filter sizes (e.g. 1,2,3):"))
		num_filters = int(input("{:30}".format("Number of filters:")))

		filter_sizes = [int(elem) for elem in filter_sizes_str.split(',')]

		return batch_size, num_epochs, filter_sizes, num_filters, train_embedding, lr

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument("--feature_level", type=helper.str2feature, default='word')
	parser.add_argument("--num_splits", type=int, default=10)
	parser.add_argument("--use_glove", type=helper.str2bool, default=True)

	args = vars(parser.parse_args())

	path = os.path.dirname(os.path.abspath(__file__)) + "/../data/target/"
	use_glove = args['use_glove']

	if args["feature_level"] == "word":
		use_ctxt_str = input("{:30}".format("Use context tweet? (True/False):"))
		use_ctxt = helper.str2bool(use_ctxt_str)
		batch_size, num_epochs, filter_sizes, num_filters, train_embedding, lr = modelSpecCNN(False)
		time_index = int(time.time())
		for index in range(args['num_splits']):
			layers.CNN_models.train_word_cnn(use_glove, use_ctxt, index, batch_size, num_epochs, filter_sizes, num_filters, train_embedding, lr, time_index)
		layers.CNN_models.test_word_cnn(use_glove, use_ctxt, args['num_splits'], time_index)

	elif args["feature_level"] == "char":
		batch_size, num_epochs, filter_sizes, num_filters, train_embedding, lr = modelSpecCNN(False)
		time_index = int(time.time())
		for index in range(args['num_splits']):
			layers.CNN_models.train_char_cnn(index, batch_size, num_epochs, filter_sizes, num_filters, train_embedding, lr, time_index)
		layers.CNN_models.test_char_cnn(args['num_splits'], time_index)

	else:
		batch_size, num_epochs, filter_sizes_word, num_filters_word, filter_sizes_char, num_filters_char, train_embedding, lr = modelSpecCNN(True)
		time_index = int(time.time())
		for index in range(args['num_splits']):
			layers.CNN_models.train_hybrid_cnn(use_glove, index, batch_size, num_epochs, filter_sizes_word, num_filters_word, filter_sizes_char, num_filters_char, train_embedding, lr, time_index)
		layers.CNN_models.test_hybrid_cnn(use_glove, index, time_index)

