#!/usr/bin/env python
"""
WordCNN Model from
Kim Y. 2014. Convolutional Neural Network for
Sentence Classification In Proceedings of EMNLP.
"""

from keras.layers import Input, Dense, Embedding, Convolution1D, MaxPooling1D
from keras.layers import Flatten, Concatenate
from keras.models import Model
from keras.optimizers import Adam

class WordCNN(object):
	def __init__(
			self, sequence_length, n_classes, vocab_size,
			filter_sizes, num_filters, learning_rate=0.001, 
			embedding_size=300, embedding_matrix=None, train_embedding=True):

		inputs = Input(shape=(sequence_length,))

		embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size,
									trainable=train_embedding, weights=[embedding_matrix])(inputs)

		conv_blocks = []
		for filter_size in filter_sizes:
			conv = Convolution1D(filters=num_filters, kernel_size=filter_size,
								 padding="valid", activation="relu", strides=1)(embedding_layer)
			conv = MaxPooling1D(pool_size=sequence_length - filter_size + 1)(conv)
			conv = Flatten()(conv)
			conv_blocks.append(conv)

		cnn = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

		output = Dense(n_classes, activation="softmax")(cnn)

		self.model = Model(inputs, output)

		self.model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=learning_rate), metrics=["accuracy"])
		self.model.summary()

class WordCNN_Ctxt(object):
	def __init__(
			self, sequence_length, n_classes, vocab_size,
			filter_sizes, num_filters, learning_rate=0.001, 
			embedding_size=300, embedding_matrix=None, train_embedding=True):

		inputs = Input(shape=(sequence_length,))
		ctxt_inputs = Input(shape=(sequence_length,))

		embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size,
									trainable=train_embedding, weights=[embedding_matrix])(inputs)
		ctxt_embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size,
									trainable=train_embedding, weights=[embedding_matrix])(ctxt_inputs)

		conv_blocks = []
		for filter_size in filter_sizes:
			conv = Convolution1D(filters=num_filters, kernel_size=filter_size,
								 padding="valid", activation="relu", strides=1)(embedding_layer)
			conv = MaxPooling1D(pool_size=sequence_length - filter_size + 1)(conv)
			conv = Flatten()(conv)
			conv_blocks.append(conv)

		ctxt_conv_blocks = []
		for filter_size in filter_sizes:
			ctxt_conv = Convolution1D(filters=num_filters, kernel_size=filter_size,
								 padding="valid", activation="relu", strides=1)(ctxt_embedding_layer)
			ctxt_conv = MaxPooling1D(pool_size=sequence_length - filter_size + 1)(ctxt_conv)
			ctxt_conv = Flatten()(ctxt_conv)
			ctxt_conv_blocks.append(ctxt_conv)

		cnn = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
		ctxt_cnn = Concatenate()(ctxt_conv_blocks) if len(ctxt_conv_blocks) > 1 else ctxt_conv_blocks[0]

		cnn = Dense(9, activation="relu")(cnn)
		ctxt_cnn = Dense(6, activation="relu")(ctxt_cnn) # Reduce dimension for context inputs.

		merged_net = Concatenate()([ctxt_cnn, cnn])

		output = Dense(n_classes, activation="softmax")(merged_net)

		self.model = Model([inputs,ctxt_inputs], output)

		self.model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=learning_rate), metrics=["accuracy"])
		self.model.summary()

class CharCNN(object):
	def __init__(
			self, char_len, char_embed_dim, n_classes,
			filter_sizes, num_filters, learning_rate=0.01):

		inputs = Input(shape=(char_len, char_embed_dim))

		conv_blocks = []
		for filter_size in filter_sizes:
			conv = Convolution1D(filters=num_filters, kernel_size=filter_size,
								 padding="valid", activation="relu", strides=1)(inputs)
			conv = MaxPooling1D(pool_size=char_len - filter_size + 1)(conv)
			conv = Flatten()(conv)
			conv_blocks.append(conv)

		cnn = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
		cnn = Dense(1024, activation="relu")(cnn)

		output = Dense(n_classes, activation="softmax")(cnn)

		self.model = Model(inputs, output)

		self.model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=learning_rate), metrics=["accuracy"])
		self.model.summary()

class HybridCNN(object):
	def __init__(
			self, n_classes,
			char_len, char_embed_dim, char_filter_sizes, char_num_filters,
			word_sequence_len, word_vocab_size, word_filter_sizes, word_num_filters, 
			word_embedding_dim=300, embedding_matrix=None, train_embedding=False,
			learning_rate=0.001):

		input_char = Input(shape=(char_len, char_embed_dim))
		input_word = Input(shape=(word_sequence_len,))
		
		embedding_layer = Embedding(input_dim=word_vocab_size,
									output_dim=word_embedding_dim,
									trainable=train_embedding,
									weights=[embedding_matrix])(input_word)

		# word cnn layers
		conv_blocks = []
		for filter_size in word_filter_sizes:
			conv = Convolution1D(filters=word_num_filters,
								 kernel_size=filter_size,
								 padding="valid",
								 activation="relu",
								 strides=1)(embedding_layer)
			conv = MaxPooling1D(pool_size=word_sequence_len - filter_size + 1)(conv)
			conv = Flatten()(conv)
			conv_blocks.append(conv)

		# char cnn layers
		for filter_size in char_filter_sizes:
			conv = Convolution1D(filters=char_num_filters,
								 kernel_size=filter_size,
								 padding="valid",
								 activation="relu",
								 strides=1)(input_char)
			conv = MaxPooling1D(pool_size=char_len - filter_size + 1)(conv)
			conv = Flatten()(conv)
			conv_blocks.append(conv)

		cnn = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

		output = Dense(n_classes, activation="softmax")(cnn)

		self.model = Model([input_word, input_char], output)

		self.model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=learning_rate), metrics=["accuracy"])
		self.model.summary()
