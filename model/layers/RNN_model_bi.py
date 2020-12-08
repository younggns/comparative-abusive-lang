# -*- coding: utf-8 -*-
# /usr/bin/python2

"""
what    : Single Encoder Model - bidirectional
data    : twitter
"""
import tensorflow as tf
from tensorflow.compat.v1.nn.rnn_cell import DropoutWrapper

from tensorflow.core.framework import summary_pb2
from random import shuffle
import numpy as np
from layers.RNN_params import Params

from layers.RNN_model_layers import *
from layers.RNN_model_GRU import gated_attention_Wrapper
from layers.RNN_layers import add_GRU


class SingleEncoderModelBi:

    def __init__(self, dic_size,
                 use_glove,
                 batch_size,
                 encoder_size,
                 num_layer, lr,
                 hidden_dim,
                 dr,
                 attn, ltc
                 ):

        self.dic_size = dic_size
        self.use_glove = use_glove
        self.batch_size = batch_size
        self.encoder_size = encoder_size
        self.num_layers = num_layer
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.dr = dr

        self.attn = attn
        self.ltc = ltc
        self.ltc_dr = Params.LTC_dr_prob

        self.encoder_inputs = []
        self.encoder_seq_length = []
        self.y_labels = []

        self.M = None
        self.b = None

        self.y = None
        self.optimizer = None

        self.batch_loss = None
        self.loss = 0
        self.batch_prob = None

        if self.use_glove == 1:
            self.embed_dim = 300
        else:
            self.embed_dim = Params.DIM_WORD_EMBEDDING

        # for global counter
        self.global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step')

    def _create_placeholders(self):
        print('[launch] placeholders')
        with tf.name_scope('text_placeholder'):

            self.encoder_inputs_o = tf.placeholder(tf.int32, shape=[
                                                   self.batch_size, self.encoder_size], name="encoder_o")  # [batch,time_step]
            self.encoder_seq_o = tf.placeholder(tf.int32, shape=[
                                                self.batch_size], name="encoder_seq_o")   # [batch] - valid word step
            # self.encoder_type_o    = tf.placeholder(tf.int32, shape=[self.batch_size, 5], name="encoder_type_o")   # [batch] - tweet type 0-5

            self.encoder_inputs_c = tf.placeholder(tf.int32, shape=[
                                                   self.batch_size, self.encoder_size], name="encoder_c")  # [batch,time_step]
            self.encoder_seq_c = tf.placeholder(tf.int32, shape=[
                                                self.batch_size], name="encoder_seq_c")   # [batch] - valid word step
            # self.encoder_type_c    = tf.placeholder(tf.int32, shape=[self.batch_size, 5], name="encoder_type_c")   # [batch] - tweet type 0-5

            self.y_labels = tf.placeholder(
                tf.float32, shape=[self.batch_size, Params.N_CATEGORY], name="label")

            self.dr_prob = tf.placeholder(tf.float32, name="dropout")
            self.dr_prob_ltc = tf.placeholder(tf.float32, name="dropout_ltc")

            # for using pre-trained embedding
            self.embedding_placeholder = tf.placeholder(
                tf.float32, shape=[self.dic_size, self.embed_dim], name="embedding_placeholder")

    def _encoding_ids(self):
        print('[launch] encoding_ids with GRU, is_bidir: ',
              Params.is_text_encoding_bidir)
        with tf.name_scope('text_encoding_layer'):
            self.embed_matrix = tf.Variable(tf.random_normal([self.dic_size, self.embed_dim],
                                                             mean=0.0,
                                                             stddev=0.01,
                                                             dtype=tf.float32,
                                                             seed=None),
                                            trainable=Params.EMBEDDING_TRAIN,
                                            name='embed_matrix')

            self.embed_en_o = tf.nn.embedding_lookup(
                self.embed_matrix, self.encoder_inputs_o, name='embed_encoder_o')
            self.embed_en_c = tf.nn.embedding_lookup(
                self.embed_matrix, self.encoder_inputs_c, name='embed_encoder_c')

            self.encoded_o, self.output_states = add_GRU(
                inputs=self.embed_en_o,
                inputs_len=self.encoder_seq_o,
                hidden_dim=self.hidden_dim,
                layers=self.num_layers,
                scope='passage_encoding',
                reuse=False,
                dr_input_keep_prob=self.dr_prob,
                dr_output_keep_prob=self.dr_prob,
                is_bidir=Params.is_text_encoding_bidir,
                is_bw_reversed=Params.reverse_bw
            )

            # self.encoded_o     = [ batch, step, dim (fw;bw) ]
            # self.output_states = [ layers, batch, each_dim (fw;bw) ]
            self.final_encoding = self.encoded_o
            self.final_step = self.output_states[-1]

            if Params.is_text_encoding_bidir:
                self.final_encoder_dimension = self.hidden_dim * 2
            else:
                self.final_encoder_dimension = self.hidden_dim

    def _use_external_embedding(self):
        if self.use_glove == 1:
            print('[launch] use pre-trained embedding')
            self.embedding_init = self.embed_matrix.assign(
                self.embedding_placeholder)

    def _add_context_gru(self):
        print('[launch] create gru cell for context data, is_bidir: ',
              Params.is_context_bidir)
        with tf.name_scope('context_encoding_layer') as scope:

            self.encoded_c, self.output_states_c = add_GRU(
                inputs=self.embed_en_c,
                inputs_len=self.encoder_seq_c,
                hidden_dim=self.hidden_dim,
                layers=self.num_layers,
                scope='context_encoding',
                reuse=False,
                dr_input_keep_prob=self.dr_prob,
                dr_output_keep_prob=self.dr_prob,
                is_bidir=Params.is_context_bidir,
                is_bw_reversed=Params.reverse_bw
            )

        # result merge
        self.final_encoding = tf.concat(
            [self.final_encoding, self.encoded_c], axis=2)
        self.final_step = tf.concat(
            [self.output_states[-1],  self.output_states_c[-1]], axis=1)

        if Params.is_context_bidir:
            self.final_encoder_dimension = self.final_encoder_dimension + self.hidden_dim * 2
        else:
            self.final_encoder_dimension = self.final_encoder_dimension + self.hidden_dim

    # memory  : attn 되어 weighted sum 될 대상 (Rnet 에서 question encoding)
    # inputs  : V_t_P = RNN (V_(t-1)_P, gated( u_t_P, c_t) ) 식을 통해 최종 update 될 input ( Rnet 에서 passage encoding )
    # self_matching : memory == inputs 일 경우 True

    def _add_attention_match_rnn(self):
        print('[launch-model_util] apply self-matching for origianl text')

        # Apply gated attention recurrent network for both query-passage matching and self matching networks
        with tf.variable_scope("self-matching"):

            self.params = get_attn_params(
                self.final_encoder_dimension/2, initializer=tf.contrib.layers.xavier_initializer)

            memory = self.final_encoding
            inputs = self.final_encoding
            self_matching = True

            self.memory_len = self.encoder_seq_o
            self.inputs_len = self.encoder_seq_o

            scopes = ["memory_input_matching", "self_matching"]
            params = [
                # (([self.params["W_u_Q"],
                # self.params["W_u_P"],
                # self.params["W_v_P"]],self.params["v"]),
                # self.params["W_g"]),
                (),
                (([self.params["W_v_P_2"],
                   self.params["W_v_Phat"]], self.params["v"]),
                 self.params["W_g"])
            ]

            args = {"num_units": self.final_encoder_dimension/2,
                    "memory": memory,
                    "params": params[1] if self_matching == True else params[0],
                    "self_matching": self_matching,
                    "memory_len": self.memory_len,     # memory 의 seq length
                    "is_training": True,
                    "use_SRU": False,
                    "batch_size": self.batch_size}

            #cell = [apply_dropout(gated_attention_Wrapper(**args), size = inputs.shape[-1], is_training = True, dropout=self.dr_prob) for _ in range(2)]
            cell = [apply_dropout(gated_attention_Wrapper(
                **args), size=int(inputs.shape[-1]), is_training=True, output_keep_prob=0.8) for _ in range(2)]
            inputs, self.output_states = attention_rnn(
                inputs=inputs,
                inputs_len=self.inputs_len,    # inputs 의 seq length
                units=self.final_encoder_dimension/2,
                attn_cell=cell,
                bidirection=Params.self_matching_bidir,
                scope=scopes[1] if self_matching == True else scopes[0],
                is_training=True,
                dr_prob=self.dr_prob,
                is_bidir=True)

            self.self_matching_output = inputs

            self.final_encoding = self.self_matching_output
            self.final_step = self.output_states

            if Params.self_matching_bidir:
                self.final_encoder_dimension = self.final_encoder_dimension
            else:
                self.final_encoder_dimension = self.final_encoder_dimension / 2

    def _add_LTC_method(self):
        from layers.RNN_model_ltc import ltc
        print('[launch-model_util] apply LTC method, N_TOPIC/Mem DIM/LTC_DR: ',
              Params.N_LTC_TOPIC, Params.N_LTC_MEM_DIM, self.ltc_dr)

        with tf.name_scope('text_LTC') as scope:
            self.final_step, self.final_encoder_dimension = ltc(batch_size=self.batch_size,
                                                                topic_size=Params.N_LTC_TOPIC,
                                                                memory_dim=Params.N_LTC_MEM_DIM,
                                                                input_hidden_dim=self.final_encoder_dimension,
                                                                input_encoder=self.final_step,
                                                                dr_memory_prob=self.dr_prob_ltc
                                                                )

    def _add_ff_layer(self):
        print('[launch] add FF layer to reduce the dim: ', Params.DIM_FF_LAYER)

        with tf.name_scope('text_FF') as scope:

            initializers = tf.contrib.layers.xavier_initializer(
                uniform=True,
                seed=None,
                dtype=tf.float32
            )

            self.final_step = tf.contrib.layers.fully_connected(
                inputs=self.final_step,
                num_outputs=Params.DIM_FF_LAYER,
                activation_fn=tf.nn.tanh,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer=initializers,
                weights_regularizer=None,
                biases_initializer=tf.zeros_initializer(),
                biases_regularizer=None,
                trainable=True
            )

            self.final_encoder_dimension = Params.DIM_FF_LAYER

    def _create_output_layers(self):
        print('[launch] create output projection layer')

        with tf.name_scope('text_output_layer') as scope:

            self.M = tf.Variable(tf.random_uniform([self.final_encoder_dimension, Params.N_CATEGORY],
                                                   minval=-0.25,
                                                   maxval=0.25,
                                                   dtype=tf.float32,
                                                   seed=None),
                                 trainable=True,
                                 name="similarity_matrix")

            self.b = tf.Variable(tf.zeros([Params.N_CATEGORY], dtype=tf.float32),
                                 trainable=True,
                                 name="output_bias")

            # e * M + b
            self.batch_pred = tf.matmul(self.final_step, self.M) + self.b

        with tf.name_scope('loss') as scope:

            #self.batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.batch_pred, labels=self.y_labels )
            self.batch_loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.batch_pred, labels=self.y_labels)
            self.tmp_batch_loss = self.batch_loss

            if Params.is_minority_use:
                print('[apply] minority E( |loss - eta|^2  eat: )', Params.eta)
                self.batch_loss = self.batch_loss - Params.eta
                self.batch_loss = tf.maximum(self.batch_loss, 0)
                self.batch_loss = tf.square(self.batch_loss)

            self.loss = tf.reduce_mean(self.batch_loss)

    def _create_optimizer(self):
        print('[launch] create optimizer')

        with tf.name_scope('text_optimizer') as scope:
            opt_func = tf.train.AdamOptimizer(learning_rate=self.lr)
            gradients, variables = zip(*opt_func.compute_gradients(self.loss))
            gradients = [None if gradient is None else tf.clip_by_norm(
                t=gradient, clip_norm=1.0) for gradient in gradients]

            self.optimizer = opt_func.apply_gradients(
                zip(gradients, variables), global_step=self.global_step)

    def _create_summary(self):
        print('[launch] create summary')

        with tf.name_scope('summary'):
            tf.summary.scalar('mean_loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._create_placeholders()
        self._encoding_ids()
        self._use_external_embedding()

        if Params.is_context_use:
            self._add_context_gru()
        else:
            if self.attn:
                self._add_attention_match_rnn()

        if self.ltc:
            self._add_LTC_method()

        self._add_ff_layer()
        self._create_output_layers()

        self._create_optimizer()
        self._create_summary()
