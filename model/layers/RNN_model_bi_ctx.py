# -*- coding: utf-8 -*-

"""
what    : Single Encoder Model - bidirectional
data    : twitter
"""
import tensorflow as tf
from tensorflow.compat.v1.nn.rnn_cell import DropoutWrapper, GRUCell, MultiRNNCell

from tensorflow.core.framework import summary_pb2
from random import shuffle
import numpy as np
from layers.RNN_params import Params


class SingleEncoderModelBiSingle:

    def __init__(self, dic_size,
                 use_glove,
                 batch_size,
                 encoder_size,
                 num_layer, lr,
                 hidden_dim,
                 dr,
                 # o_type, c_text, c_type,
                 c_text,
                 attn, ltc,
                 dbias
                 ):

        self.dic_size = dic_size
        self.use_glove = use_glove
        self.batch_size = batch_size
        self.encoder_size = encoder_size
        self.num_layers = num_layer
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.dr = dr

        # self.o_type = o_type
        self.c_text = c_text
        # self.c_type = c_type

        self.attn = attn
        self.ltc = ltc

        self.consider_data_bias = dbias

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
        print('[launch-text] placeholders')
        with tf.name_scope('text_placeholder'):
            tf.compat.v1.disable_eager_execution()

            self.encoder_inputs_o = tf.compat.v1.placeholder(tf.int32, shape=[
                self.batch_size, self.encoder_size], name="encoder_o")  # [batch,time_step]
            self.encoder_seq_o = tf.compat.v1.placeholder(tf.int32, shape=[
                self.batch_size], name="encoder_seq_o")   # [batch] - valid word step
            # self.encoder_type_o    = tf.compat.v1.placeholder(tf.float32, shape=[self.batch_size, 5], name="encoder_type_o")   # [batch] - tweet type 0-5

            self.encoder_inputs_c = tf.compat.v1.placeholder(tf.int32, shape=[
                self.batch_size, self.encoder_size], name="encoder_c")  # [batch,time_step]
            self.encoder_seq_c = tf.compat.v1.placeholder(tf.int32, shape=[
                self.batch_size], name="encoder_seq_c")   # [batch] - valid word step
            # self.encoder_type_c    = tf.compat.v1.placeholder(tf.float32, shape=[self.batch_size, 5], name="encoder_type_c")   # [batch] - tweet type 0-5

            self.y_labels = tf.compat.v1.placeholder(
                tf.float32, shape=[self.batch_size, Params.N_CATEGORY], name="label")

            self.dr_prob = tf.compat.v1.placeholder(tf.float32, name="dropout")
            self.dr_prob_ltc = tf.compat.v1.placeholder(
                tf.float32, name="dropout_ltc")

            # for using pre-trained embedding
            self.embedding_placeholder = tf.compat.v1.placeholder(
                tf.float32, shape=[self.dic_size, self.embed_dim], name="embedding_placeholder")

    def _create_embedding(self):
        print('[launch-text] create embedding')
        with tf.name_scope('embed_layer'):
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

    def _use_external_embedding(self):
        if self.use_glove == 1:
            print('[launch-text] use pre-trained embedding')
            self.embedding_init = self.embed_matrix.assign(
                self.embedding_placeholder)

    # cell instance

    def gru_cell(self):
        return GRUCell(num_units=self.hidden_dim)

    # cell instance with drop-out wrapper applied

    def gru_drop_out_cell(self):
        return DropoutWrapper(self.gru_cell(), input_keep_prob=self.dr_prob, output_keep_prob=self.dr_prob)

    def _create_gru_model_bi(self):
        print('[launch-text] create gru cell for original data - bidirectional')

        with tf.name_scope('text_RNN') as scope:

            with tf.variable_scope("text_GRU", reuse=False, initializer=tf.orthogonal_initializer()):

                cells_en_fw = MultiRNNCell(
                    [self.gru_drop_out_cell() for _ in range(self.num_layers)])
                cells_en_bw = MultiRNNCell(
                    [self.gru_drop_out_cell() for _ in range(self.num_layers)])

                (self.outputs_o, output_states) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cells_en_fw,
                    cell_bw=cells_en_bw,
                    inputs=self.embed_en_o,
                    dtype=tf.float32,
                    sequence_length=self.encoder_seq_o,
                    time_major=False)

                self.state_concat_o = tf.concat(output_states, 2)[-1]
                self.final_encoder = self.state_concat_o

        self.final_encoder_dimension = self.hidden_dim * 2

    def _add_attn_self(self):
        from model_util import luong_attention
        print('[launch-text] apply self-Attention for origianl text')

        with tf.name_scope('text_Attn_self') as scope:

            # attn at original text
            self.outputs = self.outputs_o

            # forward output
            fw = self.outputs[0]

            # reversed backward output
            bw = tf.reverse_sequence(self.outputs[1],
                                     seq_lengths=self.encoder_seq_o,
                                     seq_axis=1
                                     )

            self.output_concat = tf.concat([fw, bw], 2)

            '''
            # attention memory            
            self.attnM = tf.Variable(tf.random_uniform([self.final_encoder_dimension],
                                                       minval= -0.25,
                                                       maxval= 0.25,
                                                       dtype=tf.float32,
                                                       seed=None),
                                                     trainable=True,
                                                     name="attn_memory")

            # multiply attn memoery as many as batch_size ( use same attn memory )
            self.attnM = tf.ones( [self.batch_size, 1] ) * self.attnM
            '''

            self.final_encoder, self.attn_norm = luong_attention(batch_size=self.batch_size,
                                                                 target=self.output_concat,
                                                                 condition=self.state_concat_o,
                                                                 target_encoder_length=self.encoder_size,
                                                                 hidden_dim=self.final_encoder_dimension
                                                                 )

    def _add_context_gru(self):
        print('[launch-text] create gru cell for context data - bidirectional')

        with tf.name_scope('text_RNN') as scope:

            with tf.variable_scope("text_GRU", reuse=True, initializer=tf.orthogonal_initializer()):

                cells_en_fw_c = MultiRNNCell(
                    [self.gru_drop_out_cell() for _ in range(self.num_layers)])
                cells_en_bw_c = MultiRNNCell(
                    [self.gru_drop_out_cell() for _ in range(self.num_layers)])

                (self.outputs_c, output_states) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cells_en_fw_c,
                    cell_bw=cells_en_bw_c,
                    inputs=self.embed_en_c,
                    dtype=tf.float32,
                    sequence_length=self.encoder_seq_c,
                    time_major=False)

                self.state_concat_c = tf.concat(output_states, 2)[-1]
                final_encoder_c = self.state_concat_c

        # result merge
        self.final_encoder = tf.concat(
            [self.final_encoder, final_encoder_c], 1)
        self.final_encoder_dimension = self.final_encoder_dimension + self.hidden_dim * 2

    def _add_attn_con(self):
        from model_util import luong_attention
        print('[launch-text] create gru cell for context data - bidirectional for preparing attention info')

        with tf.name_scope('text_RNN') as scope:

            with tf.variable_scope("text_GRU", reuse=True, initializer=tf.orthogonal_initializer()):

                cells_en_fw_c = MultiRNNCell(
                    [self.gru_drop_out_cell() for _ in range(self.num_layers)])
                cells_en_bw_c = MultiRNNCell(
                    [self.gru_drop_out_cell() for _ in range(self.num_layers)])

                (self.outputs_c, output_states) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cells_en_fw_c,
                    cell_bw=cells_en_bw_c,
                    inputs=self.embed_en_c,
                    dtype=tf.float32,
                    sequence_length=self.encoder_seq_c,
                    time_major=False)

                self.state_concat_c = tf.concat(output_states, 2)[-1]
                final_encoder_c = self.state_concat_c

        print('[launch-text] apply self-Attention with given context vector')
        with tf.name_scope('text_Attn_con') as scope:

            # attn at original text
            self.outputs = self.outputs_o

            # forward output
            fw = self.outputs[0]

            # reversed backward output
            bw = tf.reverse_sequence(self.outputs[1],
                                     seq_lengths=self.encoder_seq_o,
                                     seq_axis=1
                                     )

            self.output_concat = tf.concat([fw, bw], 2)

            self.final_encoder, self.attn_norm = luong_attention(batch_size=self.batch_size,
                                                                 target=self.output_concat,
                                                                 condition=self.state_concat_c,
                                                                 target_encoder_length=self.encoder_size,
                                                                 hidden_dim=self.final_encoder_dimension
                                                                 )

    def _add_ff_layer(self):
        print('[launch-text] add FF layer to reduce the dim: ',
              Params.DIM_FF_LAYER)

        with tf.name_scope('text_FF') as scope:

            initializers = tf.nn.conv2d(
                uniform=True,
                seed=None,
                dtype=tf.float32
            )

            self.final_encoder = tf.keras.layers.Dense(inputs=self.final_encoder,
                                                       num_outputs=Params.DIM_FF_LAYER,
                                                       activation_fn=tf.nn.relu,
                                                       normalizer_fn=None,
                                                       normalizer_params=None,
                                                       weights_initializer=initializers,
                                                       weights_regularizer=None,
                                                       biases_initializer=tf.zeros_initializer(),
                                                       biases_regularizer=None,
                                                       trainable=True
                                                       )

            self.final_encoder_dimension = Params.DIM_FF_LAYER

    # def _add_type_original(self):
    #     print ('[launch-text] add tweet type for original')

    #     # result merge
    #     self.final_encoder = tf.concat( [self.final_encoder, self.encoder_type_o], 1 )
    #     self.final_encoder_dimension = self.final_encoder_dimension + 5

    # def _add_type_context(self):
    #     print ('[launch-text] add tweet type for context')

    #     # result merge
    #     self.final_encoder = tf.concat( [self.final_encoder, self.encoder_type_c], 1 )
    #     self.final_encoder_dimension = self.final_encoder_dimension + 5

    def _add_LTC_method(self):
        from model_util import sy_ltc
        print('[launch-text] apply LTC method')

        with tf.name_scope('text_LTC') as scope:
            self.final_encoder, self.final_encoder_dimension = sy_ltc(batch_size=self.batch_size,
                                                                      topic_size=4,
                                                                      memory_dim=self.final_encoder_dimension,
                                                                      hidden_dim=self.final_encoder_dimension,
                                                                      input_encoder=self.final_encoder,
                                                                      dr_memory_prob=1.0
                                                                      )

    def _create_output_layers(self):
        print('[launch-text] create output projection layer')

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
            self.batch_pred = tf.matmul(self.final_encoder, self.M) + self.b

        with tf.name_scope('loss') as scope:

            c0 = 36705
            c1 = 8331
            c2 = 12922
            s = float(c0+c1+c2)
            norm = []
            norm.append(c0/s)
            norm.append(c1/s)
            norm.append(c2/s)

            if (self.consider_data_bias):
                print('consider data bias in weight calculation')
                self.class_weight = tf.constant(
                    np.ones([3]) - norm, dtype=tf.float32)
                self.weighted_logit = tf.multiply(
                    self.batch_pred, self.class_weight)
                self.batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.weighted_logit, labels=self.y_labels)

            else:
                self.batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.batch_pred, labels=self.y_labels)
                #self.batch_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.batch_pred, labels=self.y_labels )

            self.loss = tf.reduce_mean(self.batch_loss)

    def _create_optimizer(self):
        print('[launch-text] create optimizer')

        with tf.name_scope('text_optimizer') as scope:
            #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, global_step=self.global_step)
            #self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

            opt_func = tf.train.AdamOptimizer(learning_rate=self.lr)
            gvs = opt_func.compute_gradients(self.loss)
            #capped_gvs = [(tf.clip_by_norm(t=grad, clip_norm=1), var) for grad, var in gvs]
            capped_gvs = [(tf.clip_by_value(
                t=grad, clip_value_min=-10, clip_value_max=10), var) for grad, var in gvs]
            self.optimizer = opt_func.apply_gradients(
                grads_and_vars=capped_gvs, global_step=self.global_step)

    def _create_summary(self):
        print('[launch-text] create summary')

        with tf.name_scope('summary'):
            tf.summary.scalar('mean_loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._create_placeholders()
        self._create_embedding()
        self._use_external_embedding()
        self._create_gru_model_bi()

        self._add_ff_layer()

        # if self.o_type: self._add_type_original()
        if self.c_text:
            self._add_context_gru()
        # if self.c_type: self._add_type_context()

        #if self.attn: self._add_attn_self()
        if self.attn:
            self._add_attn_con()

        if self.ltc:
            self._add_LTC_method()

        self._create_output_layers()
        self._create_optimizer()
        self._create_summary()
