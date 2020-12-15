# -*- coding: utf-8 -*-
# /usr/bin/python3

import tensorflow as tf
import numpy as np

from tensorflow.compat.v1.nn.rnn_cell import MultiRNNCell
from layers.RNN_params import Params

#from zoneout import ZoneoutWrapper
'''
attention weights from https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf
W_u^Q.shape:    (2 * attn_size, attn_size)
W_u^P.shape:    (2 * attn_size, attn_size)
W_v^P.shape:    (attn_size, attn_size)
W_g.shape:      (4 * attn_size, 4 * attn_size)
W_h^P.shape:    (2 * attn_size, attn_size)
W_v^Phat.shape: (2 * attn_size, attn_size)
W_h^a.shape:    (2 * attn_size, attn_size)
W_v^Q.shape:    (attn_size, attn_size)
'''


def get_attn_params(
        attn_size,
        initializer=tf.compat.v1.truncated_normal_initializer):
    '''
    Args:
        attn_size: the size of attention specified in https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf
        initializer: the author of the original paper used gaussian initialization however I found xavier converge faster

    Returns:
        params: A collection of parameters used throughout the layers
    '''
    with tf.compat.v1.variable_scope("attention_weights"):
        params = {
            # 0 case "W_u_Q":tf.get_variable("W_u_Q",dtype = tf.float32, shape = (2 * attn_size, attn_size), initializer = initializer()),
            # "W_ru_Q":tf.get_variable("W_ru_Q",dtype = tf.float32, shape = (2 * attn_size, 2 * attn_size), initializer = initializer()),
            # 0 case ""W_u_P":tf.get_variable("W_u_P",dtype = tf.float32, shape = (2 * attn_size, attn_size), initializer = initializer()),
            # 0 case ""W_v_P":tf.get_variable("W_v_P",dtype = tf.float32, shape = (attn_size, attn_size), initializer = initializer()),
            "W_v_P_2": tf.compat.v1.get_variable("W_v_P_2", dtype=tf.float32, shape=(2 * attn_size, attn_size), initializer=initializer()),
            "W_g": tf.compat.v1.get_variable("W_g", dtype=tf.float32, shape=(4 * attn_size, 4 * attn_size), initializer=initializer()),
            # "W_h_P":tf.get_variable("W_h_P",dtype = tf.float32, shape = (2 * attn_size, attn_size), initializer = initializer()),
            "W_v_Phat": tf.compat.v1.get_variable("W_v_Phat", dtype=tf.float32, shape=(2 * attn_size, attn_size), initializer=initializer()),
            # "W_h_a":tf.get_variable("W_h_a",dtype = tf.float32, shape = (2 * attn_size, attn_size), initializer = initializer()),
            # "W_v_Q":tf.get_variable("W_v_Q",dtype = tf.float32, shape = (attn_size,  attn_size), initializer = initializer()),
            "v": tf.compat.v1.get_variable("v", dtype=tf.float32, shape=(attn_size), initializer=initializer())}
        return params


def apply_dropout(
        inputs,
        size=None,
        is_training=True,
        input_keep_prob=1.0,
        output_keep_prob=1.0):
    '''
    Implementation of Zoneout from https://arxiv.org/pdf/1606.01305.pdf
    '''
    if ((input_keep_prob == 1.0) & (output_keep_prob == 1.0)):
        return inputs
    # if Params.zoneout is not None:
    # return ZoneoutWrapper(inputs, state_zoneout_prob= Params.zoneout,
    # is_training = is_training)
    elif is_training:
        return tf.compat.v1.nn.rnn_cell.DropoutWrapper(
            inputs,
            input_keep_prob=input_keep_prob,
            output_keep_prob=output_keep_prob,
            # variational_recurrent = True,
            # input_size = size,
            dtype=tf.float32)
    else:
        return inputs


def bidirectional_GRU(
        inputs,
        inputs_len,
        cell=None,
        cell_fn=tf.compat.v1.nn.rnn_cell.GRUCell,
        units=0,
        layers=1,
        scope="Bidirectional_GRU",
        output=0,
        is_training=True,
        reuse=None,
        dr_input_keep_prob=1.0,
        dr_output_keep_prob=1.0,
        is_bidir=False):
    '''
    Bidirectional recurrent neural network with GRU cells.

    Args:
        inputs:     rnn input of shape (batch_size, timestep, dim)
        inputs_len: rnn input_len of shape (batch_size, )
        cell:       rnn cell of type RNN_Cell.
        output:     [ batch, step, dim (fw;bw) ], [ batch, dim (fw;bw) ]
    '''
    with tf.compat.v1.variable_scope(scope, reuse=reuse, initializer=tf.compat.v1.orthogonal_initializer()):
        if cell is not None:
            (cell_fw, cell_bw) = cell
        else:
            shapes = inputs.get_shape().as_list()
            if len(shapes) > 3:
                print('input reshaped!!!')
                inputs = tf.reshape(
                    inputs, (shapes[0] * shapes[1], shapes[2], -1))
                inputs_len = tf.reshape(inputs_len, (shapes[0] * shapes[1],))

            # if no cells are provided, use standard GRU cell implementation
            if layers > 1:
                cell_fw = MultiRNNCell([apply_dropout(cell_fn(units),
                                                      size=inputs.shape[-1] if i == 0 else units,
                                                      is_training=is_training,
                                                      input_keep_prob=dr_input_keep_prob,
                                                      output_keep_prob=dr_output_keep_prob) for i in range(layers)])
                if is_bidir:
                    cell_bw = MultiRNNCell([apply_dropout(cell_fn(units),
                                                          size=inputs.shape[-1] if i == 0 else units,
                                                          is_training=is_training,
                                                          input_keep_prob=dr_input_keep_prob,
                                                          output_keep_prob=dr_output_keep_prob) for i in range(layers)])
            else:
                cell_fw = apply_dropout(
                    cell_fn(units), size=inputs.shape[-1], is_training=is_training)
                if is_bidir:
                    cell_bw = apply_dropout(
                        cell_fn(units), size=inputs.shape[-1], is_training=is_training)

        if is_bidir:
            outputs, states = tf.compat.v1.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=inputs,
                sequence_length=inputs_len,
                dtype=tf.float32,
                scope=scope,
                time_major=False
            )
            if Params.reverse_bw:
                fw = outputs[0]
                bw = tf.reverse_sequence(
                    input=outputs[1], seq_lengths=inputs_len, seq_axis=1)
                outputs = (fw, bw)

            return tf.concat(outputs, 2), tf.concat(states, axis=1)

        else:
            outputs, states = tf.compat.v1.nn.dynamic_rnn(
                cell=cell_fw,
                inputs=inputs,
                dtype=tf.float32,
                sequence_length=inputs_len,
                scope=scope,
                time_major=False)
            return outputs, states


def attention_rnn(
        inputs,
        inputs_len,
        units,
        attn_cell,
        bidirection=True,
        scope="gated_attention_rnn",
        is_training=True,
        dr_prob=1.0,
        is_bidir=False):
    with tf.compat.v1.variable_scope(scope):
        if bidirection:
            outputs, last_states = bidirectional_GRU(
                inputs=inputs,
                inputs_len=inputs_len,
                cell=attn_cell,
                units=units,
                layers=Params.self_matching_layers,
                scope=scope + "_bidirectional",
                reuse=False,
                output=0,
                is_training=True,
                dr_input_keep_prob=dr_prob,
                is_bidir=True
            )
        else:
            outputs, last_states = tf.compat.v1.nn.dynamic_rnn(
                attn_cell, inputs, sequence_length=inputs_len, dtype=tf.float32)

        return outputs, last_states


def gated_attention(
        memory,
        inputs,
        states,
        units,
        params,
        self_matching=False,
        memory_len=None,
        scope="gated_attention",
        batch_size=0):
    with tf.compat.v1.variable_scope(scope):
        weights, W_g = params
        inputs_ = [memory, inputs]
        states = tf.reshape(states, (int(batch_size), int(units)))
        if not self_matching:
            inputs_.append(states)
        scores = attention(inputs_, int(units), weights,
                           memory_len=memory_len, batch_size=int(batch_size))
        scores = tf.expand_dims(scores, -1)
        attention_pool = tf.reduce_sum(input_tensor=scores * memory, axis=1)
        inputs = tf.concat((inputs, attention_pool), axis=1)
        g_t = tf.sigmoid(tf.matmul(inputs, W_g))
        return g_t * inputs


def mask_attn_score(score, memory_sequence_length, score_mask_value=-1e8):
    score_mask = tf.sequence_mask(
        memory_sequence_length, maxlen=score.shape[1], dtype=tf.bool)
    score_mask_values = score_mask_value * tf.ones_like(score)
    return tf.compat.v1.where(score_mask, score, score_mask_values)


def attention(
        inputs,
        units,
        weights,
        scope="attention",
        memory_len=None,
        reuse=None,
        batch_size=0):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        outputs_ = []
        weights, v = weights
        for i, (inp, w) in enumerate(zip(inputs, weights)):
            shapes = inp.shape.as_list()
            inp = tf.reshape(inp, (-1, shapes[-1]))
            if w is None:
                w = tf.compat.v1.get_variable("w_%d" % i,
                                              dtype=tf.float32,
                                              shape=[shapes[-1],
                                                     units],
                                              initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                                                                          mode="fan_avg",
                                                                                                          distribution="uniform"))
            outputs = tf.matmul(inp, w)
            # Hardcoded attention output reshaping. Equation (4), (8), (9) and
            # (11) in the original paper.
            if len(shapes) > 2:
                outputs = tf.reshape(outputs, (shapes[0], shapes[1], -1))
            elif len(shapes) == 2 and shapes[0] is batch_size:
                outputs = tf.reshape(outputs, (shapes[0], 1, -1))
            else:
                outputs = tf.reshape(outputs, (1, shapes[0], -1))
            outputs_.append(outputs)
        outputs = sum(outputs_)

        b = tf.compat.v1.get_variable("b",
                                      shape=outputs.shape[-1],
                                      dtype=tf.float32,
                                      initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                                                                  mode="fan_avg",
                                                                                                  distribution="uniform"))
        outputs += b

        scores = tf.reduce_sum(input_tensor=tf.tanh(outputs) * v, axis=[-1])
        if memory_len is not None:
            scores = mask_attn_score(scores, memory_len)
        return tf.nn.softmax(scores)  # all attention output is softmaxed now
