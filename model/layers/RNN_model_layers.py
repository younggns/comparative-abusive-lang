# -*- coding: utf-8 -*-
#/usr/bin/python3

import tensorflow as tf
import numpy as np

from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import RNNCell
from params import Params

def get_attn_params(attn_size,initializer = tf.truncated_normal_initializer):
    with tf.variable_scope("attention_weights"):
        params = {
                "W_v_P_2":tf.get_variable("W_v_P_2",dtype = tf.float32, shape = (2 * attn_size, attn_size), initializer = initializer()),
                "W_g":tf.get_variable("W_g",dtype = tf.float32, shape = (4 * attn_size, 4 * attn_size), initializer = initializer()),
                "W_v_Phat":tf.get_variable("W_v_Phat",dtype = tf.float32, shape = (2 * attn_size, attn_size), initializer = initializer()),
                "v":tf.get_variable("v",dtype = tf.float32, shape = (attn_size), initializer =initializer())}
        return params

def apply_dropout(inputs, size = None, is_training = True, dropout=None):
    if dropout is None:
        return inputs
    elif is_training:
        return tf.contrib.rnn.DropoutWrapper(inputs,
                                            output_keep_prob = dropout,
                                            # variational_recurrent = True,
                                            # input_size = size,
                                            dtype = tf.float32)
    else:
        return inputs

# cell instance
def gru_cell(units):
    return tf.contrib.rnn.GRUCell(num_units=units)

# cell instance with drop-out wrapper applied
def gru_drop_out_cell(dr_prob=1.0, units=0):
    return tf.contrib.rnn.DropoutWrapper(gru_cell(units), 
                                         input_keep_prob=dr_prob,
                                         output_keep_prob=1.0,
                                         dtype = tf.float32
                                        )

    
def bidirectional_GRU(inputs, inputs_len, cell = None, cell_fn = tf.contrib.rnn.GRUCell, units = 0, layers = 1, scope = "Bidirectional_GRU", output = 0, is_training = True, reuse = None, dr_prob=1.0, is_bidir=False):

    with tf.variable_scope(scope, reuse = reuse, initializer=tf.orthogonal_initializer()):
        if cell is not None:
            (cell_fw, cell_bw) = cell
        else:
            shapes = inputs.get_shape().as_list()
            if len(shapes) > 3:
                inputs = tf.reshape(inputs,(shapes[0]*shapes[1],shapes[2],-1))
                inputs_len = tf.reshape(inputs_len,(shapes[0]*shapes[1],))

            # if no cells are provided, use standard GRU cell implementation
            if layers > 1:
                cell_fw = MultiRNNCell([apply_dropout(cell_fn(units), size = inputs.shape[-1] if i == 0 else units, is_training = is_training, dropout=dr_prob) for i in range(layers)])
                if is_bidir: 
                    cell_bw = MultiRNNCell([apply_dropout(cell_fn(units), size = inputs.shape[-1] if i == 0 else units, is_training = is_training, dropout=dr_prob) for i in range(layers)])
            else:
                cell_fw = apply_dropout(cell_fn(units), size = inputs.shape[-1], is_training = is_training)
                if is_bidir: 
                    cell_bw = apply_dropout(cell_fn(units), size = inputs.shape[-1], is_training = is_training)
                
        if is_bidir:        
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                                                            cell_fw = cell_fw,
                                                            cell_bw = cell_bw,
                                                            inputs = inputs,
                                                            sequence_length = inputs_len,
                                                            dtype = tf.float32,
                                                            scope = scope,
                                                            time_major=False
                                                        )
            if Params.reverse_bw :
                fw = outputs[0]
                bw = tf.reverse_sequence(outputs[1], seq_lengths = inputs_len, seq_axis = 1)
                outputs = (fw, bw)
                               
            return tf.concat(outputs, 2), tf.concat(states, axis=1)
                   
        
        else:
            outputs, states = tf.nn.dynamic_rnn(
                                                cell=cell_fw,
                                                inputs= inputs,
                                                dtype=tf.float32,
                                                sequence_length=inputs_len,
                                                scope = scope,
                                                time_major=False)
            return outputs, states


def attention_rnn(inputs, inputs_len, units, attn_cell, bidirection = True, scope = "gated_attention_rnn", is_training = True, dr_prob=1.0, is_bidir=False):
    with tf.variable_scope(scope):
        if bidirection:
            outputs, last_states = bidirectional_GRU(
                                        inputs = inputs,
                                        inputs_len  = inputs_len,
                                        cell = attn_cell,
                                        units  = units,
                                        layers = Params.self_matching_layers,
                                        scope = scope + "_bidirectional",
                                        reuse = False,
                                        output = 0,
                                        is_training = True,
                                        dr_prob = dr_prob,
                                        is_bidir = True
                                       )
        else:
            outputs, last_states = tf.nn.dynamic_rnn(attn_cell, inputs,
                                            sequence_length = inputs_len,
                                            dtype=tf.float32)
            
        return outputs, last_states


def gated_attention(memory, inputs, states, units, params, self_matching = False, memory_len = None, scope="gated_attention", batch_size=0):
    with tf.variable_scope(scope):
        weights, W_g = params        
        inputs_ = [memory, inputs]
        states = tf.reshape(states,(batch_size, units))
        if not self_matching:
            inputs_.append(states)
        scores = attention(inputs_, units, weights, memory_len = memory_len, batch_size = batch_size)
        scores = tf.expand_dims(scores,-1)
        attention_pool = tf.reduce_sum(scores * memory, 1)
        inputs = tf.concat((inputs,attention_pool),axis = 1)
        g_t = tf.sigmoid(tf.matmul(inputs,W_g))
        return g_t * inputs

    
def mask_attn_score(score, memory_sequence_length, score_mask_value = -1e8):
    score_mask = tf.sequence_mask(
        memory_sequence_length, maxlen=score.shape[1], dtype=tf.bool)
    score_mask_values = score_mask_value * tf.ones_like(score)
    return tf.where(score_mask, score, score_mask_values)


def attention(inputs, units, weights, scope = "attention", memory_len = None, reuse = None, batch_size=0):
    with tf.variable_scope(scope, reuse = reuse):
        outputs_ = []
        weights, v = weights
        for i, (inp,w) in enumerate(zip(inputs,weights)):
            shapes = inp.shape.as_list()
            inp = tf.reshape(inp, (-1, shapes[-1]))
            if w is None:
                w = tf.get_variable("w_%d"%i, dtype = tf.float32, shape = [shapes[-1], units], initializer = tf.contrib.layers.xavier_initializer())
            outputs = tf.matmul(inp, w)
            # Hardcoded attention output reshaping. Equation (4), (8), (9) and (11) in the original paper.
            if len(shapes) > 2:
                outputs = tf.reshape(outputs, (shapes[0], shapes[1], -1))
            elif len(shapes) == 2 and shapes[0] is batch_size:
                outputs = tf.reshape(outputs, (shapes[0],1,-1))
            else:
                outputs = tf.reshape(outputs, (1, shapes[0],-1))
            outputs_.append(outputs)
        outputs = sum(outputs_)

        b = tf.get_variable("b", shape = outputs.shape[-1], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
        outputs += b
            
        scores = tf.reduce_sum(tf.tanh(outputs) * v, [-1])
        if memory_len is not None:
            scores = mask_attn_score(scores, memory_len)
        return tf.nn.softmax(scores) # all attention output is softmaxed now
    

