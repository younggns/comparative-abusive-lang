# -*- coding: utf-8 -*-
#/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import numbers

import tensorflow as tf
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.contrib.rnn import RNNCell
from model_layers import gated_attention
from params import Params

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class GRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               is_training = True):
        
        super(GRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._is_training = is_training

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope = None):
        """Gated recurrent unit (GRU) with nunits cells."""
        if inputs.shape.as_list()[-1] != self._num_units:
            with vs.variable_scope("projection"):
                res = linear(inputs, self._num_units, False, )
        else:
            res = inputs
        with vs.variable_scope("gates"):  # Reset gate and update gate.
            # We start with bias of 1.0 to not reset and not update.
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                dtype = [a.dtype for a in [inputs, state]][0]
                bias_ones = init_ops.constant_initializer(1.0, dtype=dtype)
            value = math_ops.sigmoid(
                linear([inputs, state], 2 * self._num_units, True, bias_ones,
                    self._kernel_initializer))
            r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        with vs.variable_scope("candidate"):
            c = self._activation(
                linear([inputs, r * state], self._num_units, True,
                    self._bias_initializer, self._kernel_initializer))
        #   recurrent dropout as proposed in https://arxiv.org/pdf/1603.05118.pdf (currently disabled)
            #if self._is_training and Params.dropout is not None:
                #c = tf.nn.dropout(c, 1 - Params.dropout)
        new_h = u * state + (1 - u) * c
        return new_h + res, new_h


class gated_attention_Wrapper(RNNCell):
    def __init__(self,
               num_units,
               memory,
               params,
               self_matching = False,
               memory_len = None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               is_training = True,
               use_SRU = False,
               batch_size = 0):
        
        super(gated_attention_Wrapper, self).__init__(_reuse=reuse)
        #cell = SRUCell if use_SRU else GRUCell
        cell = GRUCell
        self._cell = cell(num_units, is_training = is_training)
        self._num_units = num_units
        self._activation = math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._attention = memory
        self._params = params
        self._self_matching = self_matching
        self._memory_len = memory_len
        self._is_training = is_training
        self._batch_size = batch_size

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope = None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with vs.variable_scope("attention_pool"):
            inputs = gated_attention(self._attention,
                                inputs,
                                state,
                                self._num_units,
                                params = self._params,
                                self_matching = self._self_matching,
                                memory_len = self._memory_len,
                                batch_size = self._batch_size)
        output, new_state = self._cell(inputs, state, scope)
        return output, new_state


def linear(args,
            output_size,
            bias,
            bias_initializer=None,
            kernel_initializer=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
      Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_initializer: starting value to initialize the bias
          (default is all zeros).
        kernel_initializer: starting value to initialize the weight.
      Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
      Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
              raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        weights = vs.get_variable(
            _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
            dtype=dtype,
            initializer=kernel_initializer)
        if len(args) == 1:
            res = math_ops.matmul(args[0], weights)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), weights)
        if not bias:
            return res
        with vs.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            if bias_initializer is None:
                bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
            biases = vs.get_variable(
                _BIAS_VARIABLE_NAME, [output_size],
                dtype=dtype,
                initializer=bias_initializer)
        return nn_ops.bias_add(res, biases)
