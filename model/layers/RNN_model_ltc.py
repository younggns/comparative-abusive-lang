# -*- coding: utf-8 -*-

import tensorflow as tf

'''
desc : latent topic cluster method

input :
   - batch_size        :
   - topic             :    : # of topics
   - memory_dim        : dim of each topic
   - input_hidden_dim  : dim of input vector
   - input_encoder     : [batch, input_hidden_dim]
   - dr_memory_prob    : dropout ratio for memory

output :
   - final_encoder : LTC applied vector [batch, vector(== concat(original, topic_mem)]
   - final_encoder_dimension :
'''


def ltc(
        batch_size,
        topic_size,
        memory_dim,
        input_hidden_dim,
        input_encoder,
        dr_memory_prob=1.0):
    print('[launch : model_ltc] s.y. Latent Topic Cluster method')

    with tf.name_scope('ltc') as scope:

        # memory space for latent topic
        memory = tf.get_variable("latent_topic_memory",
                                 shape=[topic_size, memory_dim],
                                 initializer=tf.orthogonal_initializer()
                                 )

        memory_W = tf.Variable(
            tf.random_uniform(
                [
                    input_hidden_dim,
                    memory_dim],
                minval=-0.25,
                maxval=0.25,
                dtype=tf.float32,
                seed=None),
            name="memory_projection_W")

        memory_W = tf.nn.dropout(memory_W, keep_prob=dr_memory_prob)
        memory_bias = tf.Variable(
            tf.zeros(
                [memory_dim],
                dtype=tf.float32),
            name="memory_projection_bias")

        topic_sim_project = tf.matmul(input_encoder, memory_W) + memory_bias

        # context - topic  similairty
        topic_sim = tf.matmul(topic_sim_project, memory, transpose_b=True)

        # add non-linearity
        topic_sim = tf.tanh(topic_sim)

        # normalize
        topic_sim_sigmoid_softmax = tf.nn.softmax(logits=topic_sim, dim=-1)

        shaped_input = tf.reshape(topic_sim_sigmoid_softmax, [
                                  batch_size, topic_size])

        topic_sim_mul_memory = tf.scan(
            lambda a,
            x: tf.multiply(
                tf.transpose(memory),
                x),
            shaped_input,
            initializer=tf.transpose(memory))
        tmpT = tf.reduce_sum(topic_sim_mul_memory, axis=-1, keep_dims=True)
        rsum = tf.squeeze(tmpT)

        # final context
        final_encoder = tf.concat([input_encoder, rsum], axis=-1)

        final_encoder_dimension = input_hidden_dim + memory_dim   # concat 으로 늘어났음

        return final_encoder, final_encoder_dimension
