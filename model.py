#!/usr/bin/env python2
"""This contains model functions.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import tensorflow as tf
import numpy as np
import math
import datetime


class Config(object):
    """This is a wrapper for all configurable parameters for model.

    Attributes:
        batch_size: Integer for the batch size.
        learning_rate: Float for the learning rate.
        data_size: Integer for the number of ECG data samples
        hidden1_size: Integer for the 1st hidden layer size.
        hidden2_size: Integer for the 2nd hidden layer size.
        num_classes: Integer for the number of label classes.
        max_iters: Integer for the number of training iterations.
        model_dir: String for the output model dir.
    """

    def __init__(self):
        self.batch_size = 100
        self.learning_rate = 0.001
        self.data_size = 1500
        self.hidden1_size = 16
        self.hidden2_size = 16


        self.class_weight = [3.0, 0.25, 3.0, 0.5]
        # CNN Parameters
        self.conv1_filters = 16
        self.conv1_kernel = 64
        self.pool1_size = 2
        self.conv2_filters = 8
        self.conv2_kernel = 32
        self.pool2_size = 2
        self.deep_filters = 8
        self.deep_kernel = 12
        self.deep_pool_size = 2
        
        self.frequency_response = False
        self.dropout = 0.5
        
        self.num_classes = 4
        self.k = 5
        
        self.max_iters = 3000
        self.model_dir = './_model'
        self.logs_path = "logs/tf_log_{}".format(datetime.datetime.now())
        

def placeholder_inputs_feedforward(batch_size, feat_dim):
    """Creats the input placeholders for the feedfoward neural network.

    Args:
        batch_size: Integer for the batch size.
        feat_dim: Integer for the feature dimension.

    Returns:
        data_placeholder: data placeholder.
        label_placeholder: Label placeholder.
    """
    # Creates two placeholders.
    # API link (https://www.tensorflow.org/api_docs/python/tf/placeholder).

    data_placeholder = tf.placeholder(name='input_feature', shape=[None, feat_dim], dtype=tf.float32)
    label_placeholder = tf.placeholder(name='output_value', shape=[None], dtype=tf.int32)
    return data_placeholder, label_placeholder


def fill_feed_dict(data_set, batch_size, data_ph, label_ph):
    """Given the data for current step, fills both placeholders.

    Args:
        data_set: The DataSet object.
        batch_size: Integer for the batch size.
        data_ph: The data placeholder, from placeholder_inputs_feedfoward().
        label_ph: The label placehodler, from placeholder_inputs_feedfoward().

    Returns:
        feed_dict: The feed dictionary maps from placeholders to values.
    """
    data_batch, labels_batch = data_set.next_batch(batch_size=batch_size)
    # Creates the feed dictionary.
    return {data_ph: data_batch, label_ph: labels_batch}


def create_CNN(data, config):
    """Creates a convolutional neural network.

    Args:
        data: Data placeholder.
        config: The Config object contains model parameters.

    Returns:
        logits: Output tensor with logits.
    """

    
    input_layer = tf.reshape(data, [-1,config.data_size,1])
    print('Input:',input_layer.shape)
    output_size = config.data_size

    with tf.variable_scope('conv1'):
        conv1 = tf.layers.conv1d(inputs=input_layer,filters=config.conv1_filters, kernel_size=config.conv1_kernel, padding='same',activation=tf.nn.relu)

    with tf.variable_scope('pool1'):
        pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=config.pool1_size, strides=config.pool1_size)

    output_size //= config.pool1_size

    with tf.variable_scope('conv2'):
        conv2 = tf.layers.conv1d(inputs=pool1,filters=config.conv2_filters, kernel_size=config.conv2_kernel,padding='same',activation=tf.nn.relu)

    with tf.variable_scope('pool2'):
        pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=config.pool2_size, strides=config.pool2_size)
    output_size //= config.pool2_size
    
    with tf.variable_scope('flatten'):
        pool2_flat = tf.reshape(pool2, [-1, int(output_size)*config.conv2_filters])

    with tf.variable_scope('dense'):
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    with tf.variable_scope('dropout'):
        dropout = tf.layers.dropout(inputs=dense, rate=config.dropout)

    with tf.variable_scope('logits'):
        logits = tf.layers.dense(inputs=dropout, units=config.num_classes)
                            
    return logits


def create_MLP(data, config):
        # Creates the 1st feed fully-connected layer with ReLU activation.
    with tf.variable_scope('hidden_layer_1'):
        # Creates two variables:
        # 1) hidden1_weights with size [data_size, hidden1_size].
        # 2) hidden1_biases with size [hidden1_size].
        hidden1_weights = tf.get_variable(name='feature_weight', shape=[config.data_size,config.hidden1_size], initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
        hidden1_biases = tf.get_variable(name='feature_bias', shape=[config.hidden1_size], initializer=tf.zeros_initializer())
        
        # Performs feedforward on data using the two variables defined above.
        hidden1 = tf.nn.relu(tf.matmul(data, hidden1_weights) + hidden1_biases)
        
    # Creates the 2nd feed fully-connected layer with ReLU activation.
    with tf.variable_scope('hidden_layer_2'):
        
        # Creates two variables:
        # 1) hidden2_weights with size [hidden1_size, hidden2_size].
        # 2) hidden2_biases with size [hidden2_size].

        hidden2_weights = tf.get_variable(name='feature_weight', shape=[config.hidden1_size, config.hidden2_size], initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
        hidden2_biases = tf.get_variable(name='feature_bias', shape=[config.hidden2_size], initializer=tf.zeros_initializer())

        # Performs feedforward on hidden1 using the two variables defined above.
        hidden2 = tf.nn.relu(tf.matmul(hidden1, hidden2_weights) + hidden2_biases)

    # Creates the pen-ultimate linear layer.
    with tf.variable_scope('logits_layer'):
        # Creates two variables:
        # 1) logits_weights with size [config.hidden2_size, config.num_classes].
        # 2) logits_biases with size [config.num_classes].

        logits_weights = tf.get_variable(name='feature_weight',shape=[config.hidden2_size,config.num_classes], initializer=tf.random_uniform_initializer(-0.1,0,1))
        logits_biases = tf.get_variable(name='feature_bias', shape=[config.num_classes], initializer=tf.zeros_initializer())

        # Performs linear projection on hidden2 using the two variables above.
        logits = tf.matmul(hidden2, logits_weights) + logits_biases

    return logits




def create_Deep_CNN(data, config):
    """Creates a convolutional neural network.

    Args:
        data: Data placeholder.
        config: The Config object contains model parameters.

    Returns:
        logits: Output tensor with logits.
    """

    
    input_layer = tf.reshape(data, [-1,config.data_size,1])
    print('Input:',input_layer.shape)
    output_size = config.data_size

    with tf.variable_scope('conv1'):
        conv1 = tf.layers.conv1d(inputs=input_layer,filters=config.conv1_filters, kernel_size=config.conv1_kernel, padding='same',activation=tf.nn.relu)

    with tf.variable_scope('pool1'):
        pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=config.pool1_size, strides=config.pool1_size)

    output_size //= config.pool1_size

    with tf.variable_scope('conv2'):
        conv2 = tf.layers.conv1d(inputs=pool1,filters=config.conv2_filters, kernel_size=config.conv2_kernel,padding='same',activation=tf.nn.relu)

    with tf.variable_scope('pool2'):
        pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=config.pool2_size, strides=config.pool2_size)
    output_size //= config.pool2_size

    with tf.variable_scope('conv3'):
        conv3 = tf.layers.conv1d(inputs=pool2,filters=config.deep_filters, kernel_size=config.deep_kernel,padding='same',activation=tf.nn.relu)

    with tf.variable_scope('pool3'):
        pool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=config.deep_pool_size, strides=config.deep_pool_size)
    output_size //= config.deep_pool_size

    with tf.variable_scope('conv4'):
        conv4 = tf.layers.conv1d(inputs=pool3,filters=config.deep_filters, kernel_size=config.deep_kernel,padding='same',activation=tf.nn.relu)
    with tf.variable_scope('pool4'):
        pool4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=config.deep_pool_size, strides=config.deep_pool_size)
    output_size //= config.deep_pool_size

    with tf.variable_scope('conv5'):
        conv5 = tf.layers.conv1d(inputs=pool4,filters=config.deep_filters, kernel_size=config.deep_kernel,padding='same',activation=tf.nn.relu)
    with tf.variable_scope('pool5'):
        pool5 = tf.layers.max_pooling1d(inputs=conv5, pool_size=config.deep_pool_size, strides=config.deep_pool_size)
    output_size //= config.deep_pool_size

    with tf.variable_scope('conv6'):
        conv6 = tf.layers.conv1d(inputs=pool5,filters=config.deep_filters, kernel_size=config.deep_kernel,padding='same',activation=tf.nn.relu)
    with tf.variable_scope('pool6'):
        pool6 = tf.layers.max_pooling1d(inputs=conv6, pool_size=config.deep_pool_size, strides=config.deep_pool_size)
    output_size //= config.deep_pool_size

    with tf.variable_scope('conv7'):
        conv7 = tf.layers.conv1d(inputs=pool6,filters=config.deep_filters, kernel_size=config.deep_kernel,padding='same',activation=tf.nn.relu)
    with tf.variable_scope('pool7'):
        pool7 = tf.layers.max_pooling1d(inputs=conv7, pool_size=config.deep_pool_size, strides=config.deep_pool_size)
    output_size //= config.deep_pool_size


    
    with tf.variable_scope('flatten'):
        pool7_flat = tf.reshape(pool7, [-1, int(output_size)*config.deep_filters])

    with tf.variable_scope('dense'):
        dense = tf.layers.dense(inputs=pool7_flat, units=1024, activation=tf.nn.relu)

    with tf.variable_scope('dropout'):
        dropout = tf.layers.dropout(inputs=dense, rate=config.dropout)

    with tf.variable_scope('logits'):
        logits = tf.layers.dense(inputs=dropout, units=config.num_classes)
                            
    return logits




def compute_loss(logits, labels, config):
    """Computes the cross entropy loss between logits and labels.

    Args:
        logits: A [batch_size, num_classes] sized float tensor.
        labels: A [batch_size] sized integer tensor.

    Returns:
        loss: Loss tensor.
    """


    # Computes the cross-entropy loss.
    # API (https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits).

    class_weight = tf.constant(config.class_weight)
    weighted_logits = tf.multiply(logits,class_weight)
    
    with tf.variable_scope('cross_ent_loss'):
        #loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits,weights=class_weight))
        labels_one_hot = tf.one_hot(labels,logits.shape[1])  # logits.shape[1] is the number of classes
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_one_hot, logits=weighted_logits))


    return loss


def evaluation(sess, data_ph, label_ph, data_set, eval_op):
    """Runs one full evaluation and computes accuracy.

    Args:
        sess: The session object.
        data_ph: The data placeholder.
        label_ph: The label placeholder.
        data_set: The DataSet object.
        eval_op: The evaluation accuracy op.

    Returns:
        accuracy: Float scalar for the prediction accuracy.
    """
    # Fills in how you compute the accuracy.
    batch_size = 100
    num_correct = 0

    for i in range(data_set.num_samples//batch_size):
        data, labels = data_set.next_batch(batch_size=batch_size, shuffle=False)
        num_correct += sess.run(eval_op, feed_dict={data_ph:data,label_ph:labels})

    accuracy = num_correct/data_set.num_samples
    
    return accuracy
