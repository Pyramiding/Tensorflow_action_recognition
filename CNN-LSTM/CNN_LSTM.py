#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 20:30:17 2018

@author: wu
"""
import tensorflow as tf
import numpy as np
tf.reset_default_graph() 
from tensorflow.python import pywrap_tensorflow

checkpoint_path = '/home/wu/TF_Project/CNN_LSTM/model_dir/model.ckpt-15000'


def convLayer(x, kHeight, kWidth, strideX, strideY,featureNum, name, padding = "SAME", groups = 1):
    channel = int(x.get_shape()[-1])
    conv = lambda a, b: tf.nn.conv2d(a, b, strides = [1, strideY, strideX, 1], padding = padding)
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("weights", shape = [kHeight, kWidth, channel/groups, featureNum])
        b = tf.get_variable("biases", shape = [featureNum])

        xNew = tf.split(value = x, num_or_size_splits = groups, axis = 3)
        wNew = tf.split(value = w, num_or_size_splits = groups, axis = 3)

        featureMap = [conv(t1, t2) for t1, t2 in zip(xNew, wNew)]
        mergeFeatureMap = tf.concat(axis = 3, values = featureMap)
        out = tf.nn.bias_add(mergeFeatureMap, b)
    return tf.nn.relu(tf.reshape(out, mergeFeatureMap.get_shape().as_list()), name = scope.name)

def fcLayer(x, inputD, outputD, reluFlag, name):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("weights", shape = [inputD, outputD], dtype = "float")
        b = tf.get_variable("biases", [outputD], dtype = "float")
        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out
      
def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],
                          strides = [1, strideX, strideY, 1], padding = padding, name = name)

def LRN(x, R, alpha, beta, name = None, bias = 1.0):
    return tf.nn.local_response_normalization(x, depth_radius = R, alpha = alpha,
                                              beta = beta, bias = bias, name = name)

def dropout(x, keepPro, name = None):
    """dropout"""
    return tf.nn.dropout(x, keepPro, name)

def buildmodel(x):

    conv1 = convLayer(x, 11, 11, 4, 4, 96, "conv1", "VALID")
    lrn1 = LRN(conv1, 2, 2e-05, 0.75, "norm1")
    pool1 = maxPoolLayer(lrn1, 3, 3, 2, 2, "pool1", "VALID")

    conv2 = convLayer(pool1, 5, 5, 1, 1, 256, "conv2", groups = 2)
    lrn2 = LRN(conv2, 2, 2e-05, 0.75, "lrn2")
    pool2 = maxPoolLayer(lrn2, 3, 3, 2, 2, "pool2", "VALID")

    conv3 = convLayer(pool2, 3, 3, 1, 1, 384, "conv3")

    conv4 = convLayer(conv3, 3, 3, 1, 1, 384, "conv4", groups = 2)

    conv5 = convLayer(conv4, 3, 3, 1, 1, 256, "conv5", groups = 2)
    pool5 = maxPoolLayer(conv5, 3, 3, 2, 2, "pool5", "VALID")
   
    #FC layer
    fcIn = tf.reshape(pool5, [-1, 1024])
    fc1 = fcLayer(fcIn, 1024, 4096, True, "fc6")
    x_1 = tf.reshape(fc1, [40, 4096, 1])

    with tf.variable_scope("LSTM") as scope:
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=64, name=scope.name+'/input')
        outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
                rnn_cell,               # cell you have chosen
                x_1,                    # input
                initial_state=None,     # the initial hidden state
                dtype=tf.float32,       # must given if set initial_state = None
                time_major=False,       # False: (batch, time step, input); True: (time step, batch, input)
        )  
        output = tf.layers.dense(outputs[:, -1, :], 6, name=scope.name+'/output')            

    return output



