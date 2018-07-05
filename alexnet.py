#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 20:24:20 2018

@author: wu
"""

import tensorflow as tf
import numpy as np

tf.reset_default_graph() 

def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    """max-pooling"""
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],
                          strides = [1, strideX, strideY, 1], padding = padding, name = name)

def dropout(x, keepPro, name = None):
    """dropout"""
    return tf.nn.dropout(x, keepPro, name)

def LRN(x, R, alpha, beta, name = None, bias = 1.0):
    """LRN"""
    return tf.nn.local_response_normalization(x, depth_radius = R, alpha = alpha,
                                              beta = beta, bias = bias, name = name)

def fcLayer(x, inputD, outputD, reluFlag, name):
    """fully-connect"""
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("weights", shape = [inputD, outputD], dtype = "float")
        b = tf.get_variable("biases", [outputD], dtype = "float")
        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out

def convLayer(x, kHeight, kWidth, strideX, strideY,
              featureNum, name, padding = "SAME", groups = 1):
    """convolution"""
    channel = int(x.get_shape()[-1])
    conv = lambda a, b: tf.nn.conv2d(a, b, strides = [1, strideY, strideX, 1], padding = padding)
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("weights", shape = [kHeight, kWidth, channel/groups, featureNum])
        b = tf.get_variable("biases", shape = [featureNum])

        xNew = tf.split(value = x, num_or_size_splits = groups, axis = 3)
        wNew = tf.split(value = w, num_or_size_splits = groups, axis = 3)

        featureMap = [conv(t1, t2) for t1, t2 in zip(xNew, wNew)]
        mergeFeatureMap = tf.concat(axis = 3, values = featureMap)
        # print mergeFeatureMap.shape
        out = tf.nn.bias_add(mergeFeatureMap, b)
    return tf.nn.relu(tf.reshape(out, mergeFeatureMap.get_shape().as_list()), name = scope.name)

def losses(logits, labels):
    with tf.variable_scope("loss") as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, name="xentropy_per_example")
        loss = tf.reduce_mean(cross_entropy, name="loss")
        tf.summary.scalar(scope.name + "loss", loss)
    return loss


def training(loss, learning_rate):
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    with tf.variable_scope("accuracy") as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + "accuracy", accuracy)
    return accuracy


def Batch_Normalization(x):
    """Batch Normalization layer"""
    x_mean, x_var = tf.nn.moments(x, axes=[0]) 
    scale = tf.Variable(tf.ones([4096]))
    offset = tf.Variable(tf.zeros([4096]))
    variance_epsilon = 0.001
    BN_x = tf.nn.batch_normalization(x, x_mean, x_var, offset, scale, variance_epsilon)
    return BN_x


class alexNet(object):
    """alexNet model"""
    def __init__(self, x, keepPro=1.0, classNum=6, skip_layer=['conv5','fc6','fc7','fc8'], weights_path = 'DEFAULT'):
        self.X = x
        self.KEEPPRO = keepPro
        self.CLASSNUM = classNum
        self.SKIP_LAYER = skip_layer
        self.WEIGHTS_PATH = weights_path
        self.buildCNN()
        
        #fine-tune choice
        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = '/home/wu/TF_Project/action/bvlc_alexnet.npy'
            #self.WEIGHTS_PATH = '/home/wu/TF_Project/action/model_tfrecord_sample/model.ckpt-1000'
        else:
            self.WEIGHTS_PATH = weights_path
   
    def load_initial_weights(self, session):  
        weights_dict = np.load(self.WEIGHTS_PATH, encoding = 'bytes').item()
        for op_name in weights_dict:
            if op_name not in self.SKIP_LAYER:
                with tf.variable_scope(op_name, reuse = True):  
                    for data in weights_dict[op_name]:            
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable = False)
                            session.run(var.assign(data))
                        else:              
                            var = tf.get_variable('weights', trainable = False)
                            session.run(var.assign(data))
    
    def buildCNN(self):
        """
        input 100*100*3
        conv1 23*23*96
        pool1 11*11*96
        conv2 11*11*256
        pool2 5*5*256
        conv3 5*5*384
        conv4 5*5*384
        conv5 5*5*256
        pool5 2*2*256
        fc1 1024->4096
        fc2 4096->4096
        fc3 4096->6
        """
        self.conv1 = convLayer(self.X, 11, 11, 4, 4, 96, "conv1", "VALID")
        """
        split = tf.split(self.conv1,num_or_size_splits=96,axis=3)
        tf.summary.image("conv1_features",split[0],10)
        """
        lrn1 = LRN(self.conv1, 2, 2e-05, 0.75, "norm1")
        pool1 = maxPoolLayer(lrn1, 3, 3, 2, 2, "pool1", "VALID")

        conv2 = convLayer(pool1, 5, 5, 1, 1, 256, "conv2", groups = 2)
        lrn2 = LRN(conv2, 2, 2e-05, 0.75, "lrn2")
        pool2 = maxPoolLayer(lrn2, 3, 3, 2, 2, "pool2", "VALID")

        conv3 = convLayer(pool2, 3, 3, 1, 1, 384, "conv3")

        conv4 = convLayer(conv3, 3, 3, 1, 1, 384, "conv4", groups = 2)

        conv5 = convLayer(conv4, 3, 3, 1, 1, 256, "conv5", groups = 2)
        pool5 = maxPoolLayer(conv5, 3, 3, 2, 2, "pool5", "VALID")

        fcIn = tf.reshape(pool5, [-1, 1024])
        fc1 = fcLayer(fcIn, 1024, 4096, True, "fc6")
        #dropout1 = dropout(fc1, self.KEEPPRO)
        dropout1 = Batch_Normalization(fc1)

        fc2 = fcLayer(dropout1, 4096, 4096, True, "fc7")
        #dropout2 = dropout(fc2, self.KEEPPRO)
        dropout2 = Batch_Normalization(fc2)

        self.fc3 = fcLayer(dropout2, 4096, self.CLASSNUM, True, "fc8")

