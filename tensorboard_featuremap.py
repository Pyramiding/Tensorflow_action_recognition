#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 15:50:29 2018

@author: wu
"""

import tensorflow as tf
import alexnet
import img_convert
import numpy as np

with tf.variable_scope('conv1') as scope:
    images=img_convert.convert_3_2_4_dims(img_convert.convert("E:/test"))
    images=images.astype(np.float32)
    kernel = cifar10._variable_with_weight_decay('weights',shape=[5, 5, 3, 64],stddev=5e-2,wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = cifar10._variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    cifar10._activation_summary(conv1)
    with tf.variable_scope('visualization'):
        x_min = tf.reduce_min(kernel)
        x_max = tf.reduce_max(kernel)
        kernel_0_to_1 = (kernel - x_min) / (x_max - x_min)
    # to tf.image_summary format [batch_size, height, width, channels]
        kernel_transposed = tf.transpose (kernel_0_to_1, [3, 0, 1, 2])
    # this will display random 3 filters from the 64 in conv1
        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('train')
        img0=tf.summary.image('conv1/filters', kernel_transposed, max_outputs=6)
        layer1_image1 = conv1[0:1, :, :, 0:16]
        layer1_image1 = tf.transpose(layer1_image1, perm=[3,1,2,0])
        img1=tf.summary.image("filtered_images_layer1", layer1_image1, max_outputs=16)
        writer.add_summary(sess.run(img0))
        writer.add_summary(sess.run(img1))
        writer.close()
        sess.close()