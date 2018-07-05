#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 16:17:07 2018

@author: wu
"""

import tensorflow as tf

# 计算Wx_plus_b 的均值与方差,其中axis = [0] 表示想要标准化的维度
img_shape = [128, 32, 32, 64]
Wx_plus_b = tf.Variable(tf.random_normal(img_shape))
axis = list(range(len(img_shape) - 1))
wb_mean, wb_var = tf.nn.moments(Wx_plus_b, axis)

scale = tf.Variable(tf.ones([64]))
offset = tf.Variable(tf.zeros([64]))
variance_epsilon = 0.001
Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, wb_mean, wb_var, offset, scale, variance_epsilon)

Wx_plus_b1 = (Wx_plus_b - wb_mean) / tf.sqrt(wb_var + variance_epsilon)
Wx_plus_b1 = Wx_plus_b1 * scale + offset

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('*** wb_mean ***')
    print(sess.run(wb_mean))
    print('*** wb_var ***')
    print(sess.run(wb_var))
    print('*** Wx_plus_b ***')
    print(sess.run(Wx_plus_b))
    print('**** Wx_plus_b1 ****')
    print(sess.run(Wx_plus_b1))