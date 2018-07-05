#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 21:20:45 2018

@author: wu
"""

import tensorflow as tf  
import pandas as pd  
import numpy as np  
from sklearn.metrics import confusion_matrix  
from tensorflow.python.ops import rnn,rnn_cell  
import time  
from datetime import timedelta  
import math  
import random  
import os  
import cv2  
from utils import *  
  
num_channels = 3  
img_size = 227  
num_unitlstm=64  
img_size_flat = img_size * img_size * num_channels  
  
img_shape = (img_size, img_size)  
  
  
train_tfdata=['cifar10_train_1.tfrecords','cifar10_train_2.tfrecords']  
test_tfdata=['cifar10_test.tfrecords']  
  
classes=range(10)  
num_classes = len(classes)  
starter_learning_rate = 0.001  
  
#test_path = data_dir  
  
global_step = tf.Variable(0, trainable=False)  
  
x = tf.placeholder(tf.float32,shape=[None,img_size,img_size,num_channels])  
  
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])  
  
y= tf.placeholder(tf.float32,shape=[None])  
  
y_true= tf.one_hot(tf.cast(y,tf.int32),num_classes)  
  
y_true_cls = tf.argmax(y_true, dimension=1)  
  
phase_train=tf.placeholder(bool)  
  
keep_prob = tf.placeholder(tf.float32)  
  
  
print"creating network model"  
  
with tf.name_scope('conv1') as scope:  
  kernel = tf.get_variable(name='conv1',shape=[11,11,3,96],initializer=tf.contrib.layers.xavier_initializer_conv2d())  
  conv = tf.nn.conv2d(x, kernel, [1, 4, 4, 1], padding='SAME')  
  biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),trainable=True, name='biases')  
  bias = tf.nn.bias_add(conv, biases)  
  bias=batch_norm(bias, phase_train)  
  conv1 = tf.nn.relu(bias, name=scope)  
  
pool1 = tf.nn.max_pool(conv1,ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1],padding='VALID',name='pool1')  
  
  
# conv2  
with tf.name_scope('conv2') as scope:  
  kernel = tf.get_variable('conv2',shape=[5,5,96,256],initializer=tf.contrib.layers.xavier_initializer_conv2d())  
  conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')  
  biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),trainable=True, name='biases')  
  bias = tf.nn.bias_add(conv, biases)  
  bias=batch_norm(bias, phase_train)  
  conv2 = tf.nn.relu(bias, name=scope)  
  
  
# pool2  
pool2 = tf.nn.max_pool(conv2,ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1],padding='VALID',name='pool2')  
  
  
# conv3  
with tf.name_scope('conv3') as scope:  
  kernel = tf.get_variable('conv3',shape=[3,3,256,384],initializer=tf.contrib.layers.xavier_initializer_conv2d())  
  conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')  
  biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),trainable=True, name='biases')  
  bias = tf.nn.bias_add(conv, biases)  
  bias=batch_norm(bias, phase_train)  
  conv3 = tf.nn.relu(bias, name=scope)  
  
  
# conv4  
with tf.name_scope('conv4') as scope:  
  kernel = tf.get_variable('conv4',shape=[3,3,384,384],initializer=tf.contrib.layers.xavier_initializer_conv2d())  
  conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')  
  biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),trainable=True, name='biases')  
  bias = tf.nn.bias_add(conv, biases)  
  bias=batch_norm(bias, phase_train)  
  conv4 = tf.nn.relu(bias, name=scope)  
  
  
# conv5  
with tf.name_scope('conv5') as scope:  
  kernel = tf.get_variable('conv5',shape=[3,3,384,256],initializer=tf.contrib.layers.xavier_initializer_conv2d())  
  conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')  
  biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),trainable=True, name='biases')  
  bias = tf.nn.bias_add(conv, biases)  
  bias=batch_norm(bias, phase_train)  
  conv5 = tf.nn.relu(bias, name=scope)  
    
pool5 = tf.nn.max_pool(conv5,ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1],padding='VALID',name='pool5')  
  
  
layer_flat, num_features = flatten_layer(pool5)  
  
layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features,num_outputs=num_features,name='fc1',use_relu=True)  
  
layer_fc1_split=tf.split(1,256,layer_fc1)  
  
with tf.variable_scope('lstm'):  
  
    lstm=rnn_cell.BasicLSTMCell(num_unitlstm,forget_bias=1.0)  
  
with tf.variable_scope('RNN'):  
  
    output,state=rnn.rnn(lstm,layer_fc1_split,dtype=tf.float32)  
  
rnn_output=tf.concat(1,output)  
  
  
layer_fc3 = new_fc_layer(input=rnn_output,num_inputs=num_unitlstm*256,num_outputs=num_classes,name='fc3',use_relu=False)  
  
y_pred = tf.nn.softmax(layer_fc3)  
  
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc3,labels=y_true)  
cost = tf.reduce_mean(cross_entropy)  
tf.summary.scalar('cost', cost)  
  
#learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,  
#                                           10000, 0.1, staircase=True)  
#tf.summary.scalar('learning_rate', learning_rate)  
  
#optimizer =tf.train.MomentumOptimizer(learning_rate,momentum=0.9).minimize(cost, global_step=global_step)  
  
  
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)  
  
y_pred_cls = tf.argmax(y_pred, dimension=1)  
correct_prediction = tf.equal(y_pred_cls, y_true_cls)  
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
tf.summary.scalar('accuracy', accuracy)  
  
train_img,train_label=load_tfrecords(train_tfdata,img_size,True)  
  
test_img,test_label=load_tfrecords(test_tfdata,img_size,False)  
  
train_img_batch,train_label_batch=tf.train.shuffle_batch([train_img,train_label],batch_size=64,capacity=3000,min_after_dequeue=1000)  
  
test_img_batch,test_label_batch=tf.train.shuffle_batch([test_img,test_label],batch_size=32,capacity=3000,min_after_dequeue=1000)  
  
print" starting Session"  
  
sess = tf.Session()  
  
sess.run(tf.global_variables_initializer())  
  
saver = tf.train.Saver()  
summary_op = tf.summary.merge_all()  
  
train_summary_writer = tf.summary.FileWriter('rnnsummary' + 'train', sess.graph)  
validation_summary_writer = tf.summary.FileWriter('rnnsummary'+ 'validation', sess.graph)  
  
threads=tf.train.start_queue_runners(sess=sess)  
  
print "loadinf train_batch"  
  
num_iterations=25000  
  
total_iterations = 0  
  
start_time = time.time()  
  
acc_total=[]  
  
for i in range(total_iterations,total_iterations + num_iterations):  
  
    train_batch = sess.run([train_img_batch, train_label_batch])  
      
  
    feed_dict_train={x: train_batch[0], y: train_batch[1],phase_train:True,keep_prob:0.5}  
      
    summary_train,_,train_acc=sess.run([summary_op,optimizer,accuracy], feed_dict=feed_dict_train)  
    train_summary_writer.add_summary(summary_train, i)  
  
    if (i+1) % 20 == 0:  
  
        test_batch = sess.run([test_img_batch,test_label_batch])  
  
        feed_dict_validate={x: test_batch[0], y: test_batch[1],phase_train:False,keep_prob:1}  
          
        summary_valid,val_loss ,val_acc= sess.run([summary_op,cost,accuracy], feed_dict=feed_dict_validate)  
        validation_summary_writer.add_summary(summary_valid, i)  
  
        msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"  
        print(msg.format(i + 1, train_acc, val_acc, val_loss))  
        acc_total.append(val_acc)  
       
    if(i+1)%10000==0:  
        save_path = saver.save(sess, "model.ckpt")  
          
np.save('acc_total',acc_total)  
          
  
end_time = time.time()  
  
time_dif = end_time - start_time  
  
print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))  