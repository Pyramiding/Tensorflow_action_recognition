#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 19:42:31 2018

@author: wu
"""

import tensorflow as tf
import numpy as np
import scipy.io as sio  
import matplotlib.pyplot as plt
import tfrecord
import alexnet

BATCH_SIZE=100
IMG_W=100
IMG_H=100

train_filename = '/home/wu/TF_Project/action/sample_TFrecord/train1.tfrecords'
#dir = '/home/wu/TF_Project/action/feature_tensorboard/'
dir = '/home/wu/TF_Project/action/features/'
train_img, train_label = tfrecord.read_and_decode(train_filename)
train_batch, train_label_batch = tf.train.shuffle_batch([train_img, train_label],
                                                batch_size = 100, num_threads=64,
                                                capacity=2000,                                                
                                                min_after_dequeue=1000)
x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])

train_model = alexnet.alexNet(x)
conv1_feature = train_model.conv1
"""
state = tf.Variable(0, name='counter')
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)
"""
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord,sess=sess)
    
    train_model.load_initial_weights(sess)
    
    #summary_op = tf.summary.merge_all()        
    #writer = tf.summary.FileWriter(dir,sess.graph)
    
    params=tf.trainable_variables()
    print("Trainable variables:------------------------")
    for idx, v in enumerate(params):
        print("  param {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))
    i = 0
    try:  
        while not coord.should_stop():      
            tra_images,tra_labels = sess.run([train_batch, train_label_batch])
            feature = sess.run(conv1_feature,feed_dict={x:tra_images})
            dic1 = {'feature':feature}
            sio.savemat(dir+str(i)+'_feature.mat',dic1) 
            dic2 = {'label':tra_labels}
            sio.savemat(dir+str(i)+'_label.mat',dic2) 
            i += 1
            """
            #show feature map by tensorboard
            split = tf.split(feature,num_or_size_splits=96,axis=3)
            featuremap = tf.summary.image('conv1_image',split[0],40)
            writer.add_summary(sess.run(featuremap))
            writer.close()        
            
            #show feature map by plt
            """
            conv1_reshape = sess.run(tf.reshape(feature[0,:,:,:], [96, 1, 23, 23]))
            fig1,ax1 = plt.subplots(nrows=1, ncols=1, figsize = (1,1))
            for i in range(1):
                plt.subplot(3, 4, i + 1)
                plt.imshow(conv1_reshape[i][0])
            plt.title('Conv1 featuremap')
            plt.show()
            
        
            #all_feature = sess.run(np.append(all_feature,feature,axis=0),feed_dict={:feature})
            #all_label = sess.run(np.append(all_label,tra_labels,axis=0))
    
    except tf.errors.OutOfRangeError:  
        print('Epochs Complete!')  
    finally:                
        coord.request_stop()  
    coord.join(threads)