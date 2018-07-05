#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 21:57:53 2018

@author: wu
"""

import numpy as np
import os
import tensorflow as tf
import make_data
import alexnet
import math
import scipy.io as sio 

N_CLASSES = 6
IMG_W = 100
IMG_H = 100
BATCH_SIZE = 108
CAPACITY = 2000
dropout = 1.0
train_txt = 'train.txt'
val_txt = 'val.txt'
model_dir = './model_sample/'

if not os.path.exists('./features'):
    os.mkdir('./features')
  
with tf.Graph().as_default():

    batch, label_batch, n = make_data.get_batch(val_txt, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)   
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])   
    train_model = alexnet.alexNet(x, dropout, N_CLASSES)
    feature_fc = train_model.fc2    
    saver = tf.train.Saver(tf.global_variables())
    
    with tf.Session() as sess:        
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
    
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        try:            
            num_iter = int(math.ceil(n / BATCH_SIZE))
            step = 0    
            while step < num_iter and not coord.should_stop():                
                images, labels = sess.run([batch, label_batch])    
                
                if step == 0 :
                    all_features = sess.run([feature_fc],feed_dict={x:images})
                    all_features = np.array(all_features)
                    all_features = np.squeeze(all_features)                    
                    all_labels = labels
                    pass
                
                feature_map = sess.run([feature_fc],feed_dict={x:images})
                feature_map = np.array(feature_map)
                feature_map = np.squeeze(feature_map)
                all_features = np.concatenate((all_features, feature_map),axis=0)
                all_labels = np.concatenate((all_labels, labels),axis=0)

                step = step+1
            
            dic_feature = {'features': all_features}
            sio.savemat('./features/'+'val_features.mat', dic_feature)
            dic_label = {'labels': all_labels}
            sio.savemat('./features/'+'val_labels.mat', dic_label)
            print('\nSuccessfully generated val matfile\n')
                    
        except Exception as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
        coord.join(threads)
