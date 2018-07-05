#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 20:42:22 2018

@author: wu
"""

import os
import numpy as np
import tensorflow as tf
import make_data
import alexnet

N_CLASSES = 6 
IMG_W = 100 
IMG_H = 100
BATCH_SIZE = 100
CAPACITY = 2000
MAX_STEP = 500
learning_rate = 0.0001 
dropout = 1.0
ratio = 0.7

path = '/home/wu/TF_Project/action/sample/'
model_dir = '/home/wu/TF_Project/action_list/model_sample/'
logs_train_dir = '/home/wu/TF_Project/action_list/logdir_train_sample/'
logs_val_dir = '/home/wu/TF_Project/action_list/logdir_val_sample/'
train_txt = 'train.txt'
val_txt = 'val.txt'

train_batch, train_label_batch, _ = make_data.get_batch(train_txt, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
val_batch, val_label_batch, _ = make_data.get_batch(val_txt, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
   
x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    
train_model = alexnet.alexNet(x, dropout, N_CLASSES)
logits = train_model.fc3
loss = alexnet.losses(logits, y_)  
acc = alexnet.evaluation(logits, y_)
train_op = alexnet.training(loss, learning_rate)
    
             
with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    summary_op = tf.summary.merge_all()        
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
            
            tra_images,tra_labels = sess.run([train_batch, train_label_batch])
            _, tra_loss, tra_acc = sess.run([train_op, loss, acc],feed_dict={x:tra_images, y_:tra_labels})
            
            if step % 20 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                summary_str = sess.run(summary_op, feed_dict={x:tra_images, y_:tra_labels})
                train_writer.add_summary(summary_str, step)
                
            if step % 50 == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = sess.run([val_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, acc], feed_dict={x:val_images, y_:val_labels})
                
                print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc*100.0))
                summary_str = sess.run(summary_op, feed_dict={x:val_images, y_:val_labels})
                val_writer.add_summary(summary_str, step)  
                              
            if step % 100 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()           
    coord.join(threads)