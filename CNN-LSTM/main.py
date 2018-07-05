#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 15:39:18 2018

@author: wu
"""

import tensorflow as tf
import os
import tfrecord
import CNN_LSTM

tf.reset_default_graph() 

batch_size=40
img_w=100
img_h=100  
channel=3
LR = 0.001               
epoch = 200

train_filename = '/home/wu/TF_Project/action/sample_TFrecord/train1.tfrecords'
val_filename = '/home/wu/TF_Project/action/sample_TFrecord/val1.tfrecords'
logs_train_dir = '/home/wu/TF_Project/CNN_LSTM/logs_train_dir/'
logs_val_dir = '/home/wu/TF_Project/CNN_LSTM/logs_val_dir/'
model_dir = '/home/wu/TF_Project/CNN_LSTM/model_dir/'
#%%
train_img, train_label = tfrecord.read_and_decode(train_filename)
train_batch, train_label_batch = tf.train.shuffle_batch([train_img, train_label],
                                                batch_size=40, num_threads=64,
                                                capacity=2000,                                                
                                                min_after_dequeue=1000)
val_img, val_label = tfrecord.read_and_decode(val_filename)
val_batch, val_label_batch = tf.train.shuffle_batch([val_img, val_label],
                                                batch_size=40, num_threads=64,
                                                capacity=2000,
                                                min_after_dequeue=1000)

train_label_batch = tf.one_hot(train_label_batch,depth = 6)
val_label_batch = tf.one_hot(val_label_batch,depth = 6)

x = tf.placeholder(tf.float32, shape=[batch_size, img_w, img_h, channel])       
y = tf.placeholder(tf.int32, shape=[batch_size, 6])         
                    
#%%        
output = CNN_LSTM.buildmodel(x)

with tf.variable_scope("loss") as scope:
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=output) 
    tf.summary.scalar(scope.name + "loss", loss)  
        
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

with tf.variable_scope("accuracy") as scope:
    accuracy = tf.metrics.accuracy(labels=tf.argmax(y, axis=1), predictions=tf.argmax(output, axis=1),)[1]
    tf.summary.scalar(scope.name + "accuracy", accuracy) 

#%%    
with tf.Session() as sess:

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 
    sess.run(init_op)     
    th = 0.000
    CNN_LSTM.pruning(sess,th)
    
    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord,sess=sess)
    
    #summary_op = tf.summary.merge_all()        
    #train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    #val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)
    
    params=tf.trainable_variables()
    print("Trainable variables:------------------------")
    for idx, v in enumerate(params):
        print("  param {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))
    
    try:
        for step in range(epoch+1):
            if coord.should_stop():break
            
            tra_images,tra_labels = sess.run([train_batch, train_label_batch])
            _, loss_, accuracy_ = sess.run([train_op, loss,accuracy],feed_dict={x:tra_images, y:tra_labels})        
            if step % 50 == 0 and step != 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, loss_, accuracy_*100.0))    
                #summary_str = sess.run(summary_op, feed_dict={x:tra_images, y:tra_labels})
                #train_writer.add_summary(summary_str, step)
              
            if step % 100 == 0 and step != 0:
                val_images, val_labels = sess.run([val_batch, val_label_batch])
                _, loss_, accuracy_ = sess.run([train_op, loss,accuracy],feed_dict={x:val_images, y:val_labels})
                print('Step %d, test loss = %.2f, test accuracy = %.2f%%' %(step, loss_, accuracy_*100.0))    
                #summary_str = sess.run(summary_op, feed_dict={x:val_images, y:val_labels})
                #val_writer.add_summary(summary_str, step)
                #print('')
            '''  
            if step % 500 ==0 and step != 0:
                checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
            '''
    except tf.errors.OutOfRangeError:
        print('Done reading')
    finally:
        coord.request_stop()  
        coord.join(threads)
#%%