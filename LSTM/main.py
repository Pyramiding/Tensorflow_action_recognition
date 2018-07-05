#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 19:26:17 2018

@author: wu
"""

import tensorflow as tf
import os
import tfrecord

tf.reset_default_graph() 

TIME_STEP = 100          
INPUT_SIZE = 300   
LR = 0.01               
epoch = 100000

train_filename = '/home/wu/TF_Project/action/sample_TFrecord/train1.tfrecords'
val_filename = '/home/wu/TF_Project/action/sample_TFrecord/val1.tfrecords'
logs_train_dir = '/home/wu/TF_Project/LSTM/logs_train_dir/'
logs_val_dir = '/home/wu/TF_Project/LSTM/logs_val_dir/'
model_dir = '/home/wu/TF_Project/LSTM/model_dir/'
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

x = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])       
y = tf.placeholder(tf.int32, [None, 6])                             
#%%
with tf.name_scope("LSTM"):
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=64)
    outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
        rnn_cell,               # cell you have chosen
        x,                      # input
        initial_state=None,     # the initial hidden state
        dtype=tf.float32,       # must given if set initial_state = None
        time_major=False,       # False: (batch, time step, input); True: (time step, batch, input)
    )
    output = tf.layers.dense(outputs[:, -1, :], 6, name="output")            

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

    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord,sess=sess)
    
    summary_op = tf.summary.merge_all()        
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)
    
    params=tf.trainable_variables()
    print("Trainable variables:------------------------")
    for idx, v in enumerate(params):
        print("  param {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))
    
    try:
        for step in range(epoch+1):
            if coord.should_stop():break
        
            if step % 500 == 0 and step != 0:
                tra_images,tra_labels = sess.run([train_batch, train_label_batch])
                _, loss_, accuracy_ = sess.run([train_op, loss,accuracy],feed_dict={x:tra_images, y:tra_labels})
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, loss_, accuracy_*100.0))    
                summary_str = sess.run(summary_op, feed_dict={x:tra_images, y:tra_labels})
                train_writer.add_summary(summary_str, step)
                
            if step % 1000 == 0 and step != 0:
                val_images, val_labels = sess.run([val_batch, val_label_batch])
                _, loss_, accuracy_ = sess.run([train_op, loss,accuracy],feed_dict={x:val_images, y:val_labels})
                print('Step %d, test loss = %.2f, test accuracy = %.2f%%' %(step, loss_, accuracy_*100.0))    
                summary_str = sess.run(summary_op, feed_dict={x:val_images, y:val_labels})
                val_writer.add_summary(summary_str, step)
                print('')
                
            if step % 10000 ==0 and step != 0:
                checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    
    except tf.errors.OutOfRangeError:
        print('Done reading')
    finally:
        coord.request_stop()  
        coord.join(threads)
#%%
