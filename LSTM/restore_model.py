#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 20:37:16 2018

@author: wu
"""
import tensorflow as tf
import tfrecord

TIME_STEP = 100          
INPUT_SIZE = 300  
N_CLASSES=6
learning_rate = 0.001 
epoch=10000

val_filename = '/home/wu/TF_Project/action/sample_TFrecord/val1.tfrecords'
model_dir = '/home/wu/TF_Project/LSTM/model_dir/'

tf.reset_default_graph() 
with tf.Graph().as_default():
    val_img, val_label = tfrecord.read_and_decode(val_filename)
    val_batch, val_label_batch = tf.train.shuffle_batch([val_img, val_label],
                                                batch_size=40, capacity=2000,
                                                min_after_dequeue=1000)

    x = tf.placeholder(tf.float32, shape=[None, TIME_STEP, INPUT_SIZE])
    y = tf.placeholder(tf.int32, shape=[None, 6])
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
    
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=output) 
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    accuracy = tf.metrics.accuracy(labels=tf.argmax(y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        ckpt = tf.train.get_checkpoint_state(model_dir)
        
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess,'/home/wu/TF_Project/LSTM/model_dir/model.ckpt-500000')
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print("This is %s training global step(s) model" % global_step)
            try:
                for step in range(epoch+1):
                    if coord.should_stop():break
                    val_images,val_labels = sess.run([val_batch, val_label_batch])
                    _, val_loss, val_acc = sess.run([train_op, loss, accuracy],feed_dict={x:val_images, y:val_labels})
                    if step % 500 ==0 :
                        print('Step %d, val loss = %.2f, val accuracy = %.2f%%' %(step, val_loss, val_acc*100.0))
            except tf.errors.OutOfRangeError:
                print('Done reading')
            finally:
                coord.request_stop()  
            coord.join(threads)



