#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 15:27:21 2018

@author: wu
"""
import tensorflow as tf
import os
import model
import tfrecord
import alexnet

BATCH_SIZE=40
IMG_W=100
IMG_H=100
N_CLASSES=6
learning_rate = 0.0001 
epoch=1000
dropout=1.0
all_acc = []

train_filename = '/home/wu/TF_Project/action/sample_TFrecord/train1.tfrecords'
val_filename = '/home/wu/TF_Project/action/sample_TFrecord/val1.tfrecords'
logs_train_dir = '/home/wu/TF_Project/action/logdir_train_tfrecord_sample/'
logs_val_dir = '/home/wu/TF_Project/action/logdir_val_tfrecord_sample/'
model_dir = '/home/wu/TF_Project/action/model_tfrecord_sample/'

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

x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

#based on alexnet
train_model = alexnet.alexNet(x, dropout, N_CLASSES)
logits = train_model.fc3
loss = alexnet.losses(logits, y_)  
acc = alexnet.evaluation(logits, y_)
train_op = alexnet.training(loss, learning_rate)

#based on simple net
#logits = model.inference(x, BATCH_SIZE, N_CLASSES)
#loss = model.losses(logits, y_)  
#acc = model.evaluation(logits, y_)
#train_op = model.training(loss, learning_rate)

with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
        
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord,sess=sess)
    
    summary_op = tf.summary.merge_all()        
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)
    
    #train_model.load_initial_weights(sess)
    
    params=tf.trainable_variables()
    print("Trainable variables:------------------------")
    for idx, v in enumerate(params):
        print("  param {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))
    
    try:
        for step in range(epoch+1):
            if coord.should_stop():break
        
            tra_images,tra_labels = sess.run([train_batch, train_label_batch])
            _, tra_loss, tra_acc = sess.run([train_op, loss, acc],feed_dict={x:tra_images, y_:tra_labels})
            if step % 20 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))    
                summary_str = sess.run(summary_op, feed_dict={x:tra_images, y_:tra_labels})
                train_writer.add_summary(summary_str, step)
                
            if step % 50 == 0 :
                val_images, val_labels = sess.run([val_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, acc], feed_dict={x:val_images, y_:val_labels})
                print('')
                print('Step %d, val loss = %.2f, val accuracy = %.2f%%' %(step, val_loss, val_acc*100.0))
                summary_str = sess.run(summary_op, feed_dict={x:val_images, y_:val_labels})
                val_writer.add_summary(summary_str, step)  
            all_acc = all_acc+[val_acc]
            if step % 1000 ==0 and step != 0:
                checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print('Done reading')
    finally:
        coord.request_stop()  
    coord.join(threads)
    """
    W=sess.run(params[26])
    b=sess.run(params[27])
    fea=sess.run(dense2,feed_dict={x:img})
    """