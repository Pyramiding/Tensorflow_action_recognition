#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 20:37:16 2018

@author: wu
"""
import tensorflow as tf
import tfrecord
import alexnet

BATCH_SIZE=40
IMG_W=100
IMG_H=100
N_CLASSES=6
learning_rate = 0.0001 
epoch=1000
dropout=0.5

val_filename = '/home/wu/TF_Project/action/sample_TFrecord/val1.tfrecords'
model_dir = '/home/wu/TF_Project/action/model_tfrecord_sample/'

with tf.Graph().as_default():
    
    val_img, val_label = tfrecord.read_and_decode(val_filename)
    train_filename_queue = tf.train.string_input_producer([val_filename],num_epochs=None)
    val_batch, val_label_batch = tf.train.shuffle_batch([val_img, val_label],
                                                batch_size=40, capacity=2000,
                                                min_after_dequeue=1000)

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

    train_model = alexnet.alexNet(x, dropout, N_CLASSES)
    logits = train_model.fc3
    loss = alexnet.losses(logits, y_)  
    acc = alexnet.evaluation(logits, y_)
    train_op = alexnet.training(loss, learning_rate)
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)

        ckpt = tf.train.get_checkpoint_state(model_dir)
        #if ckpt and ckpt.model_checkpoint_path:
        #    saver.restore(sess, ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess,'/home/wu/TF_Project/action/model_tfrecord_sample/model.ckpt-1000')
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print("This is %s training global step(s) model" % global_step)
            try:
                for step in range(epoch+1):
                    if coord.should_stop():break
                    val_images,val_labels = sess.run([val_batch, val_label_batch])
                    _, val_loss, val_acc = sess.run([train_op, loss, acc],feed_dict={x:val_images, y_:val_labels})
                    if step % 50 ==0 :
                        print('Step %d, val loss = %.2f, val accuracy = %.2f%%' %(step, val_loss, val_acc*100.0))
            except tf.errors.OutOfRangeError:
                print('Done reading')
            finally:
                coord.request_stop()  
            coord.join(threads)



