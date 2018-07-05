#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 21:18:24 2018

@author: wu
"""

# TEST
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import input_data

BATCH_SIZE = 40
CAPACITY = 200
IMG_W = 100
IMG_H = 100

train_dir = '/home/wu/TF_Project/action/sample/'
image_list,label_list,val_image_list,val_label_list = input_data.get_files(train_dir)
image_batch, label_batch = input_data.get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop() and i < 1:
            img, label = sess.run([image_batch, label_batch])

            for j in np.arange(BATCH_SIZE):
                print("label: %d" % label[j])
                plt.imshow(img[j, :, :, :])
                plt.show()
            i += 1
    except tf.errors.OutOfRangeError:
        print("done!")
    finally:
        coord.request_stop()
    coord.join(threads)