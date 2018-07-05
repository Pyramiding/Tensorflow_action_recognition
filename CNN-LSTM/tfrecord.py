#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 20:39:22 2018

@author: wu
"""
import tensorflow as tf

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [100, 100, 3])
    #img = tf.image.rgb_to_grayscale(img)
    #img = tf.reshape(img, [100, 100])
    #split = tf.split(img,num_or_size_splits=3,axis=2)
    #img = split[0]
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    img = tf.cast(img, tf.float32)
    label = tf.cast(label, tf.int32)
    return img, label


