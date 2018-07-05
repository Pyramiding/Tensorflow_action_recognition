#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 20:00:44 2018

@author: wu
"""

import tensorflow as tf
import numpy as np
import os
import glob
import math

ratio=0.2

# 获取文件路径和标签
def get_files(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    imgs_list=[]
    labels_list=[]
    for idx,dir1 in enumerate(cate):
        for dir2 in os.listdir(dir1):
            for im in glob.glob(dir1+'/'+dir2+'/*.jpg'):          
#               print('reading the images:%s'%(im))
#               img=io.imread(im)
#               img=transform.resize(img,(w,h))
                imgs_list.append(im)
                labels_list.append(idx)
    # 打乱文件顺序
    temp = np.array([imgs_list, labels_list])
    temp = temp.transpose()   # 转置
    np.random.shuffle(temp)
    all_image_list = list(temp[:,0])
    all_label_list = list(temp[:,1])
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample*ratio)) #测试样本数
    n_train = n_sample - n_val # 训练样本数
    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]
    
    return tra_images,tra_labels,val_images,val_labels

# 生成相同大小的批次
def get_batch(image, label, image_W, image_H, batch_size, capacity):

    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])
    image_contents = tf.read_file(input_queue[0])
    label = input_queue[1]
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # 统一图片大小-视频方法
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # 我的方法
    #image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)   # 标准化数据
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,   # 线程
                                              capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch