#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 20:39:22 2018

@author: wu
"""
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt


path = '/home/wu/TF_Project/action/sample/'
ratio = 0.2

def get_files(file_dir, ratio):
    class_train = []
    label_train = []
    k = 0
    for train_class in os.listdir(file_dir):
        for sub_train in os.listdir(file_dir+train_class):
            for image in os.listdir(file_dir+train_class+'/'+sub_train) :
                class_train.append(file_dir+train_class+'/'+sub_train+'/'+image)
                label_train.append(k)
        k+=1
    temp = np.array([class_train,label_train])
    temp = temp.transpose()
    #shuffle the samples
    np.random.shuffle(temp)
    #after transpose, images is in dimension 0 and label in dimension 1
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
    
#制作二进制数据
def create_record():
    writer_train = tf.python_io.TFRecordWriter('/home/wu/TF_Project/action/sample_TFrecord/train1.tfrecords')
    writer_val = tf.python_io.TFRecordWriter('/home/wu/TF_Project/action/sample_TFrecord/val1.tfrecords')
    tra_images,tra_labels,val_images,val_labels = get_files(path, ratio)
    
    for index, name in enumerate(tra_images):
        img = Image.open(name)
        img = img.resize((100, 100))
        img_raw = img.tobytes()
        example_train = tf.train.Example(
           features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[tra_labels[index]])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                 }))
        writer_train.write(example_train.SerializeToString())
    for index, name in enumerate(val_images):
        img = Image.open(name)
        img = img.resize((100, 100))
        img_raw = img.tobytes()
        example_val = tf.train.Example(
           features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[val_labels[index]])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                 }))
        writer_val.write(example_val.SerializeToString())

    writer_train.close()
    writer_val.close()

#data = create_record()

def read_and_decode(filename):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
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
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    img = tf.cast(img, tf.float32)
    label = tf.cast(label, tf.int32)
    return img, label

if __name__ == '__main__':
    if 1:
        data = create_record()
    else:
        train_filename = '/home/wu/TF_Project/action/sample_TFrecord/train1.tfrecords'
        img, label = read_and_decode(train_filename)
        #print(img,label)
        filename_queue = tf.train.string_input_producer([train_filename],num_epochs=None)
        img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=40, capacity=2000,
                                                    min_after_dequeue=1000)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord,sess=sess)
            try:
                for i in range(1):
                    example, l = sess.run([img_batch, label_batch])
                    plt.imshow(example[0,:,:,:])
                    #print(example, l) 
            except tf.errors.OutOfRangeError:
                print('Done reading')
            finally:
                coord.request_stop()  
    
            coord.request_stop()
            coord.join(threads)
