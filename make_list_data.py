#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 20:55:18 2018

@author: wu
"""
import os
import numpy as np
import tensorflow as tf
path = '/home/wu/TF_Project/action/sample/'
ratio = 0.7


def getPerActionList(actionPath):
    actionList = []
    actionNames = os.listdir(actionPath)
    if len(actionNames) > 0:
        for action in actionNames:
            actionName = os.path.join(actionPath, action)
            x = os.listdir(actionName)
            if len(x) > 0:
                for xt in x:
                    fullfilename = os.path.join(actionName, xt)
                    actionList.append(fullfilename)
    return actionList


def getAllActionList(actionPath, actionNum):
    img_label_list = []
    class_num = 0
    for i in actionNum:
        imageList = getPerActionList(actionPath + i)
        for img in imageList:
            _img_label = img + ' ' + str(class_num)
            img_label_list.append(_img_label)
        class_num = class_num + 1
    np.random.shuffle(img_label_list)
    return img_label_list

def get_txt(file, image):
    txtFile = open(file, 'w')
    num = 0
    for i in image:
        t = i + '\n'
        txtFile.writelines(t)
        num = num + 1
    txtFile.close()
    return num


def generate_txt(data_dir, ratio):
    trainFile = './train.txt'
    testFile = './val.txt'
    
    action_class = os.listdir(data_dir)
    img_label = getAllActionList(data_dir, action_class)
    
    train_img_label = img_label[:int(len(img_label) * ratio)]
    test_img_label = img_label[int(len(img_label) * ratio):]
    train_num = get_txt(trainFile, train_img_label)
    test_num = get_txt(testFile, test_img_label)

    print('Successfully generated txt file')
    print('Trainging num: %d' % train_num)
    print('Testing num: %d' % test_num)

def get_batch(file, image_W, image_H, batch_size, capacity):
    img_label = []
    with open(file, 'r') as file_to_read:
        for line in file_to_read.readlines():
            line = line.strip('\n')
            tmp = line.split(" ")
            img_label.append(tmp)
        img_label = np.array(img_label)
        img = img_label[:, 0]
        label = img_label[:, -1]
        label = np.int32(label)

    num = len(img)
    image = tf.cast(img, tf.string)
    label = tf.cast(label, tf.int64)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])   
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    
    ######################################
    # data argumentation should go to here
    ######################################

    # image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.resize_images(image, [image_W, image_H], method=0)
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)
    
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch, num


if __name__ == '__main__':
    generate_txt(path, ratio)