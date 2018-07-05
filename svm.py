#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 22:12:22 2018

@author: wu
"""

import scipy.io as sio
import numpy as np
from  sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler 
from datetime import datetime

start_time = datetime.now()

train_features = sio.loadmat('./features/train_features.mat')
train_labels = sio.loadmat('./features/train_labels.mat')
val_features = sio.loadmat('./features/val_features.mat')
val_labels = sio.loadmat('./features/val_labels.mat')

train_feature = np.array(train_features['features'])
train_label = np.array(train_labels['labels']).ravel()

val_feature = np.array(val_features['features'])
val_label = np.array(val_labels['labels']).ravel()

#train_feature = StandardScaler.fit_transform(train_feature)
#val_feature = StandardScaler.transform(val_feature)  
scaler = StandardScaler().fit(train_feature)
train_feature = scaler.transform(train_feature)
val_feature = scaler.transform(val_feature)  

clf = svm.SVC()
clf.fit(train_feature, train_label)

val_predict = clf.predict(val_feature)
accuarcy = metrics.accuracy_score(val_predict, val_label)
# accuarcy =clf.score(x_test,y_test)
print('Accuarcy: %.2f%%' % (accuarcy*100.0))

end_time = datetime.now()
print((end_time-start_time))