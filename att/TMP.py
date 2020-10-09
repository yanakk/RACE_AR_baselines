#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 15:47:22 2020

@author: bme106
"""

import pickle
import numpy as np
from matplotlib import pyplot as plt

train_ini = pickle.load(open('ini.pickle', 'rb'))
ini_train_acc = train_ini['train_acc']
ini_train_label = train_ini['train_label']
ini_train_probs = train_ini['train_probs']

labels = np.array(ini_train_label)
probs = np.array(ini_train_probs)

#data_min = np.nanmin(probs, axis=1)[:, np.newaxis]
#data_max = np.nanmax(probs, axis=1)[:, np.newaxis]
#probs = ((probs-data_min)/(data_max-data_min)) + 0.1
#probs[np.isnan(probs)] = 0.1

preds = np.argmax(probs, axis=1)
acc = np.mean(labels==preds)
assert ini_train_acc==(acc*100)
plt.figure()
for ans in np.arange(4):
    labels_tmp = labels[labels==ans]
    probs_tmp = probs[labels==ans, :]
    plt.violinplot(probs_tmp, positions=np.arange(4)*1.5+ans*0.2, showmeans=True, widths=0.2)
    plt.pause(1)
plt.xticks(np.arange(0.3, 5, 1.5), ('prob-A', 'prob-B', 'prob-C', 'prob-D'))