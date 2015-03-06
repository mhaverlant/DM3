#!/usr/bin/env python
# -----------------------------------------------------------------------------
# --------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


from sklearn import preprocessing
from sklearn.datasets import fetch_mldata
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm, grid_search, datasets
from sklearn.pipeline import Pipeline

from time import time

import numpy as np
#import pylab as pl
import matplotlib.pyplot as pl
import logging

import mlp_ta_base as NN
reload(NN)

mnist=fetch_mldata('MNIST original')
Xmnist = mnist.data
Ymnist = mnist.target

Xmnist = Xmnist/255.0

######################################################
######### RECUPERATION DES DONNEES ###################
######################################################

iperm = np.arange(Xmnist.shape[0])
np.random.shuffle(iperm)

Xmnist = Xmnist[iperm]
Ymnist = Ymnist[iperm]

from sklearn.preprocessing import OneHotEncoder
encod = OneHotEncoder()
Y=Ymnist.reshape(Ymnist.shape[0],1)
encod.fit(Y)

ratio_train = 0.9
bound =  np.around(Xmnist.shape[0] * ratio_train,0)

X_train = Xmnist[0:bound]
Y_train = Y[0:bound]
X_test = Xmnist[bound:-1]
Y_test = Y[bound:-1]

Y_train = encod.transform(Y_train).toarray()
Y_test = encod.transform(Y_test).toarray()

####################################################################
######################### APP d'un MLP ############################# 
####################################################################



mlp = NN.MLP([784, 100,  10],  fa='sigmoid', lr = 0.05, n_iter=10, wdecay=.000000)
mlp.reset()

mlp.fit(X_train, Y_train)

print mlp.score(X_train, Y_train)
print mlp.score(X_test, Y_test)
