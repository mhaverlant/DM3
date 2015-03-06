__author__ = 'haverlantmatthias'

from sklearn import preprocessing
from sklearn.datasets import fetch_mldata
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import svm, grid_search, datasets
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import sklearn as sk
print sk.__version__
import scipy as sci
print sci.__version__

from time import time

import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import logging
from time import time


import mlp_ta_base as NN
reload(NN)

import sparse_ae as sae
reload(sae)


mnist=fetch_mldata('MNIST original')
Xmnist = mnist.data
Ymnist = mnist.target

Xmnist = Xmnist/255.0


print Ymnist.shape
print Xmnist.shape

ratio_train = 0.9
bound =  np.around(Xmnist.shape[0] * ratio_train,0)

X_train_brut = Xmnist[0:bound]
Y_train_brut = Ymnist[0:bound]
X_test_brut = Xmnist[bound:-1]
Y_test_brut = Ymnist[bound:-1]

X_train = X_train_brut
Y_train = Y_train_brut
X_test = X_test_brut
Y_test = Y_test_brut


# 1. Apprentissage d'un autoencodeur


a= sae.SparseAutoEncoder(nb_inputs = 784, Nb_hiddens=50, n_iter = 100, learning_rate = 0.01)
print a.learning_rate
print a.nb_inputs
#print a.compute_err_TS(X_train)

a.fit(X_train)

a.plot_features_appris()


# 2. Apprentissage d'un Pipeline = Feature Extraction par Autoencoder suivi d'un SVM

a = sae.SparseAutoEncoder(nb_inputs = 784)

estimators = [('FExtract', sae.SparseAutoEncoder(nb_inputs = 784)), ('Classifieur', svm.SVC())]

clf = Pipeline(estimators)

clf.fit(X_train, Y_train)


# 3. Gridsearch sur le Pipeline

a = sae.SparseAutoEncoder(nb_inputs = 784)

estimators = [('FExtract', sae.SparseAutoEncoder(nb_inputs = 784)), ('Classifieur', svm.SVC())]

clf = Pipeline(estimators)

params = { 'FExtract__Nb_hiddens':[100, 200], 'FExtract__beta':[0.1], 'FExtract__rho':[0.1, 0.15], 'Classifieur__kernel':['linear'] }

gs = grid_search.GridSearchCV(clf, params, verbose=1, n_jobs=4)

gs.fit(X_train, Y_train)

u=gs.best_estimator_.steps[0]

best_ae= u[1]

# Appel du MLP

mlp = NN.MLP([784, 100,  10],  fa='sigmoid', lr = 0.05, n_iter=10, wdecay=.000000)
mlp.reset()

mlp.fit(best_ae.transform(X_train), Y_train)

print mlp.score(best_ae.transform(X_train), Y_train)
print mlp.score(best_ae.transform(X_test), Y_test)





