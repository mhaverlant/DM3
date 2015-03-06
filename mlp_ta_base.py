#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Code de Multi-layer perceptron
#
# Copyright (C) 2015 Thierry Artieres
# Laboratoire d'Informatique Fondamentale (LIF), Universite d'Aix-Marseille
# Ecole Centrale Marseille
# -----------------------------------------------------------------------------
# MLP with retropropagation learning.
# Includes:
# - definition with various activation functions (sigmoid, tanh, rectified linear units 
# - a strategy for adapting the learning rate online
# - L2 regularization
# -----------------------------------------------------------------------------

import numpy as np
import copy
from sklearn.metrics import accuracy_score

def sigmo(x):
    return 1.0/ (1+np.exp(- x))

def sigmoid(x):
    ''' Sigmoid like function using tanh '''
    return 1.5* sigmo(x) -0.25

def dsigmoid(x):
    ''' Derivative of sigmoid above '''
    return 1.5 * sigmo(x) * (1-sigmo(x))

def tanh(x):
    ''' Sigmoid like function using tanh '''
    return 1.7 * np.tanh(0.6 * x)

def dtanh(x):
    ''' Derivative of sigmoid above '''
    return 1.7*0.6*  (1.0-tanh(0.6* x)**2)

def lin(x):
    return x

def dlin(x):
    return 1

def RELU(x):
    y=x
    y[x<0]=0
    return y

def dRELU(x):
    y=x
    y[x<0] = 0
    y[x>0] =1 
    return y
    
activation_functions = {
    'sigmoid':(sigmoid,dsigmoid),
    'tanh':(tanh,dtanh),
    'relu':(RELU,dRELU),
    'lin':(lin, dlin)
    }


class Layer:
    '''Layer'''
    
    def __init__(self, ninputs, noutputs, fa= 'sigmoid', lr =0.01, wdecay=0.0 , loss = 'none'):

        self.ninputs = ninputs
        self.noutputs = noutputs
        self.lr = lr
        self.wdecay = wdecay
        self.loss = loss
        self.W = np.zeros((self.noutputs, self.ninputs))
        self.WBias = np.zeros((self.noutputs))
        self.dW = np.zeros((self.noutputs, self.ninputs))
        self.dWBias = np.zeros((self.noutputs))

        self.fa = fa
        self.fa, self.dfa = activation_functions[self.fa]

        self.activOut = np.zeros((self.noutputs,1))
        self.statesOut = np.zeros((self.noutputs,1)) 
        self.statesIn = np.zeros((self.ninputs,1)) 
        self.deltas = np.zeros((self.noutputs,1)) 

    def set_fa(self, fa):
        self.fa, self.dfa = activation_functions[fa]

    def forward(self, inputs):
        self.statesIn = inputs
        self.activOut = np.dot(self.W, inputs) + self.WBias
        self.statesOut = self.fa(self.activOut)
        return self.statesOut

    def backward(self, err0):
        ''' ddd '''
        self.delta = err0 * self.dfa(self.activOut)
        err = np.dot(self.W.T, self.delta)  
        return err
    
    def compute_gradient_step(self):
        ''' ddd '''
        self.dW =   self.lr * (np.dot(self.delta.reshape(self.delta.shape[0],1), self.statesIn.reshape(1,self.statesIn.shape[0] )) + 2* self.wdecay * self.W)
        self.dWBias =  self.lr * self.delta 

    def gradient_step(self):
        ''' ddd '''
        self.W =  self.W -   self.dW 
        self.WBias =   self.WBias - self.dWBias

    def reset(self):
        ''' Reset weights '''
        if (self.fa==RELU):
            self.W  = np.abs(np.random.normal(0, 1/(np.power(self.ninputs,0.5)*np.power(self.noutputs,0.5)),  (self.noutputs, self.ninputs)))
            self.dW  = np.abs(np.zeros( (self.noutputs, self.ninputs)))
        else:
            self.WBias  = np.random.normal(0,  1/(np.power(self.ninputs,0.5)*np.power(self.noutputs,0.5)),  (self.noutputs))
            self.dWBias  = np.zeros((self.noutputs))
        self.statesIn = self.statesIn *0.0
        self.statesOut = self.statesOut *0.0
        self.activOut = self.activOut *0.0

class MLP:
    ''' Multi-layer perceptron class. '''

    def __init__(self, SpecLayers, fa= 'sigmoid', lr= 0.01, wdecay=0.0 , n_iter = 10):
        ''' Initialization of the perceptron with given sizes.  '''
        self.SpecLayers = SpecLayers
        self.lr = lr
        self.lr_init = lr
        self.wdecay = wdecay
        self.n_iter= n_iter
        n=len(SpecLayers)
        self.fa=fa
        self.layers = []
        for i in range(1,n):
            self.layers.append(Layer(SpecLayers[i-1],SpecLayers[i],  fa= self.fa, lr = self.lr, wdecay = self.wdecay)) 
        self.reset()
           

    def reset(self):
        ''' Reset weights '''
        for i in range(len(self.layers)):
            self.layers[i].reset()
            
    def get_params(self, deep=True):
        return {"SpecLayers": self.SpecLayers, "lr": self.lr_init,  "wdecay": self.wdecay,  "n_iter": self.n_iter, "fa": self.fa}
    
    def set_lr(self, lr): 
        for l in self.layers:
            l.lr = lr

    def set_wdecay(self, wdecay):
        for l in self.layers:
            l.wdecay = wdecay

    def set_fa(self, fa):
        for l in self.layers:
            l.set_fa(fa)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.lr_init = self.lr
        n=len(self.SpecLayers)
        #print n
        self.layers = []      
        for i in range(1,n):
            self.layers.append(Layer(self.SpecLayers[i-1],self.SpecLayers[i], fa = self.fa, wdecay = self.wdecay)) 
        self.set_lr(self.lr)
        self.set_wdecay(self.wdecay)
        self.set_fa(self.fa)
        self.reset()

    def propagate_forward(self, data):
        ''' Propagate data from input layer to output layer. '''
        S = self.layers[0].forward(data)
        for i in range(1,len(self.layers)):
            S = self.layers[i].forward(S)
        return S

    def propagate_backward(self, Desired):
        ''' Propagate backward the error '''
        Prediction = self.layers[-1].statesOut
        Err =  Prediction - Desired
        for i in range(0,len(self.layers)):
            Err = self.layers[len(self.layers)-i-1].backward(Err)

    def MAJ_Weights(self):
        for i in range(0,len(self.layers)):
            self.layers[len(self.layers)-i-1].compute_gradient_step()
            self.layers[len(self.layers)-i-1].gradient_step()

    def compute_loss(self, predit, desired):
        return np.linalg.norm(predit-desired, 2) / len(predit)
        
    def gradient_step(self, InputData, DesiredOutput):
        S = self.propagate_forward(InputData)
        err = self.compute_loss(S, DesiredOutput)
        self.propagate_backward(DesiredOutput)
        self.MAJ_Weights()
        return err

    def fit_standard(self, X, Y):
        err_old = 0
        n_samples = X.shape[0]
        verbose = True
        err_old = 1000000000.0
        for iteration in range(self.n_iter):
            err = 0.0
            err2 =0.0
            for i  in np.arange(n_samples):
                err_sample = self.gradient_step(X[i], Y[i])
                if verbose:
                    err += err_sample
                    err2 += err_sample
            if verbose:
                err2 /= n_samples
                print("Iteration %d, Average quadratic error %.8f, learning rate=%.5f"
                      % (iteration, err2,  self.lr))
            self.lr = self.lr_init / (1+ (self.lr_init * self.wdecay * iteration * n_samples))
            self.set_lr(self.lr)
            err_old = err2
        return self

    def fit_onepass(self,Xbatch, Ybatch):
        err_batch = 0
        taille_batch = Xbatch.shape[0]
        for i  in np.arange(taille_batch):
            err_sample = self.gradient_step(Xbatch[i], Ybatch[i])
            err_batch += err_sample
        return err_batch

    def fit_auto_pas(self, X, Y, periode_batch=2):
        ### Toutes les periode_batch iterations on utilise un sous ensemble de taille_batch exemples pour determiner le pas de gradient optimal.
        err_old = 0
        verbose = True 
        err_old = 1000000000.0
        n_samples = X.shape[0]
        iperm = np.arange(X.shape[0])
        np.random.shuffle(iperm)
        periode_test = 2

        taille_batch = np.around(X.shape[0] /10.0,0)
        for iteration in range(self.n_iter):
            lr_sauv = self.lr
            if (iteration % periode_batch ==0):
                debut=np.random.random_integers(n_samples-taille_batch)
                Xbatch  = X[iperm[debut:debut+taille_batch]]
                Ybatch = Y[iperm[debut:debut+taille_batch]]
                lf = np.array([0.1, 0.5, 0.9, 0.95, 1.0, 1.05, 1.1, 2.0, 5.0, 10.0])
                lscores = np.zeros((len(lf),1))
                j=0
                for f in lf:
                    mlp2 = copy.deepcopy(self)
                    mlp2.lr = lr_sauv *f    
                    mlp2.set_lr(mlp2.lr)
                    err_batch = mlp2.fit_onepass(Xbatch, Ybatch)
                    lscores[j]=err_batch
                    j +=1
                ifmin = np.argmin(lscores)
                self.lr = lf[ifmin] * lr_sauv
                self.set_lr(self.lr)
        
            err = 0.0
            err2 =0.0
            #err_sample = 0
            for i  in np.arange(n_samples):
                err_sample = self.gradient_step(X[i], Y[i])
                if verbose:
                    err += err_sample
                    err2 += err_sample
            if verbose:
                err2 /= n_samples
                print("Iteration %d, Average quadratic error %.8f, learning rate=%.5f"
                      % (iteration, err2,  self.lr))
                if (iteration % periode_test ==0) and (iteration >0):
                    print("Accuracy on the training set %.8f" % (self.score(X,Y)))
#            self.lr = self.lr_init / (1+ (self.lr_init * self.wdecay * iteration))
            if (np.abs(err2-err_old)/err_old <0.0001):
                break
            err_old = err2
 
        return self

    def fit(self, X, Y):
        return self.fit_auto_pas(X,Y)

    def predict_classe(self,x):
        S = self.propagate_forward(x)
        return np.argmax(S)

    def predict_TS(self, X):
        nsamples = X.shape[0]
        ypredit = np.zeros((nsamples,1))
        for i in np.arange(nsamples):
            ypredit[i] = self.predict_classe(X[i])
        return ypredit

    def score(self, X,Y):
        ypredit = self.predict_TS(X)
        yreal = np.argmax(Y, axis=1)
        return  accuracy_score(yreal, ypredit)

# -----------------------------------------------------------------------------
