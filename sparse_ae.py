"""AutoEncoders
"""

# Main author:

import time

import numpy as np

import pylab as pl

import numpy
import math
import time
import scipy.io
import scipy.optimize
import matplotlib.pyplot

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
#from sklearn.externals.six.moves import xrange
from sklearn.utils import check_arrays
from sklearn.utils import check_random_state
from sklearn.utils import gen_even_slices
from sklearn.utils.extmath import safe_sparse_dot
#from sklearn.utils.extmath import logistic_sigmoid


class SparseAutoEncoder(BaseEstimator, TransformerMixin):
    """Autoencoder implementation
    
    Parameters
    ----------
    Nb_hiddens : int, optional
        Number of hidden units.

    learning_rate : float, optional
        The learning rate for weight updates. It is *highly* recommended
        to tune this hyper-parameter. Reasonable values are in the
        10**[0., -3.] range.

    n_iter : int, optional
        Number of iterations/sweeps over the training dataset to perform
        during training.

    verbose : bool, optional
        The verbosity level.
    """

    def __init__(self, nb_inputs= 20, Nb_hiddens=20, learning_rate=0.001, batch_size=10, n_iter=10, verbose=False,  
                 beta = 0.1, Lambda = 0.001, rho =0.05):
        self.nb_inputs = nb_inputs
        self.Nb_hiddens = Nb_hiddens
        self.learning_rate = learning_rate
        self.learning_rate_initial = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.Lambda= Lambda
        self.rho = rho
        self.beta = beta
        self.init_weights()
        
    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))
       
    def Fwd_2_Hid_1sample(self, x):
        self.hidden_activations  = np.dot(self.W_Hids,x) + self.WBias_hids
        self.hidden_states = self.sigmoid(self.hidden_activations)
        return self.hidden_states

    def Fwd_2_Out_1sample(self):
        self.output_states = np.dot(self.W_Outs,self.hidden_states.T) + self.WBias_outs
        return self.output_states

    def transform_1sample(self, x):
        return self.Fwd_2_Hid_1sample(x)
    
    def compute_err_1sample(self, x):
        self.hidden_states  = self.Fwd_2_Hid(x)
        self.output_states = self.Fwd_2_Out()
        v_tmp = self.output_states - x
        return np.dot(v_tmp, v_tmp.T) / x.shape[0]

    def BckProp_1sample(self,x):
        self.Err_activ_outputs = 2 * (self.output_states - x)
        self.Deltas_Outputs = self.Err_activ_outputs.reshape(self.Err_activ_outputs.shape[0],1) * self.hidden_states
        self.Err_state_hiddens = np.dot(self.W_Outs.T , self.Err_activ_outputs)
       # TA A CHANGER anciennement tanh.
        #self.Err_activ_hiddens = self.Err_state_hiddens * (1 - np.power(np.tanh(self.hidden_activations),2))
        self.Deltas_Hiddens = self.Err_activ_hiddens.reshape(self.Err_activ_hiddens.shape[0],1) * x

        self.Deltas_Bias_Outputs = self.Err_activ_outputs
        self.Deltas_Bias_Hiddens = self.Err_activ_hiddens

#        self.Deltas_Hiddens = np.dot(self.Err_hiddens, x)
        Deltas = self.Deltas_Outputs + self.Deltas_Hiddens.T
        self.W_Outs = self.W_Outs - self.learning_rate * Deltas
#        self.W_Hids = self.W_Hids - self.learning_rate * self.Deltas_Hiddens
#        self.W_Outs = self.W_Outs + self.W_Hids.T
        self.W_Hids = self.W_Outs.T
        self.WBias_outs = self.WBias_outs - self.learning_rate *  self.Deltas_Bias_Outputs
        self.WBias_hids = self.WBias_hids - self.learning_rate *  self.Deltas_Bias_Hiddens

    def fit_one_sample(self, x):
        """Inner fit for one mini-batch.
        """
        self.compute_err_1sample(x)
        self.BckProp_1sample(x)
        return self.compute_err_1sample(x)

#fonctions pour dataset entier (necessaires pour sparse autoencoders)

    def Fwd_2_Hid_TS(self, X): #TS stands for the whole Training Set
        # X is assumed matrix with one sample per column
        H = np.dot(self.W_Hids, X) + self.WBias_hids
        H = self.sigmoid(H)
        return H

    def Fwd_2_Out_TS(self, H):
        O = np.dot(self.W_Outs,H) + self.WBias_outs
        O = self.sigmoid(O)
        return O

    def predict_TS(self, X):
        H = self.Fwd_2_Hid_TS(X)
        O = self.Fwd_2_Out_TS(H)
        return self.sigmoid(O)
    
    def predict_MeanActivHid_TS(self, X):
        H = self.Fwd_2_Hid_TS(X)
        rho_cap = np.sum(H, axis = 1) / X.shape[1]
        return H, rho_cap

    def compute_err_TS(self, X):
        O = self.predict_TS(X)
        #print "O shape \n", O.shape
        #print X.shape
        diff = self.predict_TS(X) - X
        return 0.5 * np.sum(np.multiply(diff, diff)) / (X.shape[1] * X.shape[0])
    
    def compute_grad_TS(self, X):
        #print " Avant 1 \n"
        H, rho_cap = self.predict_MeanActivHid_TS(X)
        #print " Apres Inst1 \n"
        KL_div_grad = self.beta * (-(self.rho / rho_cap) + ((1 - self.rho) / (1 - rho_cap)))
        #print " Apres Inst2\n"
        O = self.Fwd_2_Out_TS(H)
        #print "O : ", O.shape
        diff = O - X
        #print "Diff : ", diff.shape
        del_out = np.multiply(diff, np.multiply(O, 1 - O)) # PB : pour la sigmoide uniqumenent ?
        #print "del_out : ", del_out.shape
        
        del_hid = np.multiply(np.dot(np.transpose(self.W_Outs), del_out) + np.transpose(np.matrix(KL_div_grad)), 
                              np.multiply(H, 1 - H))     
        #print "del_hid : ", del_hid.shape
        
        ### Compute the gradient
            
        W1_grad = np.dot(del_hid, np.transpose(X))
        W2_grad = np.dot(del_out, np.transpose(H))
        #print "delhid ", del_hid.shape
        b1_grad = np.sum(del_hid, axis = 1)
        #print "delout ", del_out.shape
        b2_grad = np.sum(del_out, axis = 1)
            
        W1_grad = W1_grad / X.shape[1] + self.Lambda * self.W_Hids
        W2_grad = W2_grad / X.shape[1] + self.Lambda * self.W_Outs
        b1_grad = b1_grad / X.shape[1]
        b2_grad = np.array(b2_grad)
        b2_grad = b2_grad / X.shape[1]
        
        """ Transform numpy matrices into arrays """
        
        W1_grad = np.array(W1_grad)
        W2_grad = np.array(W2_grad)
        b1_grad = np.array(b1_grad)
        #print "b1grad ", b1_grad.shape
        b2_grad = np.array(b2_grad)
        b2_grad.reshape(self.W_Hids.shape[1],1)
        
        #print "b2grad ", b2_grad.shape
        b2_grad = np.array(b2_grad)
        
        #print "b2grad ", b2_grad.shape
        b2_grad.reshape(self.W_Hids.shape[1],1)
        
        b2_grad =b2_grad[:,np.newaxis] # Foirueux jQuery203017504372132186585_1422273398105
        
 #       print "b2grad ", b2_grad.shape

        # theta_grad = np.concatenate((W1_grad.flatten(), W2_grad.flatten(), 
       #                              b1_grad.flatten(), b2_grad.flatten()))
        return W1_grad, W2_grad, b1_grad, b2_grad
       
    def maj(self, W1_grad, W2_grad, b1_grad, b2_grad):
        self.W_Hids = self.W_Hids - self.learning_rate * W1_grad  
        self.W_Outs = self.W_Outs - self.learning_rate * W2_grad  
        self.WBias_hids = self.WBias_hids - self.learning_rate * b1_grad  
        self.WBias_outs = self.WBias_outs - self.learning_rate * b2_grad  
 
    def maj_Adagrad(self, W1_grad, W2_grad, b1_grad, b2_grad):

        self.GW_Hids = self.GW_Hids +  (W1_grad *  W1_grad )
        self.GW_Outs = self.GW_Outs +  (W2_grad *  W2_grad )
        self.GWBias_hids = self.GWBias_hids +  (b1_grad *  b1_grad )
        self.GWBias_outs = self.GWBias_outs +  (b2_grad *  b2_grad )
        
        self.W_Hids = self.W_Hids - self.learning_rate * W1_grad  / np.sqrt(self.GW_Hids) 
        self.W_Outs = self.W_Outs - self.learning_rate  * W2_grad  / np.sqrt(self.GW_Outs)
        self.WBias_hids = self.WBias_hids - self.learning_rate * b1_grad   / np.sqrt(self.GWBias_hids) 
        self.WBias_outs = self.WBias_outs - self.learning_rate * b2_grad  / np.sqrt(self.GWBias_outs) 

       # self.GW_Hids = self.GW_Hids +  (W1_grad *  W1_grad )
       # self.GW_Outs = self.GW_Outs +  (W2_grad *  W2_grad )
       # self.GWBias_hids = self.GWBias_hids +  (b1_grad *  b1_grad )
       # self.GWBias_outs = self.GWBias_outs +  (b2_grad *  b2_grad )
       

    def init_weights(self):
        self.limit0 = 0
        self.limit1 = self.nb_inputs * self.Nb_hiddens
        self.limit2 = 2 * self.nb_inputs * self.Nb_hiddens 
        self.limit3 = 2 * self.nb_inputs * self.Nb_hiddens + self.Nb_hiddens
        self.limit4 = 2 *  self.nb_inputs * self.Nb_hiddens  +  self.nb_inputs + self.Nb_hiddens 
#        self.nb_inputs = nb_inputs
        self.W_Hids = np.asarray(np.random.normal(0, 1/(np.power(self.Nb_hiddens,0.5)*np.power(self.nb_inputs,0.5)), (self.Nb_hiddens, self.nb_inputs)), order='fortran')
        self.W_Outs = np.asarray(np.random.normal(0, 1/(np.power(self.Nb_hiddens,0.5)*np.power(self.nb_inputs,0.5)), (self.nb_inputs, self.Nb_hiddens)), order='fortran') #self.W_Hids.T
        self.WBias_hids = np.zeros((self.Nb_hiddens, 1))
        self.WBias_outs = np.zeros((self.nb_inputs, 1))

        #Initialisation de l'historique du gradient pour la version Adagrad
        self.GW_Hids = np.zeros((self.Nb_hiddens, self.nb_inputs)) + 100.0
        self.GW_Outs = np.zeros((self.nb_inputs, self.Nb_hiddens)) + 100.0
        self.GWBias_hids = np.zeros((self.Nb_hiddens, 1)) + 100.0
        self.GWBias_outs = np.zeros((self.nb_inputs, 1)) + 100.0

        self.theta = numpy.concatenate((self.W_Hids.flatten(), self.W_Outs.flatten(),
                                        self.WBias_hids.flatten(), self.WBias_outs.flatten()))

        return 1


    def sparseAutoencoderCost(self, theta, X):

        self.W_Hids = theta[self.limit0 : self.limit1].reshape(self.Nb_hiddens, self.nb_inputs)
        self.W_Outs = theta[self.limit1 : self.limit2].reshape(self.nb_inputs, self.Nb_hiddens)
        self.WBias_hids = theta[self.limit2 : self.limit3].reshape(self.Nb_hiddens, 1)
        self.WBias_outs = theta[self.limit3 : self.limit4].reshape(self.nb_inputs, 1)
        #print " Avant 1 \n"
        H, rho_cap = self.predict_MeanActivHid_TS(X)
        #print " Apres Inst1 \n"
        KL_div_grad = self.beta * (-(self.rho / rho_cap) + ((1 - self.rho) / (1 - rho_cap)))
        #print " Apres Inst2\n"
        O = self.Fwd_2_Out_TS(H)
        #print "O : ", O.shape
        diff = O - X
        #print "Diff : ", diff.shape


      
        sum_of_squares_error = 0.5 * numpy.sum(numpy.multiply(diff, diff)) / X.shape[1]
        weight_decay         = 0.5 * self.Lambda * (numpy.sum(numpy.multiply(self.W_Hids, self.W_Hids)) +
                                                   numpy.sum(numpy.multiply(self.W_Outs, self.W_Outs)))
        KL_divergence        = self.beta * numpy.sum(self.rho * numpy.log(self.rho / rho_cap) +
                                                     (1 - self.rho) * numpy.log((1 - self.rho) / (1 - rho_cap)))
        
        cost                 = sum_of_squares_error + weight_decay + KL_divergence
        
        del_out = np.multiply(diff, np.multiply(O, 1 - O)) # PB : pour la sigmoide uniqumenent ?
        #print "del_out : ", del_out.shape
        
        del_hid = np.multiply(np.dot(np.transpose(self.W_Outs), del_out) + np.transpose(np.matrix(KL_div_grad)), 
                              np.multiply(H, 1 - H))     
        #print "del_hid : ", del_hid.shape
        
        ### Compute the gradient
            
        W1_grad = np.dot(del_hid, np.transpose(X))
        W2_grad = np.dot(del_out, np.transpose(H))
        #print "delhid ", del_hid.shape
        b1_grad = np.sum(del_hid, axis = 1)
        #print "delout ", del_out.shape
        b2_grad = np.sum(del_out, axis = 1)
            
        W1_grad = W1_grad / X.shape[1] + self.Lambda * self.W_Hids
        W2_grad = W2_grad / X.shape[1] + self.Lambda * self.W_Outs
        b1_grad = b1_grad / X.shape[1]
        b2_grad = np.array(b2_grad)
        b2_grad = b2_grad / X.shape[1]
        
        """ Transform numpy matrices into arrays """
        
        W1_grad = np.array(W1_grad)
        W2_grad = np.array(W2_grad)
        b1_grad = np.array(b1_grad)
        #print "b1grad ", b1_grad.shape
        b2_grad = np.array(b2_grad)
        b2_grad.reshape(self.W_Hids.shape[1],1)
        
        #print "b2grad ", b2_grad.shape
        b2_grad = np.array(b2_grad)
        
        #print "b2grad ", b2_grad.shape
        b2_grad.reshape(self.W_Hids.shape[1],1)
        b2_grad =b2_grad[:,np.newaxis] # Foirueux jQuery203017504372132186585_1422273398105        
      
        """ Unroll the gradient values and return as 'theta' gradient """
        
        theta_grad = numpy.concatenate((W1_grad.flatten(), W2_grad.flatten(),
                                        b1_grad.flatten(), b2_grad.flatten()))
        
        return [cost, theta_grad]

    def fit_SGD(self, X, y=None):
        X = np.transpose(X)
        n_samples = X.shape[0]
        i_rand = np.random.permutation(n_samples)
        X = X[i_rand,]
        verbose = self.verbose
        
        err_old = 1000000000
        for iteration in range(self.n_iter):
            
#            print "Iteration ", iteration
            if verbose:
                begin = time.time()
            
            W1_grad, W2_grad, b1_grad, b2_grad = self.compute_grad_TS(X)
            self.maj(W1_grad, W2_grad, b1_grad, b2_grad)
            err = self.compute_err_TS(X)
            self.learning_rate = self.learning_rate_initial / (1+ (self.learning_rate_initial * self.Lambda * iteration))
#            print " : Iteration -> Erreur %s \n" %  err
#            print " Espilon ", self.learning_rate

        return self
   

    def fit_Adagrad(self, X, y=None):
        X = np.transpose(X)
        n_samples = X.shape[0]
        i_rand = np.random.permutation(n_samples)
        X = X[i_rand,]
        verbose = self.verbose
        
        err_old = 1000000000
        for iteration in range(self.n_iter):
            
            print "Iteration ", iteration
            if verbose:
                begin = time.time()
            
            W1_grad, W2_grad, b1_grad, b2_grad = self.compute_grad_TS(X)
            self.maj_Adagrad(W1_grad, W2_grad, b1_grad, b2_grad)
            err = self.compute_err_TS(X)
            #self.learning_rate = self.learning_rate_initial / (1+ (self.learning_rate_initial * self.Lambda * iteration))
            print " : Iteration -> Erreur %s \n" %  err
#            print " Espilon ", self.learning_rate

        return self     
        
    def fit_BFGS(self, X ,  y=None):

#        print "Avant transpose " 
#        print X.shape
        X2 = np.transpose(X)
#        print "Apres transpose " 
#        print X.shape
        max_iterations = 50

        opt_solution  = scipy.optimize.minimize(self.sparseAutoencoderCost, self.theta, 
                                                args = (X2,), method = 'L-BFGS-B', 
                                                jac = True, options = {'maxiter': max_iterations})
        opt_theta     = opt_solution.x
        self.theta = opt_theta
        opt_W1 = opt_theta[self.limit0 : self.limit1].reshape(self.Nb_hiddens, self.nb_inputs)
        self.W_Hids = self.theta[self.limit0 : self.limit1].reshape(self.Nb_hiddens, self.nb_inputs)
        self.W_Outs = self.theta[self.limit1 : self.limit2].reshape(self.nb_inputs, self.Nb_hiddens)
        self.WBias_hids = self.theta[self.limit2 : self.limit3].reshape(self.Nb_hiddens, 1)
        self.WBias_outs = self.theta[self.limit3 : self.limit4].reshape(self.nb_inputs, 1)
        return self
       
    def fit(self, X ,  y=None):
        return self.fit_BFGS(X)


    def fit_transform(self, X ,  y=None):
#        X = np.transpose(X)
        max_iterations = 50

        self.fit(X)
#        opt_solution  = scipy.optimize.minimize(self.sparseAutoencoderCost, self.theta, 
#                                                args = (X,), method = 'L-BFGS-B', 
#                                                jac = True, options = {'maxiter': max_iterations})
#        opt_theta     = opt_solution.x
#        self.theta = opt_theta
#        opt_W1 = opt_theta[self.limit0 : self.limit1].reshape(self.Nb_hiddens, self.nb_inputs)
#        self.W_Hids = self.theta[self.limit0 : self.limit1].reshape(self.Nb_hiddens, self.nb_inputs)
#        self.W_Outs = self.theta[self.limit1 : self.limit2].reshape(self.nb_inputs, self.Nb_hiddens)
#        self.WBias_hids = self.theta[self.limit2 : self.limit3].reshape(self.Nb_hiddens, 1)
#        self.WBias_outs = self.theta[self.limit3 : self.limit4].reshape(self.nb_inputs, 1)
        X2 = np.transpose(X)
        return np.transpose(self.Fwd_2_Hid_TS(X2))
        
    def transform(self, X, y=None):
        X = np.transpose(X)
        return np.transpose(self.Fwd_2_Hid_TS(X))

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.init_weights()

    def score(self, X):
        X2 = np.transpose(X)
        return np.mean((X2-self.predict_TS(X2))**2)


    def plot_features_appris(self):
        nb_hiddens = self.Nb_hiddens
        g = self.W_Hids
        taille_filtre = np.rint(np.power(g.shape[1],0.5))
        nblig = np.floor(np.power(nb_hiddens,0.5))
        nbcol = np.floor(np.power(nb_hiddens,0.5))+1
        
        for i in range(nb_hiddens):
            pl.subplot(nblig, nbcol, i + 1)
            pl.axis('off')
            filtre = g[i,].reshape(taille_filtre, taille_filtre)   / (np.sum(g[i,] * g[i,] )) 
            pl.imshow(filtre, cmap = pl.get_cmap('gray'), vmin = -1, vmax = 1)
        pl.show()
            
        return;

#    def Feat_Extract_AutoEncoder(self, quoi d'autre ?):


