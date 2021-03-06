#!/usr/bin/python

############################################################################
##
## File:      nn.py
##
## Purpose:   Neural networks used in the cyber security app.
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Dec. 28, 2019
##
############################################################################

from keras.layers import Dense,Dropout,Input,Embedding,BatchNormalization,Activation
from keras.models import Sequential
from keras.utils  import to_categorical

from tensorflow   import set_random_seed
from math         import log,ceil

import numpy as np
import os

import constants as const

############################################################################
##
## Purpose:   Deep belief network support function
##
############################################################################
def layers(model=None,outp=1,shape=3,act=None,useact=False):
    if not (model == None or outp < 1 or shape < 1 or act == None):
        if not useact:
            # encode the input data using the rectified linear unit
            enc  = Dense(outp,input_shape=(shape,),activation=act)
            # add the input layer to the model
            model.add(enc)
        else:
            # encode the input data using the rectified linear unit
            enc  = Dense(outp,input_shape=(shape,),use_bias=False)
            # add the input layer to the model
            model.add(enc)
            # add batch normalization
            enc  = BatchNormalization()
            # add the input layer to the model
            model.add(enc)
            # add the activation
            enc  = Activation(act)
            # add the input layer to the model
            model.add(enc)
    return

############################################################################
##
## Purpose:   Deep belief network for classification or regression
##
############################################################################
def dbn(inputs=[]
       ,outputs=[]
       ,splits=2
       ,props=2
       ,clust=const.BVAL
       ,loss='categorical_crossentropy'
       ,optimizer='rmsprop'
       ,rbmact='softmax'
       ,dbnact=None
       ,dbnout=0
       ,epochs=10
       ,embed=True
       ,encs=None
       ,useact=False
       ,verbose=0):
    model= None
    if inputs.any() and outputs.any():
        # seed the random number generator for repeatable results
        np.random.seed(12345)
        # linear stack of layers in the neural network (NN)
        model= Sequential()
        # add dense layers which are just densely connected typical artificial NN (ANN) layers
        #
        # at the input layer, there is a tendency for all NNs to act as imaging Gabor filters
        # where there's an analysis of the content of the inputs to determine whether there is
        # any specific content in any direction of the multi-dimensional inputs ... i.e. a Gabor filter
        # is a feature selector
        #
        # at the onset, the input level and second level match
        # first sibling level matches the number of features
        # all are deemed important and we will not allow feature selection as a result
        # this follows the writings in "Auto-encoding a Knowledge Graph Using a Deep Belief Network"
        #
        # if all possible levels, then M  = len(inputs[0])
        #M    = min(len(inputs[0]),props)
        M    = len(inputs[0])
        S    = splits
        # inputs have M columns and any number of rows, while output has M columns and any number of rows
        #
        # encode the input data using the rectified linear unit
        layers(model,M,M,'relu',useact)
        # add other encodings that are being passed in the encs array
        if not (encs == None):
            if not (len(encs) == 0):
                for enc in encs:
                    model.add(enc)
        # if M > const.MAX_FEATURES, then we will embed the inputs in a lower dimensional space of dimension const.MAX_FEATURES
        #
        # embed the inputs into a lower dimensional space if M > min(const.MAX_FEATURES,props)
        if embed:
            p    = min(const.MAX_FEATURES,min(props,outputs.shape[len(outputs.shape)-1]))
            if M > p:
                layers(model,p,M,'selu',useact)
                M    = int(ceil(log(p,S)/2.0))
        # add the rest of the layers according to the writings
        #
        # output dimension (odim) is initially S (number of splits in first level of hierarchy)
        odim = S
        # the loop variable counts the number of restricted boltzmann machines (RBM)
        # that are defined by the 2 extra layers that are added to the model
        # note that we should always have M RBMs, one for each property of the data
        for J in range(0,M):
            # the dimensionality will be computed in the loop as what's specified in
            # the writings works well for binary splits and 2 properties, but the loop
            # can be generalized to make the computation a bit better
            #
            # S^((J-2)/2)*M^(J/2)     for sibling levels in the writings
            # S^((J-1)/2)*M^((J-1)/2) for binary  levels in the writings
            #
            # for the sibling layers, the input is the old output dimension
            if J == 0:
                dim  = odim
            else:
                dim  = S * odim
            # next layers using the scaled exponential linear unit (Gibbs distribution) are siblings
            layers(model,dim,odim,'selu',useact)
            # redefine the input and output dimensions for the binary layers
            odim = S * dim
            # next layers using the softmax are binary layers
            #
            # note that by using dense layers of ANNs we can use back propagation within each RBM and across
            # all RBMs to tune the weights in each layer so that the hidden layers have a structure
            # that results in the correct number of clusters in the output layer giving rise to
            # the hierarchy that is detailed in the writings and the slide deck ... without this tuning
            # through back propagation, there is no guarantee that the hidden layers' structure will match
            # what is expected
            #
            # also, Gibbs sampling is being performed in the hidden layers so that edges are open
            # and closed to give rise to the correct cluster structure in the hidden layers defined above
            # so that the tuning through back propagation leads to the equilibrium distribution that can be color
            # coded into distinct regions of connected clusters ... see the writings for an example
            if not (J == M - 1):
                #layers(model,odim,dim,'sigmoid',useact)
                layers(model,odim,dim,rbmact,useact)
            else:
                if not (dbnact == None or dbnout <= 0):
                    layers(model,odim,dim,'sigmoid',useact)
                else:
                    # add another layer to change the structure of the network if needed based on clusters
                    if not (clust <= 0):
                        layers(model,clust,dim,rbmact,useact)
                    else:
                        layers(model,odim ,dim,rbmact,useact)
        # add another layer for a different kind of model, such as a regression model
        if not (dbnact == None or dbnout <= 0):
            # preceding layers plus this layer now perform auto encoding
            layers(model,M,odim,'selu',useact)
            # requested model at the output layer of this RBM
            layers(model,dbnout,M,dbnact,useact)
        # optimize using the typical categorical cross entropy loss function with root mean square optimizer to find weights
        model.compile(loss=loss,optimizer=optimizer)
        # we will allow for 100 iterations through the training data set to find the best sets of weights for the layers
        model.fit(inputs,outputs,epochs=epochs,verbose=verbose)
    # return the model to the caller
    return model

############################################################################
##
## Purpose:   Define the outputs for a classification
##
############################################################################
def categoricals(rows=0,splits=2,props=2,clust=const.BVAL):
    ret  = []
    if not (rows < 0 or ((splits < 2 or props < 2) and clust < 0)):
        # we need values to turn into labels when training
        # one-hot encode the integer labels as its required for the softmax
        s    = splits
        p    = props
        if not (clust < 0):
            nc   = clust
        else:
            nc   = s**(2*p)
        ret  = [np.random.randint(1,nc) for i in range(0,rows)]
        ret  = to_categorical(ret,num_classes=nc)
    return ret

# *************** TESTING *****************

def nn_testing(M=500,N=2):
    # number of data points, properties and splits
    m    = M
    p    = N
    if p > const.MAX_FEATURES:
        p    = const.MAX_FEATURES
    #s    = p + 1
    s    = p
    # uniformly sample values between 0 and 1
    #ivals= np.random.sample(size=(500,3))
    ivals= np.random.sample(size=(m,p))
    ovals= categoricals(M,s,p)
    # generate the clustering model for using the test values for training
    model = dbn(ivals,ovals,splits=s,props=p)
    if not (model == None):
        # generate some test data for predicting using the model
        ovals= np.random.sample(size=(m/10,p))
        # encode and decode values
        pvals= model.predict(ovals)
        # look at the original values and the predicted values
        print(ovals)
        print(pvals)
    else:
        print("Model 1 is null.")
    ovals= np.random.sample(size=(m,1))
    # generate the regression model for using the test values for training
    #model = dbn(ivals
                #,ovals
                #,splits=s
                #,props=p
                #,loss='mean_squared_error'
                #,optimizer='sgd'
                #,rbmact='sigmoid'
                #,dbnact='linear'
                #,dbnout=1)
    model = dbn(ivals
               ,ovals
               ,splits=s
               ,props=p
               ,loss='mean_squared_error'
               ,optimizer='sgd'
               ,rbmact='tanh'
               ,dbnact='linear'
               ,dbnout=1)
    if not (model == None):
        # generate some test data for predicting using the model
        ovals= np.random.sample(size=(m/10,p))
        # encode and decode values
        pvals= model.predict(ovals)
        # look at the original values and the predicted values
        print(ovals)
        print(pvals)
    else:
        print("Model 2 is null.")
    # generate the clustering model for using the test values for training
    # testing models of dimensions > 3
    p     = 5
    ivals= np.random.sample(size=(m,p))
    # we need values to turn into labels when training
    # one-hot encode the integer labels as its required for the softmax
    ovals= categoricals(M,s,const.MAX_FEATURES)
    model = dbn(ivals,ovals,splits=s,props=p)
    if not (model == None):
        # generate some test data for predicting using the model
        ovals= np.random.sample(size=(m/10,p))
        # encode and decode values
        pvals= model.predict(ovals)
        # look at the original values and the predicted values
        print(ovals)
        print(pvals)
    else:
        print("Model 2 is null.")
