#!/usr/bin/python

############################################################################
##
## File:      autoencoder.py
##
## Purpose:   Example of autoencoder usage for feature selection
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Jul. 22, 2019
##
############################################################################

from keras.layers import Dense, Dropout, Input
from keras.models import Sequential
import numpy as np
import tensorflow as tf
import os

def cmodel(inputs):
    # linear stack of layers in the neural network (NN)
    model= Sequential()
    # add dense layers which are just densely connected typical artificial NN (ANN) layers
    #
    # at the input layer, there is a tendency for all NNs to act as imaging Gabor filters
    # where there's an analysis of the content of the inputs to determine whether there is
    # any specific content in any direction of the multi-dimensional inputs ... i.e. a Gabor filter
    # is a feature selector
    #
    # we will specify that the output dimensionality be equal to the dimension of the input space
    dim  = len(inputs[0])
    odim = dim
    # inputs have dim columns and any number of rows, while output has odim columns and any number of rows
    #
    # encode the input data
    enc  = Dense(odim,input_shape=(dim,),activation='relu')
    # if you want to look at the output of this layer, including the weights
    #
    # get symbolic tensor made from the original inputs
    outx = Input(shape=(dim,),name='outx')
    # get the output of this layer
    outy = enc(outx)
    # obtain the weights from this layer
    wts  = enc.get_weights()
    # add the encoding layer to the model
    model.add(enc)
    # dropout layer makes the model more generalizable to other input data sets besides the training set
    #
    # we will drop 10% of the connections at random
    model.add(Dropout(0.1))
    # decode the encoded input data
    dec  = Dense(odim,activation='sigmoid')
    model.add(dec)
    # optimize using the typical mean squared error loss function with stochastic gradient descent to find weights
    model.compile(loss='mean_squared_error',optimizer='sgd')
    # fit the inputs to the outputs, as an autoencoder should do, which is give you back the
    # same inputs that you started with, or a reasonable facsimile from the same distribution
    #
    # we will allow for 100 iterations through the training data set to find the best sets of weights for the layers
    model.fit(inputs,inputs,epochs=100,verbose=1)
    # return the model to the caller
    return [model,wts]

# uniformly sample 100 values between 0 and 1
ivals= np.random.sample(100)
# turn the values into a matrix of dimension 10x10
ivals.resize(10,10)
# generate the model for autoencoding using the test values for training
[model,wts] = cmodel(ivals)
# generate some test data for predicting using the model
ovals= np.random.sample(10)
ovals.resize(1,10)
# see if we get back something close to the original data that we sent in
# or at least something close from the same distribution
#
# encode and decode values
pvals= model.predict(ovals)
# look at the original values and the predicted values from the autoencoder
print(ovals)
print(pvals)
# look at the weights from the rectified linear unit (relu)
#
# note that this layer is an affine transformation of a flattened approximate model of the
# linear model that we would normally obtain ... consider this
#
# in a 2 dimensional linear regression, lets say that we produce a linear model that is a function of
# both x and y, but y increases less as a function of x, i.e. slope of the line is a proper
# fraction ... then we can flatten the line to be only a function of y and find the height
# in y from the x axis such that the error between this horizontal line y = y_0 and the original
# model (in both x and y) is minimized ... this is the purpose of using relu ... it gives us
# a feature selector (keep y, drop x) that best matches the original model that approximates the
# distribution of the data as a function of both x and y
#
# printing the weights and deciding the threshold that makes certain weights to be effectively zero
# gives us the feature selection that we seek
#
# in what's printed, you see the weights and the bias
print(wts)
# create an image of the relu for an example of how it works
os.system('/home/robert/code/scripts/r/djo/relu.R')
