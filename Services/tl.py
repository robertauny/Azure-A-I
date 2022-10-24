#!/usr/bin/python

############################################################################
# Begin license text.
# Copyright Feb. 27, 2020 Robert A. Murphy

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# End license text.
############################################################################

############################################################################
##
## File:      tl.py
##
## Purpose:   Transfer learning and domain adaptation.
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Jun. 13, 2021
##
############################################################################

import sys

import includes

import constants as const

from joblib                                         import Parallel,delayed
from string                                         import punctuation
from math                                           import ceil,log,exp,floor
from PIL                                            import ImageDraw,Image

ver  = sys.version.split()[0]

if ver == const.constants.VER:
    from            keras.layers                            import Dense
    from            keras.models                            import Sequential,load_model,Model
    from            keras.utils.np_utils                    import to_categorical
    from            keras.preprocessing.text                import Tokenizer
    from            keras.preprocessing.sequence            import pad_sequences
else:
    from tensorflow.keras.layers                            import Dense
    from tensorflow.keras.models                            import Sequential,load_model,Model
    from tensorflow.keras.utils                             import to_categorical
    from tensorflow.keras.preprocessing.text                import Tokenizer
    from tensorflow.keras.preprocessing.sequence            import pad_sequences

from ai                                             import create_kg,extendglove,thought,cognitive,img2txt,wikidocs,wikilabel,importance,store,unique
from nn                                             import dbn,calcC,nn_split,dbnlayers,calcN,clustering,nn_cleanse

import config
import utils
import data

import numpy             as np
import pandas            as pd
import multiprocessing   as mp
import seaborn           as sns
import matplotlib.pyplot as plt

import os
import csv
import glob
import random

cfg  = config.cfg()

np.random.seed(const.constants.SEED)

# an instance number corresponds to the set of constants in one object
# of the JSON configuration array associated to the set of modules for this app
inst = 0

# getting constants from the JSON config file
src  = cfg["instances"][inst]["src"]["index"]
typ  = cfg["instances"][inst]["src"]["types"]["main"]
fls  = cfg["instances"][inst]["sources"][src][typ]["connection"]["files"]
flx  = cfg["instances"][inst]["sources"][src][typ]["connection"]["xfile"]
foldi= cfg["instances"][inst]["sources"][src][typ]["connection"]["foldi"]
foldm= cfg["instances"][inst]["sources"][src][typ]["connection"]["foldm"]

# we will get the subset of data rows/columns that are all non-null
# and make a determination of string vs. numeric
#
# strings will be labeled using wikilabel
if (type(fls) in [type([]),type(np.asarray([]))] and len(fls) > 0) and \
   (os.path.exists(flx) and os.stat(flx).st_size > 0):
    sfl  = const.constants.SFL if hasattr(const.constants,"SFL") else "models/obj.h5"
    # get the transfer learning data set ready for use
    tdat = pd.read_csv(flx,encoding="unicode_escape")
    cdat = nn_cleanse(inst,tdat)
    tnhdr= cdat["nhdr"]
    tdat = cdat["dat" ]
    # define the inputs to the model
    xt   = pd.DataFrame(tdat.astype(np.single),columns=np.asarray(tnhdr))
    xt   = xt.to_numpy()
    # pretrained model
    #
    # to keep the size of the model smaller while minimizing execution times
    # the larger data set can be sampled to obtain a minimal representation
    # of the underlying distribution while also capturing other nuances that
    # may not be present in the data set at one source
    #
    # to accomplish this, we will assume that the data are normally distributed
    # and apply an order statistic ... the result is a beta distributed data set
    # about which we can argue that its parameters can be chosen such that the
    # resulting beta distribution is actually uniform in nature ... this follows
    # from the Markov property (sequence of dependent random variables, one for each
    # data source, such that successive samples capture all information about all preceding
    # samples, and the most recent sample only depends upon its most recent predecessor)
    #
    # after ordering, we assume uniformity of the distribution that was sampled accordingly
    mdlr = dbn(xt
              ,xt
              ,loss="mean_squared_error"
              ,optimizer="adam"
              ,rbmact="tanh"
              ,dbnact='tanh' if ver == const.constants.VER else 'selu'
              ,dbnout=len(xt[0]))
    # don't want to retrain the layers of the pre-trained model
    #
    # if these layers are trainable, then an error will persist
    for layer in mdlr.layers:
        layer.trainable=False
    # files containing more data that should be cleaned and prepared
    #
    # modify this file logic if the data is being read from a DB or other source
    for fl in fls:
        # read the current data file
        dat  = pd.read_csv(fl,encoding="unicode_escape")
        # use other deep learning methods to cleanse the current data set
        cdat = nn_cleanse(inst,dat)
        # the set of returns after the cleansing
        #
        # header of all columns remaining (some may have been configured to be left out see constants.py)
        nhdr = cdat["nhdr"]
        # the data corresponding to the columns that were cleansed
        dat  = cdat["dat" ]
        # the data types of the columns in question ... these should be auto-detected
        #
        # at present, we are only concerned with continuous (floating point) and categorical (strings converted to integer)
        sifs = cdat["sifs"]
        # predict each column and write some output for the columns defined in perms
        if hasattr(const.constants,"PERMS"):
            # length of the permutation of columns in the data set is defined in constants.py
            if type(0) == utils.sif(const.constants.PERMS):
                perms= data.permute(list(range(0,len(nhdr))),mine=False,l=const.constants.PERMS)
            else:
                perms= const.constants.PERMS
        else:
            perms= data.permute(list(range(0,len(nhdr))),mine=False,l=len(nhdr))
        # the set of columns that should remain in the data set (configured in constants.py)
        acols= const.constants.COLUMNS if hasattr(const.constants,"COLUMNS") else nhdr
        # construct the relaxed data name and build some models
        for col in range(0,len(nhdr)):
            for cols in perms:
                if nhdr[col].lower() in [a.lower() for a in acols] and col not in cols:
                    # which loop iteration
                    print([nhdr[col],np.asarray(nhdr)[cols]])
                    # structure which columns are dependent and which are independent
                    cls  = [col]
                    cls.extend(cols)
                    # file naming convention
                    fns  = const.constants.SEP.join([nhdr[i].translate(str.maketrans("","",punctuation)).replace(" ",const.constants.SEP).lower() for i in cls])
                    # just skip this permutation if unable to fix the data
                    if type("") in utils.sif(dat[:,cls].astype(str).flatten()):
                        print("Not enough clean data to fix other data issues.")
                        break
                    # define the inputs to the model
                    sdat = nn_split(pd.DataFrame(dat[:,cls].astype(np.single)))
                    x    = pd.DataFrame( sdat["train"][:,1:],columns=np.asarray(nhdr)[cols])
                    # the outputs to be fit
                    if not type("") in sifs[:,col]:
                        # define the outputs of the model
                        fit  =  sdat["train"][:,0]
                        y    =  fit
                        # main model
                        #
                        # the main model employs the pretrained model after converting the inputs to be
                        # of the same dimension expected by the pretrained model
                        #
                        # to make this work, we add a dense layer (before and after) the pretrained model
                        # with the after dense layer being needed to make the outputs match what's expected
                        # at the input layer of the first layer of the remaining deep learning modeling
                        model= dbn(x.to_numpy()
                                  ,y
                                  ,sfl=sfl
                                  ,encs=[Dense(len(xt[0]),input_shape=(len(cols ),),activation='relu'                                          )
                                        ,mdlr
                                        ,Dense(len(cols ),input_shape=(len(xt[0]),),activation='tanh' if ver == const.constants.VER else 'selu')]
                                  ,loss="mean_squared_error"
                                  ,optimizer="adam"
                                  ,rbmact="tanh"
                                  ,dbnact='tanh' if ver == const.constants.VER else 'selu'
                                  ,dbnout=1)
                    else:
                        # random field theory to calculate the number of clusters to form (or classes)
                        #clust= calcN(len(dat))
                        clust= max(2,len(unique(sdat["train"][:,0])))
                        keys = {}
                        # define the outputs of the model
                        fit  = sdat["train"][:,0].astype(np.int8)
                        y    = to_categorical(calcC(fit,clust,keys).flatten(),num_classes=clust)
                        # main model
                        #
                        # the main model employs the pretrained model after converting the inputs to be
                        # of the same dimension expected by the pretrained model
                        #
                        # to make this work, we add a dense layer (before and after) the pretrained model
                        # with the after dense layer being needed to make the outputs match what's expected
                        # at the input layer of the first layer of the remaining deep learning modeling
                        #
                        # categorical column and classification prediction
                        #
                        # sample the data set when building the clustering model
                        model= dbn(x.iloc[random.sample(list(range(0,len(x))),max(clust,np.int64(floor(const.constants.VSPLIT*len(x))))),:].to_numpy()
                                  ,y
                                  ,sfl=sfl
                                  ,encs=[Dense(len(xt[0]),input_shape=(len(cols ),),activation='relu'                                          )
                                        ,mdlr
                                        ,Dense(len(cols ),input_shape=(len(xt[0]),),activation='tanh' if ver == const.constants.VER else 'selu')]
                                  ,clust=clust)
                        # first model is a classifier that will be passed into the next model that will do the clustering
                        # then once centroids are known, any predictions will be relative to those centroids
                        model=clustering(model,clust)
                    if model is None:
                        print("Model is null.")
                        break
                    # save the model
                    mdir = foldm + "/" + fns + "/"
                    if not os.path.exists(mdir):
                        os.makedirs(mdir,exist_ok=True)
                    fnm  = mdir + fns + ".h5"
                    model.save(fnm)
                    # get some predictions using the same input data since this
                    # is just for simulation to produce graphics
                    #
                    # yet note that for markov processes of time series, the last prediction
                    # is the next value in the time series
                    pred = model.predict(sdat["test"][:,1:])
                    if len(np.asarray(pred).shape) > 1:
                        p    = []
                        for row in list(pred):
                            p.extend(j for j,x in enumerate(row,start=1) if x == max(row))
                        pred = np.asarray(p)
                    else:
                        pred = np.asarray(list(pred))
                    # stack the recent predictions with the original inputs
                    if len(pred) == len(sdat["test"]):
                        preds= np.hstack((pred.reshape((len(pred),1)),sdat["test"][:,1:]))
                    else:
                        print("Prediction length doesn't match input data length.")
                        break
                    # produce some output
                    if len(preds) > 0:
                        idir = foldi + "/" + fns + "/"
                        if not os.path.exists(idir):
                            os.makedirs(idir,exist_ok=True)
                        fn   = idir + fns + const.constants.SEP 
                        # we need a data frame for the paired and categorial plots
                        df   = pd.DataFrame(preds,columns=np.asarray(nhdr)[cls])
                        # get the paired plots and save them
                        utils.utils._pair(             df,fn+"grid.png",nhdr[col])
                        # x label
                        if len(x.to_numpy()[0]) > 1:
                            xlbl = const.constants.XLABEL if hasattr(const.constants,"XLABEL") else "Event Number"
                        else:
                            xlbl = x.columns[0]
                        # forecast plot with an extra element to account for the
                        # markov property allowing the element to be a prediction
                        # based on the last set of inputs
                        x11  = pd.Series(list(range(1,len(pred)+1+1)),name=xlbl)
                        # add the markov prediction for the last element in the time series
                        x2   = pd.Series(np.append(sdat["test"][:,0],pred[-1]),name=nhdr[col]+" Values")
                        # if classification do an additional plot
                        if not (type(0.0) in sifs[:,col] or type(0) in sifs[:,col]):
                            # get the swarm plots of the classification
                            utils.utils._swarm(        x11,x2,fn+"class.png",nhdr[col])
                            # these plots only work for binary classifications
                            if clust == 2:
                                # get the roc
                                utils.utils._roc(      sdat["test"][:,0].astype(np.int8),pred.astype(np.int8),fn+"roc.png")
                                # get the precision vs recall
                                utils.utils._pvr(      sdat["test"][:,0].astype(np.int8),pred.astype(np.int8),fn+"pvr.png")
                            # get the precision, recall, f-score
                            utils.utils._prf(          sdat["test"][:,0].astype(np.int8),pred.astype(np.int8),fn+"prf.txt")
                            # get the confusion matrix
                            utils.utils._confusion(    sdat["test"][:,0].astype(np.int8),pred.astype(np.int8),fn+"cnf.png")
                        else:
                            # get the r-square comparison
                            utils.utils._r2(sdat["test"][:,0],pred,fn+"r2.png")
                        # regression plot
                        utils.utils._joint(            x11,x2,[-10,2*len(pred)],[0.5*min(sdat["test"][:,0] ),1.5*max(sdat["test"][:,0] )],fn+"forecast.png",nhdr[col])
                        # other plots to show that the built model is markovian
                        # since it will (hopefully) be shown that the errors are
                        # normally distributed
                        #
                        # recall that the typical linear model of an assumed normally
                        # distributed sample (central limit theorem) is the sum of a
                        # deterministic hyperplane (as a lower dimensional subspace of
                        # the sample space) plus normally distributed noise ... it's this
                        # noise that we will show graphically, lending credence to the
                        # gaussian distributed sample
                        #
                        # in addition, recall that the simplest markov model is the sum of
                        # a measured start point, plus gaussian white noise that accumulates
                        # at each time step, requiring that each new measurement only depends
                        # on the last measurement plus more additive noise
                        #
                        # residual errors (noise)
                        res  = pred - np.asarray(list(sdat["test"][:,0]))
                        # fit vs residuals plot
                        utils.utils._fitVres(          x11,x2,res,fn+"fitVres.png")
