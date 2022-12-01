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
## File:      st.py
##
## Purpose:   Clean data and make predictions
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Feb. 10, 2022
##
############################################################################

import sys

import constants as const

from joblib                                         import Parallel,delayed
from string                                         import punctuation
from math                                           import ceil,log,exp,floor
from PIL                                            import ImageDraw,Image

ver  = sys.version.split()[0]

if ver == const.constants.VER:
    from            keras.models                            import Sequential,load_model,Model
    from            keras.utils.np_utils                    import to_categorical
    from            keras.preprocessing.text                import Tokenizer
    from            keras.preprocessing.sequence            import pad_sequences
else:
    from tensorflow.keras.models                            import Sequential,load_model,Model
    from tensorflow.keras.utils                             import to_categorical
    from tensorflow.keras.preprocessing.text                import Tokenizer
    from tensorflow.keras.preprocessing.sequence            import pad_sequences

from ai                                             import create_kg,extendglove,thought,cognitive,img2txt,wikidocs,wikilabel,importance,store
from nn                                             import dbn,calcC,nn_split,dbnlayers,calcN,nn_cleanse

import config
import data
import utils

import numpy             as np
import pandas            as pd
import multiprocessing   as mp
import seaborn           as sns
import matplotlib.pyplot as plt

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
fold = cfg["instances"][inst]["sources"][src][typ]["connection"]["fold"]

# we will get the subset of data rows/columns that are all non-null
# and make a determination of string vs. numeric
#
# strings will be labeled using wikilabel
if type(fls) in [type([]),type(np.asarray([]))] and len(fls) > 0:
    for fl in fls:
        dat  = pd.read_csv(fl,encoding="unicode_escape")
        ndat = nn_cleanse(inst,dat)
        nhdr = ndat["nhdr"]
        dat  = ndat["dat" ]
        sifs = ndat["sifs"]
        # predict each column and write some output
        if hasattr(const.constants,"PERMS"):
            if type(0) == utils.sif(const.constants.PERMS):
                perms= data.permute(list(range(0,len(nhdr))),mine=False,l=const.constants.PERMS)
            else:
                perms= const.constants.PERMS
        else:
            perms= data.permute(list(range(0,len(nhdr))),mine=False,l=len(nhdr))
        acols= const.constants.COLUMNS if hasattr(const.constants,"COLUMNS") else nhdr
        # construct the relaxed data name
        sfl  = const.constants.SFL if hasattr(const.constants,"SFL") else "models/obj.h5"
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
                    x    = pd.DataFrame(dat[:,cols].astype(np.single),columns=np.asarray(nhdr)[cols])
                    #x    = x.sort_values(by=list(x.columns))#sort_values(by=list(range(0,len(x.columns))),axis=1)#sort_values(by=np.asarray(nhdr)[cols],axis=1)#.to_numpy()
                    # the outputs to be fit
                    #if type(0.0) in sifs[:,col]:
                    if not type("") in sifs[:,col]:
                        # define the outputs of the model
                        fit  = dat[:,col].astype(np.single)
                        y    = fit
                        # floating point column and regression prediction
                        model= dbn(x.to_numpy()
                                  ,y
                                  ,sfl=sfl
                                  ,loss="mean_squared_error"
                                  ,optimizer="adam"
                                  ,rbmact="tanh"
                                  ,dbnact='tanh' if ver == const.constants.VER else 'selu'
                                  ,dbnout=1)
                    else:
                        # random field theory to calculate the number of clusters to form (or classes)
                        clust= calcN(len(dat))
                        # define the outputs of the model
                        fit  = dat[:,col].astype(np.int8)
                        y    = to_categorical(calcC(fit,clust).flatten(),num_classes=clust)
                        # categorical column and classification prediction
                        #
                        # sample the data set when building the clustering model
                        model= dbn(x.iloc[random.sample(list(range(0,len(x))),max(clust,np.int64(floor(const.constants.VSPLIT*len(x))))),:].to_numpy(),y,sfl=sfl,clust=clust)
                        # first model is a classifier that will be passed into the next model that will do the clustering
                        # then once centroids are known, any predictions will be relative to those centroids
                        model=clustering(model,clust)
                    if model is None:
                        print("Model is null.")
                        break
                    # save the model
                    fnm  = "models/" + fns + ".h5"
                    model.save(fnm)
                    # get some predictions using the same input data since this
                    # is just for simulation to produce graphics
                    #
                    # yet note that for markov processes of time series, the last prediction
                    # is the next value in the time series
                    pred = model.predict(x)
                    if len(np.asarray(pred).shape) > 1:
                        p    = []
                        for row in list(pred):
                            # start = 1 forces cluster labels to begin with 1,2,...
                            #p.extend(j for j,x in enumerate(row,start=1) if x == max(row))
                            #p.extend(j for j,x in enumerate(row,start=1) if x == max(abs(row)))
                            # in clustering, we find the centroids furthest from the center of all data
                            # the labels in this case are just the numeric values assigned in order
                            # and the data should be closest to this label
                            p.extend(j for j,x in enumerate(row,start=0) if abs(x-j) == min(abs(row-list(range(len(row))))))
                        pred = np.asarray(p)
                    else:
                        pred = np.asarray(list(pred))
                    # stack the recent predictions with the original inputs
                    if len(pred) == len(x):
                        preds= np.hstack((pred.reshape((len(pred),1)),x))
                    else:
                        print("Prediction length doesn't match input data length.")
                        break
                    # produce some output
                    if len(preds) > 0:
                        fn   = "images/" + fns + const.constants.SEP 
                        # we need a data frame for the paired and categorial plots
                        df   = pd.DataFrame(preds,columns=np.asarray(nhdr)[cls])
                        # get the paired plots and save them
                        utils.utils._pair(             df,fn+"grid.png",nhdr[col])
                        # x label
                        xlbl = const.constants.XLABEL if hasattr(const.constants,"XLABEL") else "Event Number"
                        # forecast plot with an extra element to account for the
                        # markov property allowing the element to be a prediction
                        # based on the last set of inputs
                        x11  = pd.Series(list(range(1,len(pred)+1+1)),name=xlbl)
                        # add the markov prediction for the last element in the time series
                        x2   = pd.Series(np.append(fit,pred[-1]),name=nhdr[col]+" Values")
                        # if classification do an additional plot
                        if not (type(0.0) in sifs[:,col] or type(0) in sifs[:,col]):
                            # get the swarm plots of the classification
                            utils.utils._swarm(        x11,x2,fn+"class.png",nhdr[col])
                            # these plots only work for binary classifications
                            if clust == 2:
                                # get the roc
                                utils.utils._roc(      fit.astype(np.int8),pred.astype(np.int8),fn+"roc.png")
                                # get the precision vs recall
                                utils.utils._pvr(      fit.astype(np.int8),pred.astype(np.int8),fn+"pvr.png")
                            # get the precision, recall, f-score
                            utils.utils._prf(          fit.astype(np.int8),pred.astype(np.int8),fn+"prf.txt")
                            # get the confusion matrix
                            utils.utils._confusion(    fit.astype(np.int8),pred.astype(np.int8),fn+"cnf.png")
                        else:
                            # get the r-square comparison
                            utils.utils._r2(fit,pred,fn+"r2.png")
                        # regression plot
                        utils.utils._joint(            x11,x2,[-10,2*len(pred)],[0.5*min(fit ),1.5*max(fit )],fn+"forecast.png",nhdr[col])
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
                        res  = pred - np.asarray(list(fit))
                        # fit vs residuals plot
                        utils.utils._fitVres(          x11,x2,res,fn+"fitVres.png")
