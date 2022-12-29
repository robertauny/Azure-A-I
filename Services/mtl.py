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
## File:      mtl.py
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
from math                                           import ceil,log,exp,floor,isnan
from PIL                                            import ImageDraw,Image

ver  = sys.version.split()[0]

if ver == const.constants.VER:
    from            keras.layers                            import Dense,Embedding,BatchNormalization,Flatten,Reshape
    from            keras.models                            import Sequential,load_model,Model
    from            keras.utils.np_utils                    import to_categorical
    from            keras.preprocessing.text                import Tokenizer
    from            keras.preprocessing.sequence            import pad_sequences
else:
    from tensorflow.keras.layers                            import Dense,Embedding,BatchNormalization,Flatten,Reshape
    from tensorflow.keras.models                            import Sequential,load_model,Model
    from tensorflow.keras.utils                             import to_categorical
    from tensorflow.keras.preprocessing.text                import Tokenizer
    from tensorflow.keras.preprocessing.sequence            import pad_sequences

from ai                                             import create_kg,extendglove,thought,cognitive,img2txt,wikidocs,wikilabel,importance,store,unique
from nn                                             import dbn,calcC,nn_split,dbnlayers,calcN,clustering,nn_cleanse,nn_balance,nn_trim

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
    # files containing more data that should be cleaned and prepared
    #
    # modify this file logic if the data is being read from a DB or other source
    for fl in fls:
        # read the current data file
        dat  = pd.read_csv(fl,encoding="unicode_escape")
        # use other deep learning methods to cleanse the current data set
        cdat = nn_cleanse(inst,dat)
        keep = cdat["dat"].copy()
        # the input data
        k    = pd.DataFrame(cdat["dat"],columns=cdat["nhdr"]).astype(np.float32)
        # the header should be redefined
        nhdr = list(k.columns)
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
        print([nhdr,perms])
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
                    fns1 = const.constants.SEP.join([nhdr[i].translate(str.maketrans("","",punctuation)).replace(" ",const.constants.SEP).lower() for i in cols])
                    fns  = nhdr[col] + const.constants.SEP + fns1
                    # just skip this permutation if unable to fix the data
                    if type("") in utils.sif(k.to_numpy()[:,cls].astype(str).flatten()):
                        print("Not enough clean data to fix other data issues.")
                        break
                    # define the inputs to the model
                    sdat = nn_split(k.iloc[:,cls])
                    # now use the random cluster model to trim the dataset for best sensitivity
                    lbls = nn_trim(sdat["train"],0,1)
                    nlbls= [i for i in range(len(sdat["train"])) if i not in lbls]
                    label= [nlbls,lbls]
                    mdls = []
                    for s in label:
                        x    = pd.DataFrame( sdat["train"][s,1:],columns=np.asarray(nhdr)[cols])
                        # random field theory to calculate the number of clusters to form (or classes)
                        clust= max(2,len(unique(sdat["train"][s,0])))
                        keys = {}
                        # define the outputs of the model
                        fit  = sdat["train"][s,0].astype(np.int8)
                        y    = to_categorical(calcC(fit,clust,keys).flatten(),num_classes=clust)
                        # main model
                        #
                        # categorical column and classification prediction
                        #
                        # add some layers to the standard dbn for clustering to embed
                        # the integer values into the real numbers between 0 and 1
                        model= dbn(x.to_numpy()
                                  ,y
                                  ,sfl=sfl
                                  ,clust=clust)
                        # first model is a classifier that will be passed into the next model that will do the clustering
                        # then once centroids are known, any predictions will be relative to those centroids
                        #model= clustering(model,clust)
                        if not model is None:
                            mdls.append(model)
                    if len(mdls) == 0:
                        print("Models are null.")
                        break
                    # now use the random cluster model to trim the dataset for best sensitivity
                    lbls = nn_trim(sdat["test"],0,1)
                    nlbls= [i for i in range(len(sdat["test"])) if i not in lbls]
                    label= [nlbls,lbls]
                    preds= None
                    for i in range(len(label)):
                        # get some predictions using the same input data since this
                        # is just for simulation to produce graphics
                        #
                        # yet note that for markov processes of time series, the last prediction
                        # is the next value in the time series
                        pred = mdls[i].predict(sdat["test"][label[i],1:])
                        if len(np.asarray(pred).shape) > 1:
                            p    = []
                            for row in list(pred):
                                # start = 1 makes labels begin with 1, 2, ...
                                # in clustering, we find the centroids furthest from the center of all data
                                # the labels in this case are just the numeric values assigned in order
                                # and the data should be closest to this label
                                p.extend(j for j,x in enumerate(row,start=0) if abs(x-j) == min(abs(row-list(range(len(row))))))
                            pred = np.asarray(p)
                        else:
                            pred = np.asarray(list(pred))
                        # stack the recent predictions with the original inputs
                        if len(pred) == len(label[i]):
                            prds = np.hstack((pred.reshape((len(pred),1)),sdat["test"][label[i],:]))
                        else:
                            print("Prediction length doesn't match input data length.")
                            break
                        preds= prds if type(preds) == type(None) else np.vstack((preds,prds))
                        # produce some output
                        if len(preds) > 0:
                            pred0= preds[:,0]
                            pred1= preds[:,1]
                            idir = foldi + "/" + fns + "/"
                            if not os.path.exists(idir):
                                os.makedirs(idir,exist_ok=True)
                            fn   = idir + fns + const.constants.SEP 
                            # we need a data frame for the paired and categorial plots
                            df         = pd.DataFrame(preds).drop(columns=1)
                            df.columns = np.asarray(nhdr)[cls]
                            # get the paired plots and save them
                            utils.utils._pair(             df,fn+"grid.png",nhdr[col])
                            # x label
                            if len(x.to_numpy()[0]) > 1:
                                xlbl = "Enumeration of " + fns1 + " values"
                            else:
                                xlbl = x.columns[0]
                            # forecast plot with an extra element to account for the
                            # markov property allowing the element to be a prediction
                            # based on the last set of inputs
                            x11  = pd.Series(list(range(1,len(pred0)+1+1)),name=xlbl)
                            # add the markov prediction for the last element in the time series
                            x2   = pd.Series(np.append(pred1,pred0[-1]),name=nhdr[col]+" Values")
                            # get the swarm plots of the classification
                            utils.utils._swarm(        x11,x2,fn+"class.png",nhdr[col])
                            # these plots only work for binary classifications
                            if clust == 2:
                                # get the roc
                                utils.utils._roc(      pred1.astype(np.int8),pred0.astype(np.int8),fn+"roc.png")
                                # get the precision vs recall
                                utils.utils._pvr(      pred1.astype(np.int8),pred0.astype(np.int8),fn+"pvr.png")
                            # get the precision, recall, f-score
                            utils.utils._prf(          pred1.astype(np.int8),pred0.astype(np.int8),fn+"prf.txt")
                            # get the confusion matrix
                            utils.utils._confusion(    pred1.astype(np.int8),pred0.astype(np.int8),fn+"cnf.png")
                            # regression plot
                            utils.utils._joint(            x11,x2,[-10,2*len(pred0)],[0.5*min(pred1 ),1.5*max(pred1 )],fn+"forecast.png",nhdr[col])
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
                            res  = pred0 - np.asarray(list(pred1))
                            # fit vs residuals plot
                            utils.utils._fitVres(          x11,x2,res,fn+"fitVres.png")
