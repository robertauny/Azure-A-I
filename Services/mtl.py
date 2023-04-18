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
from nn                                             import dbn,calcC,nn_split,dbnlayers,calcN,clustering,nn_cleanse,nn_balance,nn_trim,nn_energy
from sklearn.ensemble                               import RandomForestClassifier

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

def threshold(pred):
    if len(np.asarray(pred).shape) > 1:
        p    = []
        for row in list(pred):
            # start = 1 makes labels begin with 1, 2, ...
            # in clustering, we find the centroids furthest from the center of all data
            # the labels in this case are just the numeric values assigned in order
            # and the data should be closest to this label
            p.extend(j for j,x in enumerate(row,start=0) if abs(x-j) == min(abs(row-list(range(len(row))))))
        ret  = np.asarray(p)
    else:
        ret  = np.asarray(list(pred))
    return ret

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
                    typs = {"nnrf":"Random Cluster + DBN","nn1":"DBN (function)","nn2":"DBN (layered)","rf":"Random Forest"}
                    for typ in typs:
                        if typ == "nnrf":
                            # now use the random cluster model to trim the dataset for best sensitivity
                            lbls,_= nn_trim(sdat["train"],0,1)
                            nlbls = [i for i in range(len(sdat["train"])) if i not in lbls]
                            label = [nlbls,lbls]
                            mdls  = []
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
                            #
                            # for testing and prediction we don't have labels but this is ok
                            # we rely upon theory from Murphy detailing the uniform nature of the distribution
                            # after projection of the higher dimensional data to the 2-D space to estimate the
                            # labels which are only used for determination of the boundary separating the outer
                            # region of constant labels from the interior where the real action occurs
                            #
                            # thinking of the projection as an image, the outer region is opaque background out
                            # to infinity while the finite bounded region contains separable entities for segmentation
                            # localization, etc.
                            #
                            # since we are only after an estimate of the boundary then we can tailor the number
                            # of live data points in the data set that correspond to fraud when performing the estimation
                            # by assigning the labels to the test data
                            #
                            # for the next 4 lines to be valid, random field theory requires the test data set to be
                            # uniform, which can be accomplished by order a normally distributed data set, where
                            # the normal distribution can be assumed by the central limit theorem
                            #
                            # recall from the setup in the theory, to calculate the number of classes to form, we
                            # know that the data will be uniform since the number of colors guarantees uniformity
                            # in addition to the assumption of normality of the data followed by an order statistic
                            # being applied ... assuming also a square lattice, we calculate the critical probability
                            # by using unformity and counting a ratio of bounded regions that sub-divide the overall
                            # region and equate this ratio to the known value for the critical probability of 1/2
                            # which allows us to solve for the mean number of classes given the number of data points
                            # in the sample
                            #
                            # so we can use calcN to get the number of classes and use uniformity to sample a portion
                            # of all of the positive data points in each smaller region ... then we will get a smaller
                            # bounded region within the larger bounded region and apply each model to the sub-divides
                            # this will be done for each of the smaller regions and we will collect them all
                            #
                            # random field theory to get the mean number of classes to form
                            crows= calcN(len(sdat["test"]))
                            # the number of positive samples
                            rows = ceil(len(sdat["test"])*const.constants.POIS)
                            # the number of positive samples per class region
                            pspr = ceil(rows/crows)
                            # the number of samples per class region
                            spr  = ceil(len(sdat["test"])/crows)
                            # processing predictions for each class region
                            preds= None
                            for j in range(crows):
                                rng  = list(range(j*spr,min((j+1)*spr,len(sdat["test"]))))
                                # the metroplis algorithm works to reduce entropy among data points
                                # in a click as the equilibrium distribution is one of minimal energy
                                # typically this low energy state would be found as the lowest energy
                                # values of a function that serves as the input to a normalized
                                # exponential distribution
                                #
                                # however, since our data is comprised of IP addresses (origin and dest),
                                # clicks, date of observed clicks count and a fraud label, then the only
                                # input to the energy function in this case will only be the clicks count
                                #
                                # so we simply use the counts to determine highest energy states to indicate fraud
                                #
                                # note that we are only estimating the boundary of the inner region using
                                # these values ... in the training case, we already have this label ... in
                                # the testing case we need estimates of the labels to estimate the boundary
                                # before attempting the predictions of the labels (set as one in the estimate)
                                #
                                # another phenomenon of note is that each of the 2 models for the binary
                                # classifier don't determine one class or the other, as they both predict
                                # elements of either label, even when only one label is used in training, as
                                # is the case for the outer region's model ... this actually comports with theory
                                # since 4 classes are predicted, and this is what's seen, since each model
                                # acts as if it's predicting 2 classes separate from the other 2 classes being
                                # predicted by the other model
                                #
                                # for the future, we will want to have an energy function to which clicks will
                                # be passed so that we can gaug the energy in the clique of sites, after which
                                # we can sort and do the rest that comes after
                                #cliks= nn_energy(sdat["test"],0,1,False,True)
                                #cliks= nn_energy(sdat["test"],0,1,False,False)
                                #cliks= sdat["test"][rng,list(np.asarray(nhdr)[cls]).index("CLICKS")]
                                #cliks= nn_energy(sdat["test"][rng,list(np.asarray(nhdr)[cls]).index("CLICKS")],0,1,False,False)
                                #cliks= nn_energy(sdat["test"][rng,list(np.asarray(nhdr)[cls]).index("CLICKS")],0,1,False,False)
                                #cliks= nn_energy(sdat["test"][rng,list(np.asarray(nhdr)[cls]).index("CLICKS")],0,1,False,False)
                                #cliks= nn_energy(sdat["test"][rng,list(np.asarray(nhdr)[cls]).index("CLICKS")],0,1,False,True)
                                #cliks= nn_energy(sdat["test"][rng,[0,list(np.asarray(nhdr)[cls]).index("CLICKS")]],0,1,False,False)
                                cliks= nn_energy(sdat["test"][rng][:,[0,list(np.asarray(nhdr)[cls]).index("CLICKS")]],0,1,False,True)
                                #cliks= sdat["test"][rng,list(np.asarray(nhdr)[cls]).index("CLICKS")]
                                inds = np.argsort(cliks)
                                samps= np.unique(np.asarray(rng)[inds[range(len(rng)-pspr+1,len(rng))]])
                                bound= np.zeros(len(rng))
                                bound[samps-rng[0]] = np.ones(len(samps))
                                # end assumption above
                                lbls,_= nn_trim(np.hstack((bound.reshape((len(bound),1)),sdat["test"][rng,1:])),0,1)
                                nlbls = [i for i in range(len(rng)) if i not in lbls]
                                label = [nlbls,lbls]
                                for i in range(len(label)):
                                    # get some predictions using the same input data since this
                                    # is just for simulation to produce graphics
                                    #
                                    # yet note that for markov processes of time series, the last prediction
                                    # is the next value in the time series
                                    pred = mdls[i].predict(sdat["test"][label[i],1:])
                                    pred = threshold(pred)
                                    # stack the recent predictions with the original inputs
                                    if len(pred) == len(label[i]):
                                        prds = np.hstack((pred.reshape((len(pred),1)),sdat["test"][label[i],:]))
                                    else:
                                        print("Prediction length doesn't match input data length.")
                                        break
                                    preds= prds if type(preds) == type(None) else np.vstack((preds,prds))
                        else:
                            x    = pd.DataFrame( sdat["train"][:,1:],columns=np.asarray(nhdr)[cols])
                            # random field theory to calculate the number of clusters to form (or classes)
                            clust= max(2,len(unique(sdat["train"][:,0])))
                            keys = {}
                            # define the outputs of the model
                            fit  = sdat["train"][:,0].astype(np.int8)
                            y    = to_categorical(calcC(fit,clust,keys).flatten(),num_classes=clust)
                            if typ == "nn1":
                                # main model
                                #
                                # categorical column and classification prediction
                                #
                                # add some layers to the standard dbn for clustering to embed
                                # the integer values into the real numbers between 0 and 1
                                model= dbn(x.to_numpy()
                                          ,y
                                          ,sfl=None
                                          ,clust=clust)
                                # get some predictions using the same input data since this
                                # is just for simulation to produce graphics
                                #
                                # yet note that for markov processes of time series, the last prediction
                                # is the next value in the time series
                                pred = model.predict(sdat["test"][:,1:])
                                pred = threshold(pred)
                                # stack the recent predictions with the original inputs
                                preds= np.hstack((pred.reshape((len(pred),1)),sdat["test"]))
                            else:
                                if typ == "nn2":
                                    # main model
                                    #
                                    # categorical column and classification prediction
                                    #
                                    # add some layers to the standard dbn for clustering to embed
                                    # the integer values into the real numbers between 0 and 1
                                    model= Sequential()
                                    dbnlayers(model,sdat["train"].shape[1]-1,tuple((sdat["train"].shape[1]-1,)),const.constants.RBMA,False)
                                    dbnlayers(model,sdat["train"].shape[1]-1,sdat["train"].shape[1]-1,const.constants.RBMA,False)
                                    dbnlayers(model,clust,sdat["train"].shape[1]-1,"sigmoid",False)
                                    model.compile(loss=const.constants.LOSS,optimizer=const.constants.OPTI)
                                    model.fit(x=x,y=y,epochs=const.constants.EPO,verbose=const.constants.VERB)
                                    # get some predictions using the same input data since this
                                    # is just for simulation to produce graphics
                                    #
                                    # yet note that for markov processes of time series, the last prediction
                                    # is the next value in the time series
                                    pred = model.predict(sdat["test"][:,1:])
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
                                    preds= np.hstack((pred.reshape((len(pred),1)),sdat["test"]))
                                else:
                                    model= RandomForestClassifier(max_depth=2,random_state=0)
                                    model.fit(x,y)
                                    preds= model.predict(sdat["test"][:,1:].astype(np.int8))
                                    preds= np.hstack((np.asarray([list(preds[i]).index(1) for i in range(len(preds))]).reshape((len(preds),1)),sdat["test"]))
                        # produce some output
                        if len(preds) > 0:
                            pred0= preds[:,0]
                            pred1= preds[:,1]
                            # estimate the probability of obtaining the positive class label
                            upred= unique(pred1.astype(np.int8))
                            probs= list(np.zeros(len(upred)))
                            for j in upred:
                                prob                  = [j for i in range(len(pred1)) if pred1[i] == j]
                                probs[upred.index(j)] = 1.0 - len(prob)/len(pred1)
                            idir = foldi + "/" + fns + "/" + typ + "/"
                            if not os.path.exists(idir):
                                os.makedirs(idir,exist_ok=True)
                            fn   = idir + fns + const.constants.SEP 
                            # we need a data frame for the paired and categorial plots
                            df         = pd.DataFrame(preds).drop(columns=1)
                            df.columns = np.asarray(nhdr)[cls]
                            # get the paired plots and save them
                            utils.utils._pair(             df,fn+"grid.png",typs[typ]+" Analysis of "+nhdr[col])
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
                            utils.utils._swarm(        x11,x2,fn+"class.png",nhdr[col]+" using "+typs[typ])
                            # these plots only work for binary classifications
                            if clust == 2:
                                # get the roc
                                utils.utils._roc(      pred1.astype(np.int8),pred0.astype(np.int8),fn+"roc.png")
                                # get the precision vs recall
                                #utils.utils._pvr(      pred1.astype(np.int8),pred0.astype(np.int8),fn+"pvr.png")
                                utils.utils._pvr(      pred1.astype(np.int8),np.asarray(list(map(lambda x: probs[upred.index(x.astype(np.int8))],pred0))),fn+"pvr.png")
                            # get the precision, recall, f-score
                            utils.utils._prf(          pred1.astype(np.int8),pred0.astype(np.int8),fn+"prf.txt")
                            # get the confusion matrix
                            utils.utils._confusion(    pred1.astype(np.int8),pred0.astype(np.int8),fn+"cnf.png")
                            # regression plot
                            utils.utils._joint(            x11,x2,[-10,2*len(pred0)],[0.5*min(pred1 ),1.5*max(pred1 )],fn+"forecast.png",nhdr[col]+" using "+typs[typ])
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
                            utils.utils._fitVres(          x11,x2,res,fn+"fitVres.png",typs[typ])
