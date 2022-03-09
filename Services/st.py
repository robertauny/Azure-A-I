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
from math                                           import ceil,log,exp
from PIL                                            import ImageDraw,Image
from scipy                                          import stats

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

from ai                                             import create_kg,extendglove,thought,cognitive,img2txt,checkdata,fixdata,wikidocs,wikilabel,importance,store
from nn                                             import dbn,calcC,nn_split,dbnlayers,calcN

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

# gremlin imports
from gremlin_python.structure.graph                 import Graph
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection

cfg  = config.cfg()

np.random.seed(const.constants.SEED)

############################################################################
##
## Purpose:   Use A-I to expand and fix a data set before processing
##
############################################################################

def r2(x,y):
    return stats.pearson(x,y)[0]**2

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
        # going to capture the header and data so it can be replaced
        hdr = list(dat.columns)
        if hasattr(const.constants,"DROP" )                                                                  and \
           type(const.constants.DROP ) in [type([]),type(np.asarray([]))] and len(const.constants.DROP ) > 0:
            drop = dat.iloc[:,[hdr.index(i) for i in const.constants.DROP ]].to_numpy().copy()
            dat  = dat.drop(columns=const.constants.DROP )
            dhdr = list(np.asarray(hdr)[[hdr.index(i) for i in const.constants.DROP ]])
            nhdr = [i for i in  hdr if i not in dhdr]
        if hasattr(const.constants,"DATES")                                                                  and \
           type(const.constants.DATES) in [type([]),type(np.asarray([]))] and len(const.constants.DATES) > 0:
            dts  = dat.iloc[:,[hdr.index(i) for i in const.constants.DATES]].to_numpy().copy()
            dat  = dat.drop(columns=const.constants.DATES)
            thdr = list(np.asarray(hdr)[[hdr.index(i) for i in const.constants.DATES]])
            nhdr = [i for i in nhdr if i not in thdr]
        # replace any NaNs
        dat  = np.asarray([[str(x).lower().replace("nan","") for x in row] for row in dat.to_numpy()])
        dat  = pd.DataFrame(dat,columns=nhdr)
        # remove any completely null columns
        cols = list(dat.columns)
        chdr = []
        for i in range(0,len(cols)):
            d    = dat.iloc[:,i].to_list()
            if d == [""] * len(d):
                dat  = dat.drop(columns=cols[i])
                chdr.append(i)
        nhdr = [i for i in nhdr if i not in chdr]
        # now continue on
        dat  = dat.to_numpy()
        #coln = {hdr[k]:k for k in range(0,len(hdr)) if hdr[k] in nhdr}
        coln = {h:i for i,h in enumerate(nhdr)}
        # check which rows/columns have any null values and remove them
        d,rows,cols = checkdata(dat)
        # have to check that we still have rows/columns of data
        sifs = None
        if not (len(d) == 0 or len(d[0]) == 0):
            # if there is something to fix
            if len(rows) in range(1,len(dat)) or len(cols) in range(1,len(dat[0])):
                # indices that actually had data originally
                indxr= [j for j in range(0,len(dat)) if j not in rows]
                # string columns will be labeled using wikilabel
                for i in range(0,len(dat[0])):
                    # rows of sifs are the actual columns so transpose later
                    #vec  = np.asarray([t for t in map(utils.sif,dat[:,i])]).reshape((1,len(dat)))
                    vec  = [t for t in map(utils.sif,dat[:,i])]
                    if not type(sifs) == type(None):
                        #sifs = np.hstack((sifs,vec))
                        sifs = np.vstack((sifs,vec))
                        b    = type(0) in sifs[-1] or type(0.0) in sifs[-1]
                    else:
                        sifs = vec
                        b    = type(0) == sifs     or type(0.0) == sifs
                    if i in cols:
                        if not b:
                            #wiki = wikilabel(inst,dat[indxr,i],True,True)
                            ccd  = np.asarray(list(calcC(dat[:,i])))
                            dat[rows,i] = ccd.flatten()[rows]
                sifs = sifs.transpose() if not type(sifs) == type(None) else None
                # fix the data by intelligently filling missing values
                dat  = fixdata(inst,dat,coln)
                dat  = pd.DataFrame(dat,columns=nhdr)
                dat  = dat.to_numpy()
        # predict each column and write some output
        perms= data.permute(list(range(0,len(nhdr))),mine=False,l=const.constants.PERMS)
        acols= const.constants.COLUMNS if hasattr(const.constants,"COLUMNS") else nhdr
        for col in range(0,len(nhdr)):
            for cols in perms:
                if nhdr[col].lower() in [a.lower() for a in acols] and col not in cols:
                    print([nhdr[col],np.asarray(nhdr)[cols]])
                    # structure which columns are dependent and which are independent
                    cls  = [col]
                    cls.extend(cols)
                    # just skip this permutation if unable to fix the data
                    if type("") in [t for t in map(utils.sif,dat[:,cls].astype(str).flatten())]:
                        print("Not enough clean data to fix other data issues.")
                        break
                    # define the inputs to the model
                    x    = pd.DataFrame(dat[:,cols].astype(np.single),columns=np.asarray(nhdr)[cols])
                    x    = x.sort_values(by=list(x.columns))#sort_values(by=list(range(0,len(x.columns))),axis=1)#sort_values(by=np.asarray(nhdr)[cols],axis=1)#.to_numpy()
                    # build a simple model
                    model= Sequential()
                    # relu at the input layer
                    dbnlayers(model,len(cols),len(cols),'relu',False)
                    # the outputs to be fit
                    if type(0.0) in sifs[:,col]:
                        fit  = dat[:,col].astype(np.single)
                        # floating point column and regression prediction
                        dbnlayers(model,1,len(cols),'tanh' if ver == const.constants.VER else 'selu',False)
                        # compile the model
                        #model.compile(loss="mean_squared_error",optimizer="sgd")
                        model.compile(loss="mean_squared_error",optimizer="adam")
                        # define the outputs of the model
                        y    = fit
                    else:
                        fit  = dat[:,col].astype(np.int8)
                        # random field theory to calculate the number of clusters to form (or classes)
                        clust= calcN(len(dat))
                        # creating a restricted Boltzmann machine here
                        dbnlayers(model,len(cols),len(cols),'tanh' if ver == const.constants.VER else 'selu',False)
                        # categorical column and classification prediction
                        dbnlayers(model,clust    ,len(cols),'sigmoid'                                       ,False)
                        # compile the model
                        model.compile(loss=const.constants.LOSS,optimizer=const.constants.OPTI)
                        # define the outputs of the model
                        y    = to_categorical(calcC(fit,clust).flatten(),num_classes=clust)
                    # fit the model
                    model.fit(x=x,y=y,epochs=const.constants.EPO,verbose=const.constants.VERB)
                    # get some predictions using the same input data since this
                    # is just for simulation to produce graphics
                    pred = model.predict(x)
                    if len(np.asarray(pred).shape) > 1:
                        p    = []
                        for row in list(pred):
                            p.extend(j for j,x in enumerate(row) if x == max(row))
                        pred = np.asarray(p)
                    else:
                        pred = np.asarray(list(pred))
                    # stack the recent predictions with the original inputs
                    if len(pred) == len(x):
                        preds= np.hstack((pred.reshape((len(pred),1)),x))
                    else:
                        continue
                    # produce some output
                    if len(preds) > 0:
                        fn   = "images/" + const.constants.SEP.join([nhdr[i].translate(str.maketrans("","",punctuation)).replace(" ",const.constants.SEP).lower() for i in cls]) + const.constants.SEP
                        # we need a data frame for the paired and categorial plots
                        df   = pd.DataFrame(preds,columns=np.asarray(nhdr)[cls])
                        # get the paired plots and save them
                        g    = sns.pairplot(df,hue=nhdr[col])
                        g.fig.suptitle(nhdr[col]+" Grid of Marginals")
                        # save the plot just created
                        plt.savefig(fn+"grid.png")
                        plt.cla()
                        plt.clf()
                        # x label
                        xlbl = const.constants.XLABEL if hasattr(const.constants,"XLABEL") else "Event Number"
                        # forecast plot
                        x11  = pd.Series(list(range(1,len(pred)+1)),name=xlbl)
                        x2   = pd.Series(fit,name=nhdr[col]+" Values")
                        # if classification do an additional plot
                        if type(0) in sifs[:,col]:
                            # get the paired plots and save them
                            sns.swarmplot(y=x11,x=x2)
                            # save the plot just created
                            plt.savefig(fn+"class.png")
                            plt.cla()
                            plt.clf()
                        g    = sns.jointplot(x=x11
                                            ,y=x2
                                            ,kind="reg"
                                            ,xlim=[-10,2*len(pred)]
                                            #,xlim=[0.5*min(pred),1.5*max(pred)]
                                            ,ylim=[0.5*min(fit ),1.5*max(fit )])#,stat_func=r2)#,hue=nhdr[col])
                        g.fig.suptitle("Forecast of "+nhdr[col])
                        # save the plot just created
                        plt.savefig(fn+"forecast.png")
                        plt.cla()
                        plt.clf()
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
                        # Two plots
                        fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(12,6))
                        # Histogram of residuals
                        sns.distplot(res,ax=ax1)
                        ax1.set_title("Histogram of Residuals")
                        # Fitted vs residuals
                        x1   = pd.Series(pred,name=xlbl)
                        sns.kdeplot(x11,x2,ax=ax2,n_levels=40)
                        sns.regplot(x=x11,y=x2,scatter=False,ax=ax2)
                        ax2.set_title("Fitted vs. Actual Values")
                        #ax2.set_aspect("equal")
                        # save the plot just created
                        plt.savefig(fn+"fitVres.png")
                        plt.cla()
                        plt.clf()
