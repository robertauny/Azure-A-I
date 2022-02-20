#!/usr/bin/python

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

from ai                                             import create_kg,extendglove,thought,cognitive,img2txt,checkdata,fixdata,wikidocs,wikilabel,importance
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
        if not (len(d) == 0 or len(d[0]) == 0):
            # indices that actually had data originally
            indxr= [j for j in range(0,len(dat)) if j not in rows]
            # string columns will be labeled using wikilabel
            for i in range(0,len(dat[0])):
                if i in cols:
                    #wiki = wikilabel(inst,dat[indxr,i],True,True)
                    dat[indxr,i] = list(calcC(dat[indxr,i]).flatten())
                else:
                    if type("") in [t for t in map(utils.sif,dat[:,i])]:
                        #wiki = wikilabel(inst,dat[:,i],True,True)
                        dat[:,i] = list(calcC(dat[:,i]).flatten())
            # fix the data by intelligently filling missing values
            dat  = fixdata(inst,dat,coln)
            dat  = pd.DataFrame(dat,columns=nhdr)
            dat  = dat.to_numpy()
        # predict each column and write some output
        for col in range(0,len(nhdr)):
            # begin and end columns for each loop
            b    = col + 1
            e    = col + const.constants.MAX_COLS + 1
            # structure which columns are dependent and which are independent
            cls  = [col]
            cols = list(range(b,e))
            cls.extend(cols)
            # define the inputs to the model
            x    = dat[:,cols].astype(np.single)
            # build a simple model
            model= Sequential()
            # relu at the input layer
            dbnlayers(model,len(cols),len(cols),'relu',False)
            # the outputs to be fit
            fit  = dat[:,col].astype(np.single)
            if type(0.0) in [t for t in map(utils.sif,dat[:,col])]:
                # floating point column and regression prediction
                dbnlayers(model,1,len(cols),'tanh' if ver == const.constants.VER else 'selu',False)
                # compile the model
                model.compile(loss="mean_squared_error",optimizer="sgd")
                # define the outputs of the model
                y    = fit
            else:
                # random field theory to calculate the number of clusters to form
                clust= calcN(len(dat))
                # creating a restricted Boltzmann machine here
                dbnlayers(model,len(cols),len(cols),'tanh' if ver == const.constants.VER else 'selu',False)
                # categorical column and classification prediction
                dbnlayers(model,clust,len(cols),'sigmoid',False)
                # compile the model
                model.compile(loss=const.constants.LOSS,optimizer=const.constants.OPTI)
                # define the outputs of the model
                y    = to_categorical(calcC(fit,clust).flatten(),num_classes=clust)
            # fit the model
            model.fit(x=x,y=y,epochs=const.constants.EPO,verbose=const.constants.VERB)
            # get some predictions using the same input data since this
            # is just for simulation to produce graphics
            pred = model.predict(x)
            # stack the recent predictions with the original inputs
            preds= np.hstack((pred,x))
            # produce some output
            if len(preds) > 0:
                # we need a data frame for the paired plots
                df   = pd.DataFrame(preds)#,columns=np.asarray(nhdr)[cls])
                # get the paired plots and save them
                sns.pairplot(df)
                # save the plot just created
                plt.savefig("images/"+const.constants.SEP.join([i for i in map(str,cls)])+const.constants.SEP+"grid.png")
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
                res  = store(pred) if len(np.asarray(pred).shape) > 1 else list(pred) - list(fit)
                # Two plots
                fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(12,6))
                # Histogram of residuals
                sns.distplot(res,ax=ax1)
                ax1.set_title("Histogram of Residuals")
                # save the plot just created
                plt.savefig("images/"+const.constants.SEP.join([i for i in map(str,cls)])+const.constants.SEP+"hist.png")
                # Fitted vs residuals
                x1 = pd.Series(pred,name="Fitted "+nhdr[col])
                x2 = pd.Series(fit[nhdr[col]], name=nhdr[col]+" Values")
                sns.kdeplot(x1,x2,n_levels=40,ax=ax2)
                sns.regplot(x=x1,y=x2,scatter=False,ax=ax2)
                ax2.set_title("Fitted vs. Actual Values")
                ax2.set_xlim([0,120])
                ax2.set_ylim([0,120])
                ax2.set_aspect("equal")
                # save the plot just created
                plt.savefig("images/"+const.constants.SEP.join([i for i in map(str,cls)])+const.constants.SEP+"fitVres.png")
            if e >= len(nhdr):
                break
