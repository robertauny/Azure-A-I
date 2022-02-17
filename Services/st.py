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
from nn                                             import dbn,calcC,nn_split

import config
import data
import utils

import numpy             as np
import pandas            as pd
import multiprocessing   as mp
import seaborn           as sns
import matplotlib.pyplot as plt

import csv

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
        with open(fl) as f:
            dat  = np.asarray([line for line in csv.reader(f)])
            f.close()
            # going to capture the header and data so it can be replaced
            hdr = list(dat[0].copy())
            # move beyond the header and add it back later
            dat  = dat[1:]
            # columns to drop, including date columns .. add them back later
            dat         = pd.DataFrame(dat)
            dat.columns = hdr
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
                        wiki = wikilabel(inst,dat[indxr,i],True,True)
                        for k in indxr:
                            # when using the open source (local=True) method, the return
                            # should contain the numeric label and the topic, separated by const.constants.SEP
                            #
                            # we only want the integer label for now, with the rest being used later
                            dat[k,i] = wiki[indxr.index(k)].split(const.constants.SEP)[0]
                    else:
                        if type("") == utils.sif(dat[0,i]):
                            wiki = wikilabel(inst,dat[:,i],True,True)
                            for k in range(0,len(dat[:,i])):
                                # when using the open source (local=True) method, the return
                                # should contain the numeric label and the topic, separated by const.constants.SEP
                                #
                                # we only want the integer label for now, with the rest being used later
                                dat[k,i] = wiki[k].split(const.constants.SEP)[0]
                # fix the data by intelligently filling missing values
                dat  = fixdata(inst,dat,coln)
                # add the header and drop/date columns back to the data set
                if hasattr(const.constants,"DROP" )                                                                  and \
                   type(const.constants.DROP ) in [type([]),type(np.asarray([]))] and len(const.constants.DROP ) > 0:
                    dat  = np.hstack((drop,dat))
                if hasattr(const.constants,"DATES")                                                                  and \
                   type(const.constants.DATES) in [type([]),type(np.asarray([]))] and len(const.constants.DATES) > 0:
                    dat  = np.hstack((dts ,dat))
                dat  = pd.DataFrame(dat,columns=hdr)
            # take the data and split it into a train and test set
            split= nn_split(dat)
            train= np.asarray(split["train"]).astype(np.single)
            test = np.asarray(split["test" ]).astype(np.single)
            # create a knowledge graph for the modeling using instance 1
            # (requires addition of an object in the json config file)
            # so we don't overwrite the knowledge graph used when fixing the data set
            inst = 1
            kgdat= create_kg(inst,dat=train,limit=True)
            # instantiate a JanusGraph object
            graph= Graph()
            # connection to the remote server
            conn = DriverRemoteConnection(data.url_kg(inst),'g')
            # get the remote graph traversal
            g    = graph.traversal().withRemote(conn)
            # write the knowledge graph
            dump = [data.write_kg(const.constants.V,inst,list(coln.items()),k,g,False) for k in kgdat]
            dump = [data.write_kg(const.constants.E,inst,list(coln.items()),k,g,True ) for k in kgdat]
            # with the trained models in instance 1, we will predict the test data
            # one column at a time, each as a function of the remaining columns
            # then we will concatenate all of the predicted columns together and
            # use seaborn's pairplot to see the plots for each column in a table
            #
            # first build the data outputs
            preds= []
            for col in range(0,len(test[0])):
                cls  = [col]
                cls.extend([j for j in range(0,len(test[0])) if j not in cls])
                mdl  = "models/" + const.constants.SEP.join([s for s in map(str,cls)]) + ".h5"
                model= load_model(mdl)
                pred = model.predict(test[:,cls[1:]])
                preds= np.hstack((preds,pred)) if not len(preds) == 0 else pred
            # we need a data frame for the paired plots
            df   = pd.DataFrame(preds,columns=hdr)
            # get the paired plots and save them
            sns.pairplot(df)
            # save the plot just created
            plt.save("images/st.png")
