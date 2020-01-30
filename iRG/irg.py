#!/usr/bin/python

############################################################################
##
## File:      irg.py
##
## Purpose:   Knowledge graph encoding with a deep belief network
##            followed by data being written to Cosmos (or other graph) DB.
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Dec. 20, 2019
##
############################################################################

from keras.utils  import to_categorical
from keras.models import load_model
from joblib       import Parallel, delayed

from pdf2image    import convert_from_bytes,convert_from_path
from PIL          import Image

import nn
import ai
import data
import constants as const

import multiprocessing as mp

import numpy  as np
import pandas as pd

############################################################################
##
## Purpose:   Extend an array
##
############################################################################
def extend(dat1=[],dat2=[]):
    ret  = []
    ret.append(dat1)
    ret.extend(dat2)
    return ret

############################################################################
##
## Purpose:   Extend an array
##
############################################################################
def extend1(dat1=[],dat2=[]):
    # number of cpu cores
    nc   = mp.cpu_count()
    ret  = Parallel(n_jobs=nc)(delayed(extend)(dat1,dat2[i]) for i in range(0,len(dat2)))
    return ret

############################################################################
##
## Purpose:   Split a string
##
############################################################################
def split(dat=[],ind=0):
    ret  = None
    if not (len(dat) == 0 or ind < 0 or ind > len(dat)-1):
        ret  = dat.split("-")[ind]
    return ret

############################################################################
##
## Purpose:   Define edges between vertices of a knowledge graph
##
############################################################################
def edges(clus=None,rows=[]):
    ret  = None
    if not (clus == None or len(rows) == 0):
        # number of cpu cores
        nc   = mp.cpu_count()
        # append the data point row numbers of those data points that are connected to the i-th data point in the current cluster
        #
        # have to sourround [x] so that extend doesn't separate the characters
        tret = Parallel(n_jobs=nc)(delayed(extend1)(rows[i],[[x] for j, x in enumerate(rows) if rows[j] != rows[i]]) for i in range(0,len(rows)))
        # append the cluster number
        ret  = Parallel(n_jobs=nc)(delayed(extend1)(clus,tret[i]) for i in range(0,len(tret)))
    return ret

############################################################################
##
## Purpose:   Create vertices and edges of a knowledge graph
##
############################################################################
def create_kg_ve(dat=[],lbls=[],lbl=None,ve=None):
    ret  = None
    if not (len(dat) == 0 or len(lbls) == 0 or lbl == None or ve == None):
        # number of cpu cores
        nc   = mp.cpu_count()
        if ve == "v":
            # only need to append the unique id defined by the row label to the data row
            # this is the set of vertices for each data point in the data set
            ret  = Parallel(n_jobs=nc)(delayed(extend)(lbl+'-'+lbls[i],dat[i]) for i in range(0,len(lbls)))
        else:
            # which cluster has been identified for storing the data
            clus = Parallel(n_jobs=nc)(delayed(split)(lbls[i]  ) for i in range(0,len(lbls)))
            ucs  = np.unique(clus)
            # get the row number of the original data point
            rows = Parallel(n_jobs=nc)(delayed(split)(lbls[i],1) for i in range(0,len(lbls)))
            # only need to extract the cluster label to go along with the brain label that was passed in
            # the edges will consist of the cluster label, brain label and connected data point pairs
            # this shows which data points are connected to form a cluster under the model in the current brain
            ret  = Parallel(n_jobs=nc)(delayed(edges)(ucs[i],[x for j, x in enumerate(rows) if clus[j] == ucs[i]]) for i in range(0,len(ucs)))
    return ret

############################################################################
##
## Purpose:   Heavy lift of creating a knowledge graph
##
############################################################################
def build_kg(inst,dat=[],brn={},splits=2):
    ret  = {"vertices":[],"edges":[]}
    if not (inst == None  or
            inst < 0      or
            len(dat) == 0 or
            brn == {}     or
            splits < 2):
        # get the nn model for this brain
        mdl  = brn["model"]
        # get the nn label for this brain
        lbl  = brn["label"]
        # make the predictions using this model
        model= load_model(mdl)
        # make sure to get the right subset of the data
        l    = list(map(int,lbl.split("-")))
        d    = dat[:,l]
        # make the predictions
        preds= model.predict(d)
        preds= to_categorical(np.sum(preds,axis=1),num_classes=splits**(2*len(l)))
        # generate the labels for the data
        lbls = ai.label(preds)
        # create the vertices
        v    = create_kg_ve(d,lbls,lbl,"v")
        # create the edges
        e    = create_kg_ve(d,lbls,lbl,"e")
        ret["vertices"] = v
        ret["edges"   ] = e
    return ret 

############################################################################
##
## Purpose:   Append vertices and edges to a knowledge graph
##
############################################################################
def append_kg(ret={},dat={}):
    v    = []
    e    = []
    if not (ret == {} or dat == {} or len(dat["vertices"]) == 0 or len(dat["edges"]) == 0):
        # vertices
        rv   = ret["vertices"]
        dv   = dat["vertices"]
        if not (len(rv) == 0):
            v    = extend(rv,dv)
        else:
            v    = dv
        # edges
        re   = ret["edges"   ]
        de   = dat["edges"   ]
        if not (len(re) == 0):
            e    = extend(re,de)
        else:
            e    = de
    return {"vertices":v,"edges":e}

############################################################################
##
## Purpose:   Create a knowledge graph
##
############################################################################
def create_kg(inst,dat=[],splits=2):
    ret  = {"vertices":[],"edges":[]}
    if not (inst == None or inst < 0 or len(dat) == 0 or splits < 2):
        # number of cpu cores
        nc   = mp.cpu_count()
        # generate the brains
        brns = ai.brain(dat)
        # generate the vertices and edges
        bret = Parallel(n_jobs=nc)(delayed(build_kg)(inst,dat,brn,splits) for brn in brns)
        rret = ret
        ret  = Parallel(n_jobs=nc)(delayed(append_kg)(rret,bret[i]) for i in range(0,len(bret)))
    return ret[0]

# *************** TESTING *****************

def irg_testing(M=500,N=2):
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
    # create the data for the sample knowledge graph
    kg   = create_kg(0,ivals,s)
    print(kg["edges"])
    # we need values to turn into labels when training
    # one-hot encode the integer labels as its required for the softmax
    ovals= nn.categoricals(M,s,p)
    # generate the model for using the test values for training
    model = nn.dbn(ivals,ovals,splits=s,props=p)
    if not (model == None):
        # generate some test data for predicting using the model
        ovals= np.random.sample(size=(m/10,p))
        # encode and decode values
        pvals= model.predict(ovals)
        # look at the original values and the predicted values
    else:
        print("iRG model is null.")
    # spark sqlContext should be used to create the data frame
    # of edges and vertices in the following format
    #
    # vertices
    #
    # unique id, column listing
    # unique id column name, column listing name
    #
    # df = sqlContext.createDataFrame( [ ("id1","val11","val12",...,"val1n"),
    #                                    ("id2","val21","val22",...,"val2n"),
    #                                    ...,
    #                                    ["id","col1","col2",...,"coln"] ] )
    #
    # edges
    #
    # entity 1, related to entity 2, relationship
    # source entity, destination entity, relationship heading
    #
    # df = sqlContext.createDataFrame( [ ("id1","id2","rel12"),
    #                                    ("id2","id3","rel23"),
    #                                    ...,
    #                                    ("idm","idn","relmn"),
    #                                    ...,
    #                                    ["src","dst","relationship"] ] )
    #
    # when making predictions for reports, gather the data by cluster label
    # then train the regression model using the data ... finally make predictions
    # using new data passed to the prediction method of the trained model
