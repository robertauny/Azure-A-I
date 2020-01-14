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
        preds= model.predict(dat)
        # generate the labels for the data
        lbls = ai.label(preds)
        # create the vertices
        v    = create_kg_ve(dat,lbls,lbl,"v")
        # create the edges
        e    = create_kg_ve(dat,lbls,lbl,"e")
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

############################################################################
##
## Purpose:   Convert a list of strings into their ordinal representation
##
############################################################################
def chars(dat=[],pre=0):
    ret  = []
    sz   = len(dat)
    if not (sz == 0 or pre < 0):
        e    = dat.rjust(pre)
        ret.extend(e)
    return ret

############################################################################
##
## Purpose:   Convert a list of strings into their ordinal representation
##
############################################################################
def numbers(dat=[],pre=0):
    ret  = None
    sz   = len(dat)
    if not (sz == 0 or pre < 0):
        d    = chars(dat,pre)
        ret  = [ord(x) for i, x in enumerate(d)]
    return ret

############################################################################
##
## Purpose:   Which string in a list appears most (by whole words)
##
############################################################################
def almost(dat=[]):
    ret  = None
    sz   = len(dat)
    if not (sz == 0):
        ret  = max(set(dat),key=dat.count)
    return ret

############################################################################
##
## Purpose:   Which string in a list appears most (by character)
##
############################################################################
def most(dat=[],pre=0):
    ret  = None
    sz   = len(dat)
    if not (sz == 0 or pre < 0):
        # number of cpu cores
        nc   = mp.cpu_count()
        # character-wise append of most frequent characters in each feature of the list of strings
        cdat = Parallel(n_jobs=nc)(delayed(chars)(dat[i],pre) for i in range(0,sz))
        cdat = np.asarray(cdat)
        mdat = Parallel(n_jobs=nc)(delayed(max)(set(cdat[:,i].tolist()),key=cdat[:,i].tolist().count) for i in range(0,pre))
        ret  = "".join(mdat).lstrip()
        if not (ret in dat):
            ret  = almost(dat)
    return ret

############################################################################
##
## Purpose:   Use DBN to correct a list of corrupt or misclassified strings
##
############################################################################
def correction(dat=[]):
    ret  = []
    sz   = len(dat)
    if not (sz == 0):
        # number of cpu cores
        nc   = mp.cpu_count()
        # length of longest string in the list
        ssz  = max([len(x) for i, x in enumerate(dat)])
        # numeric representations of the strings in the list
        ndat = Parallel(n_jobs=nc)(delayed(numbers)(dat[i],ssz) for i in range(0,sz))
        # we need values to turn into labels when training
        # one-hot encode the numeric data as its required for the softmax
        #
        # properties
        p    = len(ndat[0])
        #
        # splits
        s    = 2
        #
        # number of classes, one-hot encodings
        ncls = s**(2*p)
        #
        # compute the values
        #
        # the outputs force the hierarchical classification model to 
        # produce regressors to segregate the data that amount to
        # auto-encoders, as we are using all of the input data elements
        # in the definition of the outputs
        #tdat = [x[len(x)-1]/max(x) for i, x in enumerate(ndat)]
        tdat = [sum(x)/(len(x)*max(x)) for i, x in enumerate(ndat)]
        odat = to_categorical(tdat,num_classes=ncls)
        ndat = np.asarray(ndat)
        # generate the model of the data
        model= nn.dbn(ndat,odat,splits=s,props=p)
        # predict the same data to get the labels
        preds= model.predict(ndat)
        # get the labels
        lbls = ai.label(preds)
        # split the labels to know the available clusters
        slbl = Parallel(n_jobs=nc)(delayed(lbls[i].split)("-") for i in range(0,len(lbls)))
        slbl = np.asarray(slbl)
        # cluster labels
        clus = slbl[:,0]
        ucls = np.unique(clus)
        # row numbers
        rows = slbl[:,1]
        # collect all data for each cluster and assign most numerously appearing value
        ret  = np.empty(sz,dtype=object)
        for i in range(0,len(ucls)):
            ind      = [j for j, x in enumerate(clus) if x == ucls[i]]
            idat     = [dat[x] for j, x in enumerate(ind)]
            mdat     = most(idat,ssz)
            ret[ind] = np.full(len(ind),mdat)
    return ret

############################################################################
##
## Purpose:  Read data from an array of PDF files
##
############################################################################
def ocre(imgs=[]):
    ret  = None
    if not (len(imgs) == 0):
        # number of cpu cores
        nc   = mp.cpu_count()
        # converted images
        ret  = Parallel(n_jobs=nc)(delayed(ai.pil2array)(imgs[i]) for i in range(0,len(imgs)))
    return ret

############################################################################
##
## Purpose:  Read data from an array of PDF files
##
############################################################################
def ocr(pdfs=[],inst=ai.BVAL,testing=True):
    ret  = None
    if not (len(pdfs) == 0 or inst <= ai.BVAL):
        # number of cpu cores
        nc   = mp.cpu_count()
        # converted images
        imgs =     Parallel(n_jobs=nc)(delayed(convert_from_path)( pdfs[i]             ) for i in range(0,len(pdfs )))
        pimgs=     Parallel(n_jobs=nc)(delayed(ocre             )( imgs[i]             ) for i in range(0,len(imgs )))
        oimgs=     Parallel(n_jobs=nc)(delayed(ai.img2txt       )(pimgs[i],inst,testing) for i in range(0,len(pimgs)))
        if not (len(oimgs) <= 1):
            ret  = Parallel(n_jobs=nc)(delayed(oimgs[0].append  )(oimgs[i]             ) for i in range(1,len(oimgs)))
        else:
            ret  = oimgs
    return ret

# *************** TESTING *****************

def irg_testing(M=500,N=2):
    # uniformly sample values between 0 and 1
    #ivals= np.random.sample(size=(500,3))
    ivals= np.random.sample(size=(M,N))
    # create the data for the sample knowledge graph
    kg   = create_kg(0,ivals)
    print(kg["edges"])
    # test ocr
    o    = ocr(["files/kg.pdf"],0)
    print(o)
    # number of data points, properties and splits
    m    = np.size(ivals,0)
    p    = np.size(ivals,1)
    #s    = p + 1
    s    = p
    # we need values to turn into labels when training
    # one-hot encode the integer labels as its required for the softmax
    nc   = s**(2*p)
    ovals= []
    for i in range(0,M):
        ovals.append(np.random.randint(1,nc))
    ovals= to_categorical(ovals,num_classes=nc)
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
    # test the data correction neural network function
    # output should be "robert", overwriting corrupt data and misclassifications
    bdat = ['robert','robert','robert','r0bert','rob3rt','r0b3rt','andre','murphy','murphy']
    corr = correction(bdat)
    print(corr)
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
