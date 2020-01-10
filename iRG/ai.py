#!/usr/bin/python

############################################################################
##
## File:      ai.py
##
## Purpose:   Other A-I and machine learning functions needed for iRG.
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Dec. 28, 2019
##
############################################################################

from joblib       import Parallel, delayed
from itertools    import combinations
from keras.utils  import to_categorical

from pdf2image    import convert_from_bytes,convert_from_path
from PIL          import Image

import requests
import io

import numpy as np

import multiprocessing as mp
import os

import config

from nn import dbn

BVAL = -1

############################################################################
##
## Purpose:   Identify the cluster for the data row
##
############################################################################
def store(dat=[]):
    ret  = []
    sz   = len(dat)
    if not (sz == 0):
        # which cluster has been identified for storing the data
        tdat = [j for j, x in enumerate(dat) if x == max(dat)]
        ret  = tdat[0]
    return ret

############################################################################
##
## Purpose:   Cluster number
##
############################################################################
def prefix(i=BVAL):
    ret  = BVAL
    if not (i <= BVAL):
        ret  = i + 1
    return ret

############################################################################
##
## Purpose:   Cluster label and row number appended together
##
############################################################################
def append(i=BVAL,n=BVAL,m=BVAL):
    ret  = None
    if not (i <= BVAL or n <= BVAL or m <= BVAL):
        n   += 1 # note that this modifies the passed in value for all to see
        m   += (i+1) # note that this modifies the passed in value for all to see
        ret  = str(n) + "-" + str(m)
        #ret  = str(n+1) + "-" + str(m)
    return ret

############################################################################
##
## Purpose:   Unique label for each row of data with cluster number and row number
##
############################################################################
def label(dat=[]):
    ret  = []
    # number of data points in all clusters to label
    sz   = len(dat)
    if not (sz == 0):
        num_cores = mp.cpu_count()
        # which cluster has been identified for storing the data
        ret  = Parallel(n_jobs=num_cores)(delayed(store)(dat[i]) for i in range(0,sz))
        # initialize a 2-D array for the counts of elements in a cluster
        tret = np.zeros((len(dat[0]),2),dtype=int)
        # set the cluster label prefix from 1 to length of dat[0]
        # which should be a binary word with exactly one 1 and a
        # number of zeros ... total length matches number of clusters
        tret[:,0] = Parallel(n_jobs=num_cores)(delayed(prefix)(i) for i in range(0,len(dat[0])))
        # append a unique number to the cluster label to identify
        # each unique point in the cluster
        #
        # also keeping track of the original data point ordering, marking which data points
        # went into which cluster ... this is done in the append function by adding (i+1) to tret
        ret  = Parallel(n_jobs=num_cores)(delayed(append)(i,ret[i],tret[ret[i]][1]) for i in range(0,len(ret)))
    return ret

############################################################################
##
## Purpose:   Unique list of labels for hierarchies depending upon features used
##
############################################################################
def unique(dat):
    ret  = []
    sz   = len(dat)
    if not (sz == 0):
        for i in range(0,sz):
            rsz  = len(ret)
            if rsz == 0:
                ret.append(dat[i])
            else:
                for j in range(0,rsz):
                    if ret[j] == dat[i]:
                        break
                    else:
                        if j == rsz - 1:
                            if not (ret[j] == dat[i]):
                                ret.append(dat[i])
    return ret

############################################################################
##
## Purpose:   Permutations of a list of integers for use in labeling hierarchies
##
############################################################################
def permute(dat):
    ret  = []
    sz   = len(dat)
    if not (sz == 0):
        # permute the array of indices beginning with the first element
        for j in range(0,sz+1):
            # all permutations of the array of indices
            jdat = dat[j:] + dat[:j]
            for i in range(0,sz):
                # only retain the sub arrays that are length >= 2
                tmp = [list(x) for x in combinations(jdat,i+2)]
                if len(tmp) > 0:
                    ret.extend(tmp)
    return unique(ret)

############################################################################
##
## Purpose:   Knowledge brain of all data hierarchies and neural networks
##
############################################################################
def brain(dat=[],splits=2):
    ret  = []
    # number of data points in all clusters to label
    sz   = len(dat)
    if not (sz == 0 or splits <= 0):
        # get all permutations of the 
        perms= permute(range(0,len(dat[0])))
        # add all models to the return
        for perm in perms:
            # model label
            lbl  = '-'.join(map(str,perm))
            # number of data points, properties and splits
            m    = len(dat)
            p    = len(perm)
            # we need values to turn into labels when training
            # one-hot encode the integer labels as its required for the softmax
            #
            # the outputs force the hierarchical classification model to 
            # produce regressors to segregate the data that amount to
            # projections onto the last feature of the data, as we are only using
            # the last of the input features in the definition of the outputs
            odat = to_categorical(dat[:,len(perm)-1],num_classes=splits**(2*p))
            # generate the model
            mdl  = dbn(dat[:,perm],odat,splits=splits,props=p)
            # save the model to a local file
            fl   = "models/" + lbl + ".h5"
            mdl.save(fl)
            # add the current label and model to the return
            ret.append({"label":lbl,"model":fl})
    return ret

############################################################################
##
## Purpose:   Correct model for predicting future data as a service
##
############################################################################
def thought(inst=BVAL,lbl=None):
    ret  = []
    if not (inst <= BVAL or lbl == None):
        # assume that models are built so we will search the models
        # in the DB and return the one corresponding to the label 
        #
        # neural networks obtained from the DB later ... None for now
        df   = rw_kg(inst)
        nns  = df["nns"]
        for nn in nns:
            if lbl == nn["label"]:
                # load the model from the file path that is returned
                ret  = nn["model"]
                break
    return ret

############################################################################
##
## Purpose:   Convert Python image library (PIL) image to an array
##
############################################################################
def pil2array(pil=None):
    ret  = None
    if not (pil == None):
        # get addressable memory for writing the image
        imgb = io.BytesIO()
        # save the passed image to the addressable memory
        pil.save(imgb,format='PNG')
        # get a reference to the addressable memory for the return
        ret  = imgb.getvalue()
    return ret

############################################################################
##
## Purpose:   Convert Python image library (PIL) image data to text
##
############################################################################
def img2txt(img=[],inst=BVAL,testing=True):
    ret  = []
    if not (len(img) == 0 or inst <= BVAL):
        # get the default configuration
        cfg  = config.cfg()
        # ordering of the data elements in the JSON file
        src  = cfg["instances"][inst]["src"]["index"]
        typ  = cfg["instances"][inst]["src"]["types"]["ocr"]
        # azure subscription key
        key  = cfg["instances"][inst]["sources"][src][typ]["connection"]["key"]
        # azure vision api
        host = cfg["instances"][inst]["sources"][src][typ]["connection"]["host"]
        # directory
        drct = cfg["instances"][inst]["sources"][src][typ]["connection"]["dir"]
        # version
        ver  = cfg["instances"][inst]["sources"][src][typ]["connection"]["ver"]
        # app
        app  = cfg["instances"][inst]["sources"][src][typ]["connection"]["app"]
        # url
        url  = "https://" + host + "/" + drct + "/" + ver + "/" + app
        # request headers. Important: content should be bytestream as we are sending an image from local
        hdrs = {"Ocp-Apim-Subscription-Key":key,"Content-Type":"application/octet-stream"}
        # request parameters: language is unknown, and we do detect orientation
        parms= {"language":"unk","detectOrientation":"true"}
        if not testing:
            # get response from the server
            resp = requests.post(url,headers=hdrs,params=parms,data=img)
            resp.raise_for_status()
            # get json data to parse it later
            json = resp.json()
            # all the line from a page, including noise
            ftext= []
            for reg in json["regions"]:
                line = reg["lines"]
                for elem in line:
                    ltext = " ".join([word["text"] for word in elem["words"]])
                    ftext.append(ltext.lower())
            # clean array containing only important data
            for line in ftext:
                ret.append(line)
        else:
            ret  = [src,typ,key,host,url,hdrs,parms]
    return ret

# *************** TESTING *****************

def ai_testing(M=500,N=2):
    # uniformly sample values between 0 and 1
    #ivals= np.random.sample(size=(500,3))
    ivals= np.random.sample(size=(M,N))
    print(permute(range(0,len(ivals[0]))))
    print(brain(ivals))
    imgs = convert_from_path("files/kg.pdf")
    print(imgs)
    for img in imgs:
        print(pil2array(img))
    src,typ,key,host,url,hdrs,parms = img2txt(imgs,0)
    print(src)
    print(typ)
    print(key)
    print(host)
    print(url)
    print(hdrs)
    print(parms)
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
    model = dbn(ivals,ovals,splits=s,props=p)
    if not (model == None):
        # generate some test data for predicting using the model
        ovals= np.random.sample(size=(m/10,p))
        # encode and decode values
        pvals= model.predict(ovals)
        # look at the original values and the predicted values
        print(ovals)
        print(pvals)
        print(label(pvals))
    else:
        print("Label model is null.")
