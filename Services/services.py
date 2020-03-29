#!/usr/bin/python

############################################################################
##
## File:      services.py
##
## Purpose:   Multiple services
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

import csv
import sys

import multiprocessing as mp
import numpy           as np

sys.path.append('/home/robert/code/scripts/python/services')

import constants       as const

import ai

############################################################################
##
## Purpose:   For each column in the passed data set, attempt to correct errors.
##
############################################################################
def corr(fl="/home/robert/data/food-inspections.csv",samp=100,cols=[4,5]):
    ret  = None
    with open(fl) as f:
        dat  = [row for row in csv.reader(f)]
        sz   = len(dat)
        if not (sz == 0 or samp <= 0):
            if (samp < sz):
                rnd  = np.random.randint(low=0,high=sz-1,size=samp)
                dat  = np.asarray([dat[r] for r in rnd])
            else:
                dat  = np.asarray(dat)
            if not (len(cols) == 0):
                dat  = dat[:,cols]
            # number of cpu cores
            nc   = mp.cpu_count()
            # perform the correction
            ret  = Parallel(n_jobs=nc)(delayed(ai.correction)(dat[:,i]) for i in range(0,len(dat[0])))
        f.close()
    # the corrections will be collected as rows so we need to return the transpose
    return np.asarray(ret).transpose()

############################################################################
##
## Purpose:   Attempt to identify security vulnerabilities in software code
##
############################################################################
def cs(docs=["/home/robert/data/java/test/test.java"]
      ,words=20
      ,ngrams=3
      ,splits=2
      ,props=2):
    # use custom glove, random fields and deep belief networks for the modeling
    return ai.cyberglove(docs,words,ngrams,splits,props)

############################################################################
##
## Purpose:   For each column in the passed data set, attempt to correct errors.
##
############################################################################
def kg(M=500,N=2,EV=const.E):
    # number of data points and properties
    m    = M
    p    = N
    if p > const.MAX_FEATURES:
        p    = const.MAX_FEATURES
    # define the number of splits of each property
    s    = p
    # uniformly sample values between 0 and 1
    ivals= np.random.sample(size=(m,p))
    # create the data for the sample knowledge graph
    ret  = ai.create_kg(0,ivals,s)
    # return the edges ... use the key const.V to return the vertices
    return ret[EV]

############################################################################
##
## Purpose:   For each column in the passed data set, attempt to correct errors.
##
############################################################################
def ocr(pdfs=["/home/robert/data/files/kg.pdf"],inst=0,testing=True):
    return ai.ocr(pdfs,inst,testing)