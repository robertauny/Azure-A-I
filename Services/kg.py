#!/usr/bin/python

############################################################################
##
## File:      kg.py
##
## Purpose:   Knowledge graph encoding with a deep belief network
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Dec. 20, 2019
##
############################################################################

from services import kg
from data     import read_kg
from ai       import create_kg,thought

import constants as const

import numpy     as np

############################################################################
##
## Purpose:   Generate a knowledge graph
##
############################################################################
def kg_testing(inst=0,M=100,N=5,testing=False):
    # number of data points and properties
    m    = M
    p    = N
    if p > const.MAX_FEATURES:
        p    = const.MAX_FEATURES
    # define the number of splits of each property
    s    = p
    # uniformly sample values between 0 and 1 as the data set
    dat  = np.random.sample(size=(m,p))
    # create column names (normally obtained by var.dtype.names)
    #
    # use an explicit dict to make sure that the order is preserved
    coln = [("col"+str(i),(i-1)) for i in range(1,len(dat[0])+1)]
    # create the data for the sample knowledge graph
    kgdat= create_kg(inst,dat,s)
    # populate the knowledge graph into the remote DB
    #
    # force an ordering on the columns always
    print(kg(inst,np.asarray(coln)[:,0],kgdat,testing))
    # read the knowledge graph
    # shuffle the ordering of coln (keys/values) to read other data
    print(read_kg(inst,coln))
    # test the thought function with the default number of predictions 3
    print(thought(inst,coln))
