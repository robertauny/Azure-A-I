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

import numpy     as np

import constants as const

import ai

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
    # create the data for the sample knowledge graph
    kgdat= ai.create_kg(inst,dat,s)
    # create column names (normally obtained by var.dtype.names)
    coln = {"col"+str(i):(i-1) for i in range(1,len(dat[0])+1)}
    # populate the knowledge graph into the remote DB
    print(kg(inst,coln.keys(),kgdat,testing))
    # read the knowledge graph
    print(read_kg(inst,coln))
