#!/usr/bin/python

############################################################################
##
## File:      images.py
##
## Purpose:   Read text from images
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      May 06, 2020
##
############################################################################

from data                                           import url_kg,read_kg,sodaget
from services                                       import images
from ai                                             import create_kg,extendglove,thought,cognitive

import constants as const

import config

import numpy     as np

# gremlin imports
from gremlin_python.structure.graph                 import Graph
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection

cfg  = config.cfg()

np.random.seed(12345)

############################################################################
##
## Purpose:   Process the image data
##
############################################################################
def images_testing(inst=0,objd=True,lim=0,train=False,testing=False):
    # getting constants from the JSON config file
    src  = cfg["instances"][inst]["src"]["index"]
    typ  = cfg["instances"][inst]["src"]["types"]["ocri"]
    imgs = cfg["instances"][inst]["sources"][src][typ]["connection"]["files"]
    # get the ocr data ... using the real way to get the ocr data here
    #
    # first parameter is a list of images
    # second parameter is an integer instance of this code base that is being run
    # third parameter is a boolean value indicating whether (or not) we are testing
    cdat = images(imgs,inst,objd,testing)
    print(cdat)
    # perform the query against the NIH database with limit = 100 rows
    #sg   = sodaget(inst,cdat,100)
    sg   = sodaget(inst,cdat,objd,lim,train)
    print(sg)
