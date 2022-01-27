#!/usr/bin/python

############################################################################
##
## File:      ee.py
##
## Purpose:   Entity extraction from a set of documents
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Nov. 22, 2021
##
############################################################################

from services import ee

import constants as const

############################################################################
##
## Purpose:   Entity extraction from a set of documents
##
############################################################################
def ee_testing(docs=["README.txt"],inst=0,testing=False):
    ret  = ee(const.constants.EE,docs,inst,testing)
    return ret
