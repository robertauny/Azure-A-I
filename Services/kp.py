#!/usr/bin/python

############################################################################
##
## File:      kp.py
##
## Purpose:   Text summarization on a set of documents
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Nov. 22, 2021
##
############################################################################

from services import kp

import constants as const

############################################################################
##
## Purpose:   Text summarization on a set of documents
##
############################################################################
def kp_testing(docs=["README.txt"],inst=0,testing=False):
    ret  = kp(const.constants.KP,docs,inst,testing)
    return ret
