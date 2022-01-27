#!/usr/bin/python

############################################################################
##
## File:      sent.py
##
## Purpose:   Sentiment analysis on a set of text documents.
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Nov. 22, 2021
##
############################################################################

from services import cognitive

import constants as const

############################################################################
##
## Purpose:   Sentiment analysis on a set of text documents.
##
############################################################################
def sent_testing(docs=["README.txt"],inst=0,testing=False):
    ret  = cognitive(const.constants.SENT,docs,inst,testing)
    return ret
