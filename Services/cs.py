#!/usr/bin/python

############################################################################
##
## File:      cs.py
##
## Purpose:   Code security service
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Dec. 20, 2019
##
############################################################################

from services import cs

############################################################################
##
## Purpose:   Attempt to predict code security issues of concern
##
############################################################################
def cs_testing(docs=["/home/robert/data/java/test/test.java"]
              ,words=40
              ,ngrams=3
              ,splits=2
              ,props=2):
    print(cs(docs,words,ngrams,splits,props))
