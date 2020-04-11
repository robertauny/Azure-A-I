#!/usr/bin/python

############################################################################
##
## File:      ocr.py
##
## Purpose:   Optical character recognition on a set of PDFs
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Dec. 20, 2019
##
############################################################################

from services import cognitive

import constants as const

############################################################################
##
## Purpose:   Extraction of characters from PDFs
##
############################################################################
def cognitive_testing(pdfs=["/home/robert/data/files/kg.pdf"],inst=0,testing=False):
    print(cognitive(const.OCR,pdfs,inst,testing))
    print(cognitive(const.EE ,pdfs,inst,testing))
