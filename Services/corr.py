#!/usr/bin/python

############################################################################
##
## File:      corr.py
##
## Purpose:   Data corrections service
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Dec. 20, 2019
##
############################################################################

import csv 
import os 

import numpy as np

from services import corr

############################################################################
##
## Purpose:   For each column in the passed data set, attempt to correct errors.
##
############################################################################
def corr_testing(fl="/home/robert/data/food-inspections.csv",samp=100,cols=[4,5]):
    ret  = []
    with open(fl) as f:
        if not (len(cols) == 0):
            dat  = [row for row in csv.reader(f,delimiter=",")]
            f.close()
            ld   = len(dat)
            if (ld < samp):
                rnd  = range(0,ld)
            else:
                rnd  = np.random.randint(low=0,high=ld-1,size=samp)
            dat  = np.asarray([dat[r] for r in rnd])
            dat  = dat[:,cols]
            tfl  = "/tmp/food-inspections.csv"
            with open(tfl,"wb") as f:
                w    = csv.writer(f,delimiter=",")
                w.writerows(dat)
                f.close()
                ld   = len(dat[0])
                rret = corr(tfl,len(rnd),range(0,ld))
                os.remove(tfl)
                for i in range(0,ld):
                    ret.append( dat[:,i])
                    ret.append(rret[:,i])
                ret  = np.asarray(ret).transpose()
                with open(tfl,"wb") as f:
                    w    = csv.writer(f,delimiter=",")
                    w.writerows(ret)
                    f.close()
    return ret
