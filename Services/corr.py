############################################################################
# Begin license text.
# Copyright 2020 Robert A. Murphy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# End license text.
############################################################################

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
