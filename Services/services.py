#!/usr/bin/python

############################################################################
# Begin license text.
# Copyright Feb. 27, 2020 Robert A. Murphy

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# End license text.
############################################################################

############################################################################
##
## File:      services.py
##
## Purpose:   Multiple services
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Nov. 22, 2021
##
############################################################################

import csv
import sys

import multiprocessing as mp
import numpy           as np

import constants       as const

ver  = sys.version.split()[0]

if ver == const.constants.VER:
    from            keras.models                            import Sequential,load_model,Model
    from            keras.utils.np_utils                    import to_categorical
else:
    from tensorflow.keras.models                            import Sequential,load_model,Model
    from tensorflow.keras.utils                             import to_categorical

from joblib                                         import Parallel, delayed

from pdf2image                                      import convert_from_bytes,convert_from_path
from PIL                                            import Image

# gremlin imports
from gremlin_python.structure.graph                 import Graph
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection

sys.path.append('/home/robert/code/scripts/python/services')

import ai
import data
import config

# get the default configuration
cfg  = config.cfg()

np.random.seed(const.constants.SEED)

############################################################################
##
## Purpose:   For each column in the passed data set, attempt to correct errors.
##
############################################################################
def corr(fl="/home/robert/data/food-inspections.csv",samp=100,cols=[4,5]):
    ret  = None
    with open(fl) as f:
        dat  = [row for row in csv.reader(f)]
        sz   = len(dat)
        if not (sz == 0 or samp <= 0):
            if (samp < sz):
                rnd  = np.random.randint(low=0,high=sz-1,size=samp)
                dat  = np.asarray([dat[r] for r in rnd])
            else:
                dat  = np.asarray(dat)
            if not (len(cols) == 0):
                dat  = dat[:,cols]
            # number of cpu cores
            nc   = mp.cpu_count()
            # perform the correction
            ret  = Parallel(n_jobs=nc)(delayed(ai.correction)(dat[:,i]) for i in range(0,len(dat[0])))
        f.close()
    # the corrections will be collected as rows so we need to return the transpose
    return np.asarray(ret).transpose()

############################################################################
##
## Purpose:   Attempt to identify security vulnerabilities in software code
##
############################################################################
def cs(docs=["/home/robert/data/java/test/test.java"]
      ,words=20
      ,ngrams=3
      ,splits=2
      ,props=2):
    # use custom glove, random fields and deep belief networks for the modeling
    return ai.cyberglove(docs,words,ngrams,splits,props)

############################################################################
##
## Purpose:   Perform optical character recognition on a set of PDFs
##
############################################################################
def cognitive(wtyp=const.constants.OCR,pdfs=["/home/robert/data/files/kg.pdf"],inst=0,testing=True):
    return ai.cognitive(wtyp,pdfs,inst,False,testing)

############################################################################
##
## Purpose:   Entity extraction on a set of text documents
##
############################################################################
def ee(wtyp=const.constants.EE,docs=["README.txt"],inst=0,testing=True):
    return ai.img2txt(wtyp,docs,inst,testing)

############################################################################
##
## Purpose:   Summarization on a set of text documents
##
############################################################################
def kp(wtyp=const.constants.SENT,docs=["README.txt"],inst=0,testing=True):
    return ai.img2txt(wtyp,docs,inst,testing)

############################################################################
##
## Purpose:   Sentiment on a set of text documents
##
############################################################################
def sent(wtyp=const.constants.SENT,docs=["README.txt"],inst=0,testing=True):
    return ai.img2txt(wtyp,docs,inst,testing)

############################################################################
##
## Purpose:   Write a knowledge graph
##
############################################################################
def kg(stem=None,inst=const.constants.BVAL,coln=[],kgdat=[],g=None,drop=True,testing=True):
    ret  = []
    lcol = len(coln)
    lkg  = len(kgdat)
    if not (stem == None or inst <= const.constants.BVAL or lcol == 0 or lkg == 0 or g == None):
        # create the url
        url  = data.url_kg(inst)
        if not testing:
            try:
                # write the graph to memory
                if stem in [const.constants.V,const.constants.E]:
                    ret  = [data.write_kg(stem,inst,coln,k,g,drop) for k in kgdat]
                else:
                    if stem == const.constants.ENTS:
                        # single row processing in the function call
                        ret  = data.write_kg(stem,inst,coln,kgdat,g,drop)
                    else:
                        if stem == const.constants.CONS:
                            # single row processing in the function call
                            ret  = data.write_kg(stem,inst,coln,kgdat,g,drop)
                        else:
                            # just a placeholder for the moment
                            # call the appropriate function in the future
                            # then append things to the graph using property tags
                            ret  = None
            except Exception as err:
                ret  = [str(err)]
        else:
            ret  = [url]
    return ret

############################################################################
##
## Purpose:   Extract any image text along with other characteristics
##
############################################################################
def images(imgs=["/home/robert/data/files/IMG_0569.jpeg","/home/robert/data/files/IMG_0570.jpg"],inst=0,objd=True,testing=True):
    ret  = []
    limg = len(imgs)
    if not (limg == 0 or inst <= const.constants.BVAL):
        if not testing:
            ret  = ai.cognitive(const.constants.IMG,imgs,inst,objd,testing)
        else:
            ret  = [imgs,inst,testing]
    return ret
