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
## File:      kg.py
##
## Purpose:   Knowledge graph encoding with a deep belief network
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Nov. 22, 2021
##
############################################################################

from data                                           import url_kg,read_kg
from services                                       import kg
from ai                                             import create_kg,extendglove,thought,cognitive
from ocr                                            import ocr_testing

import constants as const

import config

import numpy     as np

# gremlin imports
from gremlin_python.structure.graph                 import Graph
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection

cfg  = config.cfg()

np.random.seed(const.constants.SEED)

############################################################################
##
## Purpose:   Generate a knowledge graph
##
############################################################################
def kg_testing(inst=1,M=10,N=5,testing=False):
    # number of data points and properties
    m    = M
    p    = N
    if p > const.constants.MAX_FEATURES:
        p    = const.constants.MAX_FEATURES
    # define the number of splits of each property
    s    = p if p <= const.constants.MAX_SPLITS else const.constants.MAX_SPLITS
    # uniformly sample values between 0 and 1 as the data set
    dat  = np.random.sample(size=(m,p))
    # create column names (normally obtained by var.dtype.names)
    #
    # use an explicit dict to make sure that the order is preserved
    coln = [("col"+str(i),(i-1)) for i in range(1,p+1)]
    # create the data for the sample knowledge graph (only one brain)
    kgdat= create_kg(inst,dat,s,[[int(i) for i in np.asarray(coln)[:,1]]])
    # populate the knowledge graph into the remote DB
    #
    # instantiate a JanusGraph object
    graph= Graph()
    # connection to the remote server
    conn = DriverRemoteConnection(url_kg(inst),'g')
    # get the remote graph traversal
    g    = graph.traversal().withRemote(conn)
    # we only want to process the right brain
    print(kg(const.constants.V,inst,coln,kgdat,g,False,testing))
    # after building the knowledge graph, use the output of ocr to test the GloVe write
    #
    # call cognitive to produce the ocr output
    oret = ocr_testing()
    # get the location of the glove file
    src  = cfg["instances"][inst]["src"]["index"]
    typ  = cfg["instances"][inst]["src"]["types"]["glove"]
    gfl  = cfg["instances"][inst]["sources"][src][typ]["connection"]["file"]
    # call extendglove to produce the GloVe output and transform it to an array
    # with the first term in each row being the key and all other terms are values
    rdat = extendglove(oret[0][0],gfl[0])
    rdat = [(k,v) for k,v in list(rdat.items())[0:M]]
    # write the glove output to the knowledge graph
    print(kg(const.constants.ENTS,inst,coln,rdat,g,False,testing))
    # get the ocr data ... using the real way to get the ocr data here
    typ  = cfg["instances"][inst]["src"]["types"]["ocrf"]
    pdfs = cfg["instances"][inst]["sources"][src][typ]["connection"]["files"]
    cdat = cognitive(const.constants.OCR,pdfs,inst,False,testing)
    # write the ocr data to the graph
    print(kg(const.constants.CONS,inst,coln,cdat[1:],g,True,testing))
    # close the connection
    conn.close()
    # test the thought function with the default number of predictions 3
    print(thought(inst,coln))
