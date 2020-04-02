#!/usr/bin/python

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
## Date:      Dec. 20, 2019
##
############################################################################

from keras.utils                                    import to_categorical
from keras.models                                   import load_model
from joblib                                         import Parallel, delayed

from pdf2image                                      import convert_from_bytes,convert_from_path
from PIL                                            import Image

# gremlin imports
from gremlin_python.structure.graph                 import Graph
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection

import csv
import sys

import multiprocessing as mp
import numpy           as np

sys.path.append('/home/robert/code/scripts/python/services')

import constants       as const

import ai
import data
import config

# get the default configuration
cfg  = config.cfg()

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
## Purpose:   For each column in the passed data set, attempt to correct errors.
##
############################################################################
def ocr(pdfs=["/home/robert/data/files/kg.pdf"],inst=0,testing=True):
    return ai.ocr(pdfs,inst,testing)

############################################################################
##
## Purpose:   Write a knowledge graph
##
############################################################################
def kg(inst=const.BVAL,coln=[],kgdat=[],testing=True):
    ret  = []
    lcol = len(coln)
    lkg  = len(kgdat)
    if not (inst <= const.BVAL or lcol == 0 or lkg == 0):
        # ordering of the data elements in the JSON file
        src  = cfg["instances"][inst]["kg"]
        # subscription key
        key  = cfg["instances"][inst]["sources"][src]["connection"]["key" ]
        # graph host
        host = cfg["instances"][inst]["sources"][src]["connection"]["host"]
        # graph port
        port = cfg["instances"][inst]["sources"][src]["connection"]["port"]
        # api
        api  = cfg["instances"][inst]["sources"][src]["connection"]["api" ]
        # api
        ext  = cfg["instances"][inst]["sources"][src]["connection"]["ext" ]
        # set the url
        if not (key == None):
            if not (len(key) == 0):
                # protocol
                prot = "wss://"
            else:
                # protocol
                prot = "ws://"
        else:
            # protocol
            prot = "ws://"
        # create the url
        url  = prot + host + ":" + port + "/" + api
        # set the output file extension
        if not (ext == None):
            if len(ext) == 0:
                ext  = const.EXT
        else:
            ext  = const.EXT
        if not testing:
            try:
                ret  = []
                for k in kgdat:
                    # instantiate a JanusGraph object
                    graph= Graph()
                    # connection to the remote server
                    conn = DriverRemoteConnection(url,'g')
                    # get the remote graph traversal
                    g    = graph.traversal().withRemote(conn)
                    # write the graph to memory
                    ret.append(data.write_kg(coln,k,g))
                    # write the graph to disk
                    ids  = np.asarray(k[const.V][0][0].split("-"))
                    ids  = "-".join(ids[range(0,len(ids)-1)])
                    g.io("data/"+ids+ext).write().iterate()
                    # close the connection
                    conn.close()
            except Exception as err:
                ret  = str(err)
        else:
            ret  = [src,typ,key,host,url]
    return ret
