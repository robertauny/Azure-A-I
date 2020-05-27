#!/usr/bin/python

############################################################################
##
## File:      images.py
##
## Purpose:   Read text from images
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      May 06, 2020
##
############################################################################

from joblib                                         import Parallel,delayed

from data                                           import url_kg,read_kg,sodaget
from services                                       import images,kg
from ai                                             import create_kg,extendglove,thought,cognitive

import constants as const

import config

import numpy           as np
import multiprocessing as mp

# gremlin imports
from gremlin_python.structure.graph                 import Graph
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection

cfg  = config.cfg()

np.random.seed(12345)

############################################################################
##
## Purpose:   Process the image data
##
############################################################################
def images_testing(inst=0,objd=True,lim=0,train=False,testing=False):
    # instantiate a JanusGraph object
    graph= Graph()
    # connection to the remote server
    conn = DriverRemoteConnection(url_kg(inst),'g')
    # get the remote graph traversal
    g    = graph.traversal().withRemote(conn)
    # getting constants from the JSON config file
    src  = cfg["instances"][inst]["src"]["index"]
    typ  = cfg["instances"][inst]["src"]["types"]["ocri"]
    imgs = cfg["instances"][inst]["sources"][src][typ]["connection"]["files"]
    typ  = cfg["instances"][inst]["src"]["types"]["pill"]
    sel  = cfg["instances"][inst]["sources"][src][typ]["connection"]["sel"  ]
    # get the ocr data ... using the real way to get the ocr data here
    #
    # first parameter is a list of images
    # second parameter is an integer instance of this code base that is being run
    # third parameter is a boolean value indicating whether (or not) we are testing
    cdat = images(imgs,inst,objd,testing)
    # perform the query against the NIH database with limit = 100 rows
    sg   = sodaget(inst,cdat,objd,lim,train)
    # if we are training, then go ahead and get the wikipedia pages
    # for the returns using the script medication in a post
    if train:
        # image keys and values
        files= list(sg.keys  ())
        vals = list(sg.values())
        # number of cpu cores
        nc   = mp.cpu_count()
        # GloVe configurations
        typ  = cfg["instances"][inst]["src"]["types"]["glove"]
        gfl  = cfg["instances"][inst]["sources"][src][typ]["connection"]["file"]
        # for each image file, get all wikipedia files using the scripts returned
        rdat = None
        for fl in files:
            # all of the scripts
            scrs = sg[fl][sel["rxstring"]]
            if not (scrs == None):
                # remove all scripts that are None
                inds = [i for i,x in enumerate(scrs) if not (x == None)]
                if not (len(inds) == 0):
                    # capture the remaining scripts
                    scrs = list(np.asarray(scrs)[inds])
                    # extract the entities from the scripts for gathering the wikipedia pages
                    ents = cognitive(const.EE,scrs,inst,objd,testing)
                    # each ents return is an array of entity arrays; search wikipedia for the entity
                    wikis= []
                    for ent in ents:
                        # tie each of the returns from the OCR imprint extraction to the entities in the script string
                        rdat = extendglove(np.append(cdat[fl][const.IMG],ent),rdat if not (rdat == None) else gfl)
                        # grab the wikipedia data
                        ret  = Parallel(n_jobs=nc)(delayed(cognitive)(const.WIK,e,inst,objd,testing) for e in ent)
                        for r in ret:
                            wikis.extend(r)
                    # call extendglove to produce the GloVe output and transform it to an array
                    # with the first term in each row being the key and all other terms are values
                    rdat = extendglove(wikis,rdat if not (rdat == None) else gfl)
                # write the glove output to the knowledge graph
                #rdat = [(k,v) for k,v in list(rdat.items())[0:M]]
                #print(kg(const.ENTS,inst,coln,rdat,g,False,testing))
    else:
        print(cdat)
        print(sg)
