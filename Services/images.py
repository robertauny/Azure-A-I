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
from string                                         import punctuation
from math                                           import ceil,log,exp

from keras.utils                                    import to_categorical

from data                                           import url_kg,read_kg,sodaget
from services                                       import images,kg
from ai                                             import create_kg,extendglove,thought,cognitive
from nn                                             import dbn

import constants as const

import config
import data

import numpy           as np
import multiprocessing as mp
import _pickle         as pickle

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
        # number of clusters
        nc   = len(files)
        # GloVe configurations
        typ  = cfg["instances"][inst]["src"]["types"]["glove"]
        gfl  = cfg["instances"][inst]["sources"][src][typ]["connection"]["file"]
        # for each image file, get all wikipedia files using the scripts returned
        rdat = None
        inp  = []
        out  = []
        ent  = []
        for fl in files:
            if "error" not in list(sg[fl].keys())        and \
                sel["splimprint"] in list(sg[fl].keys()) and \
                sg[fl][sel["rxstring"]] not in [None]:
                # the scripts
                scrs = sg[fl][sel["rxstring"]].translate(str.maketrans('','',punctuation)).lower().split()
                # everything that will be added to the glove data set
                impr = sg[fl][sel["splimprint"]].translate(str.maketrans('','',punctuation)).replace(" ","").lower()
                ents = list(np.append(impr,data.unique(np.append(cdat[fl][const.IMG],scrs))))
                ent.extend(ents)
                # each ents return is an array of entity arrays; search wikipedia for the entity
                # tie each of the returns from the OCR imprint extraction to the entities in the script string
                rdat = extendglove(ents ,rdat if not (rdat == None) else gfl[0])
                # grab the wikipedia data
                wikis= cognitive(const.WIK,scrs,inst,objd,testing)
                # add the wikipedia data to the extended glove data set
                rdat = extendglove(wikis,rdat if not (rdat == None) else gfl[0])
                # associate the inputs with the outputs for building the model used when predicting
                for s in np.unique(cdat[fl][const.IMG]):
                    s    = s.translate(str.maketrans('','',punctuation)).replace(" ","").lower()
                    inp.append(rdat[s])
                    out.append(np.min(rdat[impr]))
        # build and save the model using the inputs and outputs
        s    = 2
        p    = int(ceil(log(len(inp[0]),s)/2.0))
        odat = to_categorical(out,num_classes=nc)
        # generate the cluster model
        mdl  = dbn(np.asarray(inp),odat,splits=s,props=p,clust=nc)
        # save the cluster model to a local file
        mdl.save(gfl[2])
        # limit the data
        rdat = [(k,v) for k,v in list(rdat.items()) if k in ent]
        # write the extended glove data to a file for later recall
        with open(gfl[1],"w+") as f:
            for k,v in rdat:
                f.write(str(k))
                for i in range(0,len(v)):
                    f.write(" %lf" % v[i])
                f.write("\n")
            f.close()
        if False:
            # write the glove output to the knowledge graph
            #
            # keys and values in the GloVe dataset
            keys = list(list(rdat)[i][0] for i in range(0,len(rdat)))
            vals = list(list(rdat)[i][1] for i in range(0,len(rdat)))
            # use an explicit dict to make sure that the order is preserved
            coln = [(keys[i],i) for i in range(0,len(keys))]
            # create the data for the sample knowledge graph (only one brain)
            kgdat= create_kg(inst,vals,s)
            # populate the knowledge graphs with the data
            k1   = kg(const.V   ,inst,coln,kgdat,g,False,testing)
            # write the glove output to the knowledge graph
            k2   = kg(const.ENTS,inst,coln,rdat ,g,True ,testing)
    else:
        print(cdat)
        print(sg)
