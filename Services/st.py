#!/usr/bin/python

############################################################################
##
## File:      st.py
##
## Purpose:   Clean data and make predictions
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Feb. 10, 2022
##
############################################################################

import sys

import constants as const

from joblib                                         import Parallel,delayed
from string                                         import punctuation
from math                                           import ceil,log,exp
from PIL                                            import ImageDraw,Image

ver  = sys.version.split()[0]

if ver == const.constants.VER:
    from            keras.models                            import Sequential,load_model,Model
    from            keras.utils.np_utils                    import to_categorical
else:
    from tensorflow.keras.models                            import Sequential,load_model,Model
    from tensorflow.keras.utils                             import to_categorical

from ai                                             import brain,create_kg,extendglove,thought,cognitive,wvec,almost,glovemost
from nn                                             import dbn,calcC,nn_importance

import config
import data

import numpy           as np
import multiprocessing as mp

import re

# gremlin imports
from gremlin_python.structure.graph                 import Graph
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection

cfg  = config.cfg()

np.random.seed(const.constants.SEED)

############################################################################
##
## Purpose:   Process the data
##
############################################################################
def fixdata(inst=0,dat=[]):
    ret  = None
    if inst > const.constants.BVAL and type(dat ) in [type([]),type(np.asarray([]))] and len(dat ) > 0:
        d    = np.asarray(dat).copy()
        # check which rows have any null values and remove them
        #
        # doing rows first since a null row will eliminate all columns
        rows = [i for i in range(0,len(d)) if d[i].all() == None]
        d    = d[-rows,:] if len(rows) > 0 else d
        # have to check that we still have rows of data
        if not (len(d) == 0 or len(d[0]) == 0):
            # check which columns have any null values and remove them
            d1   = d.transpose()
            cols = [i for i in range(0,len(d1)) if d1[i].all() == None]
            d    = d[:,-cols] if len(cols) > 0 else d
            # have to check that we still have columns of data
            if not (len(d) == 0 or len(d[0]) == 0):
                # for those remaining rows, we want to keep track of any columns
                # that have missing values, as we will only model with completely
                # full rows/columns

                # for the rows that are left, we will use to fill missing values
                #
                # first we will determine importance of each feature, with the reason
                # being, we will be modeling each feature in the data set as a function
                # of the top features as determined by importance
                #
                # first we will make use of the central limit theorem when making the
                # assumption that the original data set is a mixed bag of normals that
                # we want to separate (cluster)
                #
                # implicitly, we are making use of a result from the random cluster model
                # that allows us to determine the number of clusters based upon the assumption
                # of normality, plus ordering that gives uniformity
                #
                # inputs for importance are the subset of rows that have values
                ip   = d
                # mixed bag of normals used as inputs when calculating labels for use as outputs
                ndat = np.random.normal(size=(len(d),1)).flatten()
                # outputs for importance calculated as categorical labels
                op   = calcC(ndat)
                # gauge the importance of each feature of the modified data set
                imp  = nn_importance(ip,op)
                # finally replace the values for use in building the models to fix the data
                ipt  = imp.transform(ip)
                # gonna brute force a way to check which features are being selected from the data
                ncols= []
                for i in range(0,len(dat[0])):
                    for j in range(0,len(ipt)):
                        if dat[-rows,i] == ipt[:,j]:
                            ncols.append(i)
                # now we will build "brains" from the transformed data that will do the "thinking"
                # for us when we want to replace missing values in the original data set
                #
                # instantiate a JanusGraph object
                graph= Graph()
                # connection to the remote server
                conn = DriverRemoteConnection(data.url_kg(inst),'g')
                # get the remote graph traversal
                g    = graph.traversal().withRemote(conn)
                # make a data set for each column of data needing replacement values
                for i in cols:
                    ndat = np.vstack((dat[-rows,i],ipt))
                    # create column names (normally obtained by var.dtype.names)
                    coln = {"col"+str(k):(k-1) for k in range(1,len(ndat[0])+1)}
                    # create the knowledge graph that holds the "brains"
                    kgdat= create_kg(inst,ndat,permu=[tuple(list(range(len(coln))))],limit=True)
                    # write the knowledge graph
                    dump = [data.write_kg(const.constants.V,inst,list(coln.items()),k,g,False) for k in kgdat]
                    dump = [data.write_kg(const.constants.E,inst,list(coln.items()),k,g,False) for k in kgdat]
                    # thought function will give us the predictions for replacement in original data set
                    for j in rows:
                        dat[j,i] = thought(inst,list(coln.items()),dat[j,ncols]) if not dat[j,i].any() else dat[j,i]
    return ret

############################################################################
##
## Purpose:   Process the data
##
############################################################################
def st_testing(inst=0,train=False,testing=False):
    ret  = {}
    try:
        # turn the punctuation into a list
        punc = []
        punc.extend(punctuation)
        # getting constants from the JSON config file
        src  = cfg["instances"][inst]["src"]["index"]
        typ  = cfg["instances"][inst]["src"]["types"]["ocri"]
        imgs = cfg["instances"][inst]["sources"][src][typ]["connection"]["files"]
        typ  = cfg["instances"][inst]["src"]["types"]["pill"]
        sel  = cfg["instances"][inst]["sources"][src][typ]["connection"]["sel"  ]
        typ  = cfg["instances"][inst]["src"]["types"]["glove"]
        gfl  = cfg["instances"][inst]["sources"][src][typ]["connection"]["file"]
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
            # for each image file, get all wikipedia files using the scripts returned
            rdat = None
            inp  = []
            out  = []
            ent  = []
            cent = []
            kimpr= {}
            for fl in files:
                if "error" not in list(sg[fl].keys())        and \
                    sel["splimprint"] in list(sg[fl].keys()) and \
                   (not type(sg[fl][sel["rxstring"]]) == type(None)):
                    imprs= []
                    impr = []
                    # the scripts
                    if type(sg[fl][sel["rxstring"]]) in [type([]),type(np.asarray([]))]:
                        for i in range(0,len(sg[fl][sel["rxstring"]])):
                            scrs = sg[fl][sel["rxstring"]][i].translate(str.maketrans('','',punctuation)).lower().split()
                            # everything that will be added to the glove data set
                            simpr= "|".join(p for p in punc if p in sg[fl][sel["splimprint"]][i])
                            if not (len(simpr) == 0):
                                imprs       = re.split(simpr,sg[fl][sel["splimprint"]][i].lower())
                                impr        = "".join(imprs)
                                kimpr[impr] = sg[fl][sel["splimprint"]][i]
                    else:
                        scrs = sg[fl][sel["rxstring"]].translate(str.maketrans('','',punctuation)).lower().split()
                        # everything that will be added to the glove data set
                        simpr= "|".join(p for p in punc if p in sg[fl][sel["splimprint"]])
                        if not (len(simpr) == 0):
                            imprs       = re.split(simpr,sg[fl][sel["splimprint"]].lower())
                            impr        = "".join(imprs)
                            kimpr[impr] = sg[fl][sel["splimprint"]]
                    ents = list(np.append(imprs,np.append(impr,data.unique(np.append(cdat[fl][const.constants.IMG],scrs)))))
                    ent.extend(ents)
                    # each ents return is an array of entity arrays; search wikipedia for the entity
                    # tie each of the returns from the OCR imprint extraction to the entities in the script string
                    rdat = extendglove(ents ,rdat if not (rdat == None) else gfl[0])
                    # grab the wikipedia data
                    wikis= cognitive(const.constants.WIK,scrs,inst,objd,testing)
                    # add the wikipedia data to the extended glove data set
                    rdat = extendglove(wikis,rdat if not (rdat == None) else gfl[0])
            # limit the data and revert the keys back to having the original imprints when possible
            rdat = [(kimpr[k] if k in list(kimpr.keys()) else k,list(np.asarray(v)[range(0,min(len(kimpr),len(v)))])) for k,v in list(rdat.items()) if k in kimpr]
            # write the extended glove data to a file for later recall
            with open(gfl[1],"w+") as f:
                for k,v in rdat:
                    f.write(str(k))
                    for i in range(0,len(v)):
                        f.write(" %lf" % v[i])
                    f.write("\n")
                f.close()
            # instantiate a JanusGraph object
            graph= Graph()
            # connection to the remote server
            conn = DriverRemoteConnection(url_kg(inst),'g')
            # get the remote graph traversal
            g    = graph.traversal().withRemote(conn)
            # write the glove output to the knowledge graph
            #
            # keys and values in the GloVe dataset
            keys = list(list(rdat)[i][0] for i in range(0,len(rdat)))
            vals = list(list(rdat)[i][1] for i in range(0,len(rdat)))
            # use an explicit dict to make sure that the order is preserved
            coln = [(keys[i],i) for i in range(0,len(keys))]
            # create the data for the sample knowledge graph (only one brain)
            perms= permutes(len(keys))
            kgdat= create_kg(inst,vals,2,perms)
            # populate the knowledge graphs with the data
            k1   = kg(const.constants.V,inst,coln,kgdat,g,True,testing)
            # see the extended glove data
            ret  = rdat
        else:
            # get the glove data
            gdat = {}
            fls  = {"pills":gfl[1],"imprs":gfl[2]}
            for key in fls:
                with open(fls[key]) as f:
                    gdat[key] = [(list(wvec(line).keys())[0],list(wvec(line).values())[0]) for line in f.readlines()]
                    f.close()
            # keys and values for pills and imprints glove data
            gpk  = list(dict(gdat["pills"]).keys()  )
            gpv  = list(dict(gdat["pills"]).values())
            gik  = list(dict(gdat["imprs"]).keys()  )
            giv  = list(dict(gdat["imprs"]).values())
            # for each file to be predicted
            # get its imprints from cdat to be used to match the keys in gdat["imprs"].keys()
            # get the row of data from gdat["imprs"].values() for inputs to the model
            # then take the location of the maximum of the return to get the cluster
            # take the key from pills that has row number matching the cluster number
            ret  = {}
            for fl in cdat:
                if "error" not in list(sg[fl].keys())        and \
                    sel["splimprint"] in list(sg[fl].keys()) and \
                   (not type(sg[fl][sel["rxstring"]]) == type(None)):
                    # start the return for this file
                    ret[fl]  = {sel[key]:None for key in list(sel.keys())}
                    # if nothing else, we will return the most frequently appearing data
                    if not (type(sg[fl][sel["rxstring"]]) == type("")):
                        # string column of the imprints in the image
                        sret = list(sg[fl][sel["rxstring"]])
                        # most frequently appearing response from the DB
                        gm   = glovemost(sret)
                        # row index of the most frequently appearing imprint
                        row  = sret.index(gm)
                    # row of data corresponding to the most frequently appearing imprint
                    for key in sel:
                        if (type(sg[fl][sel[key]]) == type("")):
                            ret[fl][sel[key]] = sg[fl][sel[key]]
                        else:
                            ret[fl][sel[key]] = sg[fl][sel[key]][row]
                    # now we will start to see if we can make a prediction
                    #
                    # for each file predicted
                    cont = True
                    for i in cdat[fl][const.constants.IMG]:
                        if cont:
                            # transformation of this imprint to match how it would have been stored
                            spl  = "|".join(p for p in punc if p in i)
                            impr = i.lower()
                            if not (len(spl) == 0):
                                impr = re.split(spl,impr)
                                impr = "".join(impr)
                            # this could generate a key error if the imprint is not in the glove data set
                            # so we will just check gdat["imprs"].keys() first then go forward
                            for key in gik:
                                # need these next lines because some of the imprints are not stripped and lowered
                                nkey = key
                                nspl = "|".join(p for p in punc if p in nkey)
                                nimpr= nkey.lower()
                                if not (len(nspl) == 0):
                                    nimpr= re.split(nspl,nimpr)
                                    nimpr= "".join(nimpr)
                                # if this (possibly) partial imprint matches part
                                # of a transformed full imprint taken from the current imprint
                                if impr in nimpr:
                                    # taking the constants just obtained, we can use the key to get the right
                                    # model and make a prediction about the right cluster for this key that
                                    # corresponds to an imprint ... if this imprint is noisy, we can take an
                                    # imprint in the same cluster and get all parameters corresponding to that
                                    # imprint by querying the knowledge graph for the other parameters in the
                                    # the knowledge graph using "sel" from the JSON config file
                                    #
                                    # no need for a cluster return from thought ... simply get all columns
                                    # starting with the one matching the key (use pop to get it first)
                                    #
                                    # read the data file to get the ordering, as the data will have been written
                                    # to the file in the same order as the constants should appear
                                    # then place the one matching key first and order all others in their original order
                                    # and pass this as coln to "thought" ... make thought return the values to be
                                    # matched with the values that were stored during training to get the key
                                    #
                                    # keys in gik are in the same ordering as when they were written during training
                                    # so there is no need to try to figure out the ordering before constructing coln for thought
                                    perms= permutes(len(gik))
                                    pred = {}
                                    for perm in perms:
                                        if key in np.asarray(gik)[perm]:
                                            coln = [(gik[k],k) for k in perm]
                                            # call thought to get the data values for comparison
                                            pred = thought(inst,coln,preds=0)
                                            if not (len(pred) == 0 or "error" in pred):
                                                break
                                    # string column of the imprints in the image
                                    sret = [a.translate(str.maketrans('','',punctuation)).lower() for a in list(sg[fl][sel["splimprint"]])]
                                    # use GloVe to calculate the most frequently appearing imprint
                                    gm   = glovemost(sret)
                                    # either we didn't get anything so that pred = {}
                                    # or we got something that might be {"error":"error string"}
                                    if not (len(pred) == 0 or "error" in pred):
                                        if pred["pred"] in sret:
                                            # row index of the most frequently appearing imprint
                                            row  = sret.index(pred["pred"])
                                        else:
                                            # use glove to find make a prediction using the sret data
                                            row  = sret.index(gm)
                                    else:
                                        # use glove to find make a prediction using the sret data
                                        row  = sret.index(gm)
                                    # row of data corresponding to the most frequently appearing imprint
                                    for k in sel:
                                        if (type(sg[fl][sel[k]]) == type("")):
                                            if sg[fl][sel["splshape_text"]]      == ret[fl][sel["splshape_text"]] and \
                                               sg[fl][sel["splcolor_text"]]      == ret[fl][sel["splcolor_text"]]:
                                                ret[fl][sel[k]] = sg[fl][sel[k]]
                                        else:
                                            if sg[fl][sel["splshape_text"]][row] == ret[fl][sel["splshape_text"]] and \
                                               sg[fl][sel["splcolor_text"]][row] == ret[fl][sel["splcolor_text"]]:
                                                ret[fl][sel[k]] = sg[fl][sel[k]][row]
                                    # no need to go further just break
                                    cont = False
                                    break
                        else:
                            break
                else:
                    ret[fl] = sg[fl]
                # draw the image with the predicted medication
                if not ("error" in ret[fl]):
                    if not (ret[fl][sel["rxstring"]] == None):
                        img  = Image.open(fl)
                        draw = ImageDraw.Draw(img)
                        draw.text((10,10),ret[fl][sel["rxstring"]],(0,0,0))
                        img.save(fl[0:fl.rfind(".")]+"_PRED"+fl[fl.rfind("."):len(fl)])
                        img.show()
    except Exception as err:
        ret["error"] = str(err)
    return ret
