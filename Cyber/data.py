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
## File:      data.py
##
## Purpose:   Data utilities for connecting and reading/writing data.
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Nov. 22, 2021
##
############################################################################

from joblib                                         import Parallel, delayed
from itertools                                      import combinations,combinations_with_replacement
from string                                         import punctuation

import os
import sys
import traceback
import logging

from sodapy                                         import Socrata

# gremlin imports
from gremlin_python.structure.graph                 import Graph
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection

import pandas          as pd
import pandasql        as ps
import numpy           as np
import multiprocessing as mp

import config
import constants       as const

from nn                                             import dbn

# read the default json config file
cfg  = config.cfg()

np.random.seed(const.constants.SEED)

logging.disable(logging.WARNING)

############################################################################
##
## Purpose:   Unique list because np.unique returns strange results
##
############################################################################
def unique(l=[]):
    ret  = []
    if not (len(l) == 0):
        s    = l
        if type(s[0]) == type([]):
            s    = [tuple(t) for t in s]
        ret  = list(set(s))
        if type(s[0]) == type([]):
            ret  = [list(t) for t in ret]
    return np.asarray(ret)

############################################################################
##
## Purpose:   Permutations of a list of integers for use in labeling hierarchies
##
############################################################################
def permute(dat=[],mine=True,l=3):
    ret  = []
    sz   = len(dat)
    if not (sz == 0):
        if mine:
            # permute the array of indices beginning with the first element
            for j in range(0,sz+1):
                # all permutations of the array of indices
                jdat = list(dat[j:])
                jdat.extend(list(dat[:j]))
                for i in range(0,sz):
                    # only retain the sub arrays that are length >= 2
                    tmp = [list(x) for x in combinations(jdat,i+2)]
                    if len(tmp) > 0:
                        ret.extend(tmp)
        else:
            # number of cpu cores
            nc   = mp.cpu_count()
            # permute the array of indices beginning with the first element
            lsz  = l
            if not (0 < lsz and lsz < min(3,sz)):
                lsz  = 3
            ret.extend(list(combinations(dat,lsz)))
    return unique(ret)

############################################################################
##
## Purpose:   Use SODAPy to get data
##
############################################################################
def sodaget(inst=const.constants.BVAL,pill={},objd=True,lim=0,train=False):
    ret  = {}
    lpill= len(pill)
    if not (inst <= const.constants.BVAL or lpill == 0):
        # ordering of the data elements in the JSON file
        src  = cfg["instances"][inst]["src"]["index"]
        typ  = cfg["instances"][inst]["src"]["types"]["pill"]
        # NIH host
        host = cfg["instances"][inst]["sources"][src][typ]["connection"]["host"]
        # NIH DB
        db   = cfg["instances"][inst]["sources"][src][typ]["connection"]["db"  ]
        # generated NIH api token
        api  = cfg["instances"][inst]["sources"][src][typ]["connection"]["api" ]
        # generated NIH username
        user = cfg["instances"][inst]["sources"][src][typ]["connection"]["user"]
        # generated NIH password
        passw= cfg["instances"][inst]["sources"][src][typ]["connection"]["pass"]
        if not (host == None or db == None):
            # Unauthenticated client only works with public data sets. Note 'None'
            # in place of application token, and no username or password:
            cli  = Socrata(host,None)
            # Example authenticated client (needed for non-public datasets):
            #
            # cli  = Socrata(host,api,userame=user,password=passw)
            #
            # results, returned as JSON from API
            # converted to Python list of dictionaries by sodapy.
            # 
            # recursive loop through the results in pill
            if not (lpill == 1):
                nc   = mp.cpu_count()
                ret  = Parallel(n_jobs=nc)(delayed(sodaget)(inst,{k:pill[k]},objd,lim,train) for k in pill)
                # reformat the output
                ret  = {list(r.keys())[0]:list(r.values())[0] for r in ret}
            else:
                # single pill will contain image values and might contain object detection values
                keys = list(pill.  keys())[0]
                vals = list(pill.values())[0]
                if not (len(vals) == 0):
                    # we will filter the terms found in the image
                    # if there is only one unique term, then that is fine
                    # but if there are more than one unique terms, then
                    # we won't use the ones that are a single character
                    # because too many results are returned that are not useful
                    #
                    # image values
                    p    = pill[keys][const.constants.IMG] if objd else vals
                    p1   = unique(p)
                    #if not (len(p1) == 1):
                        #p1   = [v for v in p1 if len(v) > 1]
                    if not (len(p1) == 0):
                        # only a where clause for now
                        #
                        # not using a simple " or ".join(["splimprint like '%" + str(val) + "%'" for val in p[1]])
                        # since we want to strip white space and treat strings of len = 1 differently
                        spl  = "splimprint"
                        clr  = "splcolor_text"
                        shp  = "splshape_text"
                        rxs  = "rxstring"
                        whr1 = " OR ".join(["(" + spl + " LIKE '%;%" + str(val).replace(" ",";") + "%;%')" for val in p1])
                        whr2 = " OR ".join(["(" + spl + " LIKE '%;%" + str(val).replace(" ",";") + "'   )" for val in p1])
                        whr3 = " OR ".join(["(" + spl + " LIKE '"    + str(val).replace(" ",";") + "%;%')" for val in p1])
                        whr4 = " OR ".join(["(" + spl + "    = '"    + str(val).replace(" ",";") + "'   )" for val in p1])
                        whr  =                whr1 + " OR " + whr2 + " OR " + whr3 + " OR " + whr4
                        whr1 = " OR ".join(["(" + spl + " LIKE '%;%" + str(val).replace(" ","" ) + "%;%')" for val in p1])
                        whr2 = " OR ".join(["(" + spl + " LIKE '%;%" + str(val).replace(" ","" ) + "'   )" for val in p1])
                        whr3 = " OR ".join(["(" + spl + " LIKE '"    + str(val).replace(" ","" ) + "%;%')" for val in p1])
                        whr4 = " OR ".join(["(" + spl + "    = '"    + str(val).replace(" ","" ) + "'   )" for val in p1])
                        whr  = whr + " OR " + whr1 + " OR " + whr2 + " OR " + whr3 + " OR " + whr4
                        qry  = "$where " + whr
                        # select data columns
                        sel  = cfg["instances"][inst]["sources"][src][typ]["connection"]["sel"]
                        # build the query based upon the pills being sought and execute it using sodapy
                        try:
                            if not (sel == None):
                                if not (len(sel) == 0):
                                    spl  = sel[spl]
                                    clr  = sel[clr]
                                    shp  = sel[shp]
                                    rxs  = sel[rxs]
                                    cols = ",".join([s + " AS " + sel[s] for s in sel])
                                    qry  = "$select " + cols + qry
                                    if not (lim <= 0):
                                        res  = cli.get(db,app_token=api,select=cols,where=whr,limit=lim)
                                    else:
                                        res  = cli.get(db,app_token=api,select=cols,where=whr)
                                else:
                                    if not (lim <= 0):
                                        res  = cli.get(db,app_token=api,where=whr,limit=lim)
                                    else:
                                        res  = cli.get(db,app_token=api,where=whr)
                                # Convert to pandas DataFrame and drop duplicates
                                ret[keys] = pd.DataFrame.from_records(res)
                                if not train:
                                    ret[keys] = ret[keys].drop_duplicates()
                                # try to match the imprint on the medication
                                if not (len(ret[keys]) <= 1):
                                    # if object detection is turned on
                                    if objd:
                                        o    = pill[keys][const.constants.OBJ]
                                        s    = pill[keys][const.constants.SHP]
                                        # try to check the color
                                        clrs = np.append([o0.lower() for o0 in o[0]],[o[i].lower() for i in range(1,len(o))])
                                        bret = [(a.lower() in clrs) for a in ret[keys][clr].to_list()]
                                        iret = [i for i,b in enumerate(bret) if b]
                                        # rows matching colors of the pills
                                        if not (len(iret) <= 1):
                                            ret[keys] = ret[keys].iloc[iret,0:len(sel)]
                                            # try to check the shape
                                            bret = [(a.lower() in s) for a in ret[keys][shp].to_list()]
                                            iret = [i for i,b in enumerate(bret) if b]
                                            # rows matching shapes of the pills
                                            ret[keys] = ret[keys].iloc[iret,0:len(sel)]
                                        else:
                                            if not (len(iret) == 0):
                                                ret[keys] = ret[keys].iloc[iret,0:len(sel)]
                                    if not (len(ret[keys]) <= 1):
                                        # string column of the imprints in the image
                                        sret = ret[keys][rxs].to_list()
                                        # if not training then try to further de-dup
                                        if not train:
                                            usret     = unique(sret)
                                            rows      = [sret.index(u) for u in usret if not str(u).lower() == "nan"]
                                            if not (len(rows) == 0):
                                                ret[keys] = ret[keys].iloc[rows,0:len(sel)]
                                                sret      = list(np.asarray(sret)[rows])
                                        # row indices of all rx stirngs that are not NaN
                                        rows = [i for i,val in enumerate(sret) if not (len(str(val)) == 0 or str(val).lower() == "nan")]
                                        if not (len(rows) <= 1):
                                            # if we are not training, then return all matching rows of the imprints found
                                            ret[keys] = ret[keys].iloc[rows,0:len(sel)]
                                            if not (type(rows) == type(0)):
                                                ret[keys] = {sel[key]:ret[keys][sel[key]].to_numpy()[range(0,len(rows))] for key in list(sel.keys())}
                                            else:
                                                ret[keys] = {sel[key]:ret[keys][sel[key]]                                for key in list(sel.keys())}
                                        else:
                                            if not (len(rows) == 0):
                                                ret[keys] = ret[keys].iloc[rows[0],0:len(sel)]
                                                ret[keys] = {sel[key]:ret[keys][sel[key]] for key in list(sel.keys())}
                                            else:
                                                ret[keys] = {sel[key]:ret[keys][sel[key]].to_list()[0] for key in list(sel.keys())}
                                    else:
                                        if not (len(ret[keys]) == 0):
                                            # row of data corresponding to the most frequently appearing imprint
                                            ret[keys] = {sel[key]:ret[keys][sel[key]].to_list()[0] for key in list(sel.keys())}
                                        else:
                                            ret[keys] = {sel[key]:None for key in list(sel.keys())}
                                else:
                                    if not (len(ret[keys]) == 0):
                                        # row of data corresponding to the most frequently appearing imprint
                                        ret[keys] = {sel[key]:ret[keys][sel[key]].to_list()[0] for key in list(sel.keys())}
                                    else:
                                        ret[keys] = {sel[key]:None for key in list(sel.keys())}
                            else:
                                ret[keys] = {"error":"No configured DB columns"}
                        except Exception as err:
                            ret[keys] = {"error":str(err)}
                    else:
                        ret[keys] = {"error":"No UNIQUE imprints read"}
                else:
                    ret[keys] = {"error":"No imprints read"}
    return ret

############################################################################
##
## Purpose:   Get the URL to the knowledge graph
##
############################################################################
def url_kg(inst=const.constants.BVAL):
    ret  = None
    if not (inst <= const.constants.BVAL):
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
        ret  = prot + host + ":" + port + "/" + api
    return ret

############################################################################
##
## Purpose:   Read a knowledge graph from the remote DB
##
############################################################################
def read_kg(inst=const.constants.BVAL,coln=[],g=None):
    ret  = {"fl":[],"labels":[],"nns":[],"dat":[]}
    if not (inst <= const.constants.BVAL or len(coln) == 0):
        drop = False
        # column keys and values
        ckeys= np.asarray(coln)[:,0]
        cvals= np.asarray(coln)[:,1]
        try:
            if g == None:
                drop = True
                # depending upon what I decide to pass as arguments,
                # might be able to get the instance from lbl
                #
                # url to the remote KG DB
                url  = url_kg(inst)
                # instantiate a JanusGraph object
                graph= Graph()
                # connection to the remote server
                conn = DriverRemoteConnection(url,'g')
                # get the remote graph traversal
                g    = graph.traversal().withRemote(conn)
            # ordering of the data elements in the JSON file
            src  = cfg["instances"][inst]["kg"]
            # gremlin home dir
            home = cfg["instances"][inst]["sources"][src]["connection"]["home"]
            # file extension
            ext  = cfg["instances"][inst]["sources"][src]["connection"]["ext" ]
            # create the label that defines the ID of the data
            lbl  = [str(inst)]
            lbl.extend(cvals)
            lbl  = const.constants.SEP.join(map(str,lbl))
            # read all files matching the current pattern determined by lbl
            fls  = [{fl[0:fl.rfind(ext)]:fl for fl in f if fl.rfind(ext) > -1 and fl[0:fl.rfind(const.constants.SEP)] == lbl} for r,d,f in os.walk(home+"/data/")]
            if not (len(fls) == 0 or len(fls[0]) == 0):
                for fl in fls:
                    # file keys and values
                    fkeys= list(fl.keys())
                    fvals= list(fl.values())
                    # add the file from which the data came
                    ret["fl"].append(home+"/data/"+fvals[0])
                    # add the label for this file
                    ret["labels"].append(fkeys[0])
                    # add the neural network for this file
                    ret["nns"].append("models/"+fkeys[0]+".h5")
                    if drop:
                        # read the data for this file
                        g.io(ret["fl"][0]).read().iterate()
                    # get the data set
                    dat  = g.V().hasLabel(fkeys[0]).valueMap(True).toList()
                    if drop:
                        # drop all of the data that was just loaded
                        g.E().drop().iterate()
                        g.V().drop().iterate()
                    # parse the data set into what we want, assuming that the label
                    # consists of instance, brain, cluster and row numbers
                    if not (len(dat) == 0):
                        datk = unique([list(dat[j].keys()) for j in range(0,len(dat))])[0]
                        datv = [list(dat[j].values()) for j in range(0,len(dat))]
                        cdat = []
                        ddat = []
                        for k,key in enumerate(ckeys):
                            for d,x in enumerate(datk):
                                if str(x).translate(str.maketrans('','',punctuation)).lower() in key:
                                    cdat.append(k)
                                    ddat.append(d)
                                    break
                        if not (len(cdat) == 0):
                            if len(cdat) == len(ckeys):
                                ndat = []
                                for j in range(0,len(dat)):
                                    row  = []
                                    for i in ddat:
                                        row.append(datv[j][i][0])
                                    ndat.append(row)
                                # add the data for this file
                                #
                                # the data has the markov property so that
                                # each data point is the sum of the previous
                                # data point and zero-mean white noise
                                # but our data is almost certainly to not be zero mean so
                                # we will generate the data using the sample median and variance
                                #
                                # median
                                md   = np.median(np.asarray(ndat).astype(np.float),axis=0)
                                # variance
                                var  = np.var(np.asarray(ndat).astype(np.float),axis=0)
                                # the data
                                ret["dat"].append(md )
                                ret["dat"].append(var)
                            else:
                                ret  = read_kg(inst,np.asarray(coln)[cdat],None)
                        else:
                            ret["dat"].append([None])
                    else:
                        ret["dat"].append([None])
            else:
                ret["dat"].append([None])
            if drop:
                # close the connection
                conn.close()
        except Exception as err:
            ret  = str(err)
    return ret

############################################################################
##
## Purpose:   Heavy lift of writing a knowledge graph vertices and edges
##
############################################################################
def write_ve(stem=None,coln=[],kgdat=[],g=None):
    ret  = None
    lcol = len(coln)
    lkg0 = len(kgdat)
    # creating vertices/edges for exactly one KG
    if not (stem == None or lcol == 0 or lkg0 == 0 or g == None):
        if stem == const.constants.V:
            cols = np.asarray(coln)[:,0]
            # create IDs using the ID defined by the vertices
            ret  = kgdat[0].split(const.constants.SEP)
            if not (len(ret) <= 1):
                ret  = [const.constants.SEP.join(np.asarray(ret)[range(0,len(ret)-1)])]
            for i in range(0,len(kgdat)):
                if i == 0:
                    # add the ID
                    if not (len(kgdat) <= 1):
                        g    = g.addV(ret[i]).property("id",kgdat[i])
                    else:
                        ret.append(g.addV(cols[i]).property("id",kgdat[i]).next())
                        g    = ret[len(ret)-1]
                else:
                    # add all of the other properties
                    if i == len(kgdat)-1:
                        ret.append(g.property(cols[i-1],str(kgdat[i])).next())
                        # get the beginning of the graph
                        g    = ret[len(ret)-1]
                    else:
                        g    = g.property(cols[i-1],str(kgdat[i]))
    return ret

############################################################################
##
## Purpose:   Write word edges
##
############################################################################
def write_we(r=[],coln=[],kgdat=[],g=None):
    lcol = len(coln)
    lkg  = len(kgdat)
    ret  = False
    if not (len(r) == 0 or lcol == 0 or lkg == 0 or g == None):
        if lcol == lkg:
            for i in range(0,len(coln)):
                # id, vertex and GloVe constants of the current word
                ids1 = r[i][0]
                v1   = r[i][1]
                c1   = kgdat[i][1]
                for j in range(i+1,len(coln)):
                    # id, vertex and GloVe constants of the next words after the current one
                    ids2 = r[j][0]
                    v2   = r[j][1]
                    c2   = kgdat[j][1]
                    # create the ID
                    ids12= const.constants.SEP.join((ids1,ids2))
                    # create the edge between the vertices in question
                    g.V(v1).has("id",ids1).addE(const.constants.E).to(v2).property("id",ids12).property("weight",np.exp(np.dot(c1,c2))).fold()
            ret  = True
    return ret

############################################################################
##
## Purpose:   Heavy lift of writing a knowledge graph with needed NLP
##
############################################################################
def write_kg(stem=None,inst=const.constants.BVAL,coln=[],kgdat=[],g=None,drop=True):
    ret  = None
    if not (stem == None or inst <= const.constants.BVAL or len(coln) == 0 or len(kgdat) == 0 or g == None):
        if stem in [const.constants.V,const.constants.E]:
            # ordering of the data elements in the JSON file
            src  = cfg["instances"][inst]["kg"]
            # file extension
            ext  = cfg["instances"][inst]["sources"][src]["connection"]["ext" ]
            # current set of vertices
            vert = kgdat[const.constants.V]
            # returns for the current stem, one KG row at a time
            ret  = [write_ve(const.constants.V,coln,k,g) for k in vert]
            # possibly have multiple clusters and multiple rows in that cluster
            for clus in kgdat[const.constants.E]:
                if not (len(clus) == 0):# or len(clus[0]) == 0): # for the limit on edges in a cluster
                    # needed length for the weight of the edge
                    lclus= float(len(clus))
                    crow = []
                    for blk in clus:
                        # needed length for the weight of the edge
                        lblk = float(len(blk))
                        for row in blk:
                            row1 = int(row[1])
                            row2 = int(row[2])
                            # create the ID
                            ids1 = ret[row1-1][0] + const.constants.SEP + str(row1)
                            ids2 = ret[row2-1][0] + const.constants.SEP +                                   str(row2)
                            ids12= ret[row1-1][0] + const.constants.SEP + str(row1) + const.constants.SEP + str(row2)
                            # get vertices associated to the IDs in question
                            v1   = ret[row1-1][1]
                            v2   = ret[row2-1][1]
                            # create the edge between the vertices in question
                            g.V(v1).has("id",ids1).addE(const.constants.E).to(v2).property("id",ids12).property("weight",lblk/lclus).next()
                            # add the current row1 value to the list of rows in this cluster
                            crow.append(row1)
                    # write the graph to disk
                    fl   = "data/" + ret[0][0] + ext
                    g.io(fl).write().iterate()
                    if not len(crow) == 0:
                        # data to generate a neural network for the rows in the current cluster
                        dat  = np.asarray(vert)[unique(crow)-1,1:]
                        # generate the model
                        mdl  = dbn(np.asarray([[float(dat[j,i]) for i in range(0,len(dat[j,1:]))] for j in range(0,len(dat))])
                                  ,np.asarray( [float(dat[j,0])                                   for j in range(0,len(dat))])
                                  ,loss="mean_squared_error"
                                  ,optimizer="sgd"
                                  ,rbmact="selu"
                                  ,dbnact="tanh"
                                  ,dbnout=1)
                        # identify the file and save the data from the current cluster
                        fl   = "models/" + ret[0][0] + ".h5"
                        mdl.save(fl)
            # drop all of the data that was just loaded
            if drop:
                g.E().drop().iterate()
                g.V().drop().iterate()
        else:
            if stem == const.constants.ENTS:
                # read the graph associated with this sequence of columns
                kg   = read_kg(inst,coln,g)
                # returns for the current stem, one KG row at a time
                ret  = [write_ve(const.constants.V,[["word"]],[str(k[0])],g) for k in kgdat]
                # write the edges for these words just returned in ret
                if not (ret == None):
                    # write the edges to the graph
                    if write_we(ret,coln,kgdat,g):
                        # write the graph to disk
                        g.io(kg["fl"][0]).write().iterate()
                # drop all of the data that was just loaded
                if drop:
                    g.E().drop().iterate()
                    g.V().drop().iterate()
            else:
                if stem == const.constants.CONS:
                    # read the graph associated with this sequence of columns
                    kg   = read_kg(inst,coln,g)
                    # we should have the phrases and the sentiment in 2 separate blocks
                    # but we will make use of both at the same time when adding the
                    # phrases as concepts with words being connected to concepts
                    #
                    # before this logic can be used, the words should have been added
                    # to the graph already
                    #
                    # probe the graph to find a listing of all words and get the length
                    cols = [("word",0),("sentiment",1),("weight",2)]
                    for k in range(0,len(kgdat[0])):
                        # add the words in each phrase to the graph
                        for ph in kgdat[0][k]:
                            # some of the phrases are length 0 ... skip them
                            if not (len(ph) == 0):
                                # each phrase should be separated by a space
                                words= ph.split()
                                # each word should be in the graph already
                                # connect the words directly to each other
                                # and also to the concept if len > 1
                                # otherwise, there is no need
                                l    = len(words)
                                if not (l == 1):
                                    for i in range(0,l-1):
                                        # the word in question
                                        ret  = [write_ve(const.constants.V,cols,["word",words[i],kgdat[1][k][0],kgdat[1][k][1]],g)]
                                        if ret == None:
                                            continue
                                        # obtain the vertex ID associated with this word
                                        v1   = ret[0][1]
                                        for j in range(i+1,l):
                                            # the word in question
                                            ret  = [write_ve(const.constants.V,cols,["word",words[j],kgdat[1][k][0],kgdat[1][k][1]],g)]
                                            if ret == None:
                                                continue
                                            # obtain the vertex ID associated with this word
                                            v2   = ret[0][1]
                                            # create the ID
                                            ids12= const.constants.SEP.join((words[i],words[j]))
                                            # add the edge
                                            g.V(v1).addE(const.constants.E).to(v2).property("id",ids12).property("weight",kgdat[1][k][1]**2).next()
                                else:
                                    ret  = [write_ve(const.constants.V,cols,["word",words[0],kgdat[1][k][0],kgdat[1][k][1]],g)]
                    # write the graph to disk
                    g.io(kg["fl"][0]).write().iterate()
                    # drop all of the data that was just loaded
                    if drop:
                        g.E().drop().iterate()
                        g.V().drop().iterate()
                else:
                    # just a placeholder for the moment
                    # call the appropriate function in the future
                    # then append things to the graph using property tags
                    ret  = None
    return ret

############################################################################
##
## Purpose:   Read/write a knowledge graph from/to a graph DB using spark
##
############################################################################
def rw_kg(inst=const.constants.BVAL,df=None,testing=True):
    if not (inst <= const.constants.BVAL):
        # ordering of the data elements in the JSON file
        kg   = cfg["instances"][inst]["kg"]
        # configuration parameters
        host = cfg["instances"][inst]["sources"][kg]["connection"]["host"]
        port = cfg["instances"][inst]["sources"][kg]["connection"]["port"]
        acc  = cfg["instances"][inst]["sources"][kg]["connection"]["acc" ]
        db   = cfg["instances"][inst]["sources"][kg]["connection"]["db"  ]
        app  = cfg["instances"][inst]["sources"][kg]["connection"]["app" ]
        key  = cfg["instances"][inst]["sources"][kg]["connection"]["key" ]
        sec  = cfg["instances"][inst]["sources"][kg]["connection"]["sec" ]
        con  = cfg["instances"][inst]["sources"][kg]["connection"]["con" ]
        dirn = cfg["instances"][inst]["sources"][kg]["connection"]["dirn"]
        tbl  = cfg["instances"][inst]["sources"][kg]["connection"]["tbl" ]
        if not (host == None or
                port == None or
                acc  == None or
                db   == None or
                app  == None or
                sec  == None or
                con  == None or
                dirn == None or
                tbl  == None):
            url  = "https://" + host + ":" + port
            if not testing:
                # Write configuration
                wc = {
                    "Endpoint"  : url,
                    "Masterkey" : key,
                    "Database"  : dirn,
                    "Collection": table,
                    "Upsert"    : "true"
                }
                # spark sqlContext should be used to create the data frame
                # of edges and vertices in the following format
                #
                # vertices
                #
                # unique id, column listing
                # unique id column name, column listing name
                #
                # df = sqlContext.createDataFrame( [ ("id1","val11","val12",...,"val1n"),
                #                                    ("id2","val21","val22",...,"val2n"),
                #                                    ...,
                #                                    ["id","col1","col2",...,"coln"] ] )
                #
                # edges
                #
                # entity 1, related to entity 2, relationship
                # source entity, destination entity, relationship heading
                #
                # df = sqlContext.createDataFrame( [ ("id1","id2","rel12"),
                #                                    ("id2","id3","rel23"),
                #                                    ...,
                #                                    ("idm","idn","relmn"),
                #                                    ...,
                #                                    ["src","dst","relationship"] ] )
                if not df == None:
                    df.write                                        \
                      .format("com.microsoft.azure.cosmosdb.spark") \
                      .options(**wc                               ) \
                      .save()
                else:
                    # Read Configuration
                    rc = {
                        "Endpoint"                       : url,
                        "Masterkey"                      : key,
                        "Database"                       : db,
                        "Collection"                     : table,
                        "ReadChangeFeed"                 : "true",
                        "ChangeFeedQueryName"            : dirn,
                        "ChangeFeedStartFromTheBeginning": "false",
                        "InferStreamSchema"              : "true",
                        "ChangeFeedCheckpointLocation"   : "dbfs:/" + dirn
                    }
                    # Open a read stream to the Cosmos DB Change Feed via azure-cosmosdb-spark to create Spark DataFrame
                    df = (spark.readStream                                                                    \
                               .format("com.microsoft.azure.cosmosdb.spark.streaming.CosmosDBSourceProvider") \
                               .options(**rc                                                                ) \
                               .load())
            else:
                print(url)
    return df

############################################################################
##
## Purpose:   Housekeeping done when logging into a graph DB.
##
############################################################################
def blob_conf_set(inst=const.constants.BVAL,acc=None,testing=True):
    ret  = False
    if not (inst <= const.constants.BVAL or acc == None):
        # ordering of the data elements in the JSON file
        wh   = cfg["instances"][inst]["wh"]
        # configuration parameters
        key  = cfg["instances"][inst]["sources"][wh]["connection"]["key"]
        cstr = "fs.azure.account.key." + acc + ".blob.core.windows.net"
        if not testing:
            spark.conf.set(cstr,key)
        else:
            print(key)
            print(cstr)
        ret  = True
    return ret

############################################################################
##
## Purpose:   Read/write data from/to a data warehouse using spark
##
############################################################################
def rw_warehouse(inst=const.constants.BVAL,df=None,testing=True):
    if not (inst <= const.constants.BVAL):
        # ordering of the data elements in the JSON file
        wh   = cfg["instances"][inst]["wh"]
        # configuration parameters
        host = cfg["instances"][inst]["sources"][wh]["connection"]["host"]
        port = cfg["instances"][inst]["sources"][wh]["connection"]["port"]
        acc  = cfg["instances"][inst]["sources"][wh]["connection"]["acc" ]
        db   = cfg["instances"][inst]["sources"][wh]["connection"]["db"  ]
        app  = cfg["instances"][inst]["sources"][wh]["connection"]["app" ]
        key  = cfg["instances"][inst]["sources"][wh]["connection"]["key" ]
        sec  = cfg["instances"][inst]["sources"][wh]["connection"]["sec" ]
        con  = cfg["instances"][inst]["sources"][wh]["connection"]["con" ]
        dirn = cfg["instances"][inst]["sources"][wh]["connection"]["dirn"]
        tbl  = cfg["instances"][inst]["sources"][wh]["connection"]["tbl" ]
        if not (host == None or
                port == None or
                acc  == None or
                db   == None or
                app  == None or
                sec  == None or
                con  == None or
                dirn == None or
                tbl  == None):
            url  = "jdbc:sqlserver://" + host + ":" + port + ";databaseName=" + db + ";integratedSecurity=" + sec + ";applicationName=" + app
            wasbs= "wasbs://" + con + "@" + acc + ".blob.core.windows.net/" + dirn
            blob_conf_set(inst,acc,testing)
            if not testing:
                if not df == None:
                    df.write                                                \
                      .format("com.databricks.spark.sqldw"                ) \
                      .option("url"                                ,url   ) \
                      .option("forwardSparkAzureStorageCredentials","true") \
                      .option("dbTable"                            ,tbl   ) \
                      .option("tempDir"                            ,wasbs ) \
                      .save()
                else:
                    qry  = "select * from " + tbl + ";"
                    df   = spark.read                                                 \
                                .format("com.databricks.spark.sqldw"                ) \
                                .option("url"                                ,url   ) \
                                .option("tempDir"                            ,wasbs ) \
                                .option("forwardSparkAzureStorageCredentials","true") \
                                .option("query"                              ,qry   ) \
                                .load()
            else:
                print(url)
                print(wasbs)
    return df

#***************** TESTING ********************

def data_testing(inst=0):
    ret  = rw_warehouse(0)
    print(ret)
    ret  = rw_warehouse(0,"testing")
    print(ret)
    ret  = rw_kg(0)
    print(ret)
    ret  = rw_kg(0,"testing")
    print(ret)
    # demo of data source info
    src  = cfg["instances"][inst]["src"]["index"]
    mysql= cfg["instances"][inst]["src"]["types"]["mysql"]
    host = cfg["instances"][inst]["sources"][src][mysql]["connection"]["host"]
    print(str(src)+const.constants.SEP+str(mysql)+const.constants.SEP+host)
