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
    from            keras.preprocessing.text                import Tokenizer
    from            keras.preprocessing.sequence            import pad_sequences
else:
    from tensorflow.keras.models                            import Sequential,load_model,Model
    from tensorflow.keras.utils                             import to_categorical
    from tensorflow.keras.preprocessing.text                import Tokenizer
    from tensorflow.keras.preprocessing.sequence            import pad_sequences

from ai                                             import brain,create_kg,extendglove,thought,cognitive,wvec,almost,glovemost,img2txt
from nn                                             import dbn,calcC,nn_importance

import config
import data

import numpy           as np
import multiprocessing as mp

import csv

# gremlin imports
from gremlin_python.structure.graph                 import Graph
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection

cfg  = config.cfg()

np.random.seed(const.constants.SEED)

############################################################################
##
## Purpose:   Return the subset of data rows that are full
##
############################################################################
def checkdata(dat=[]):
    ret  = []
    rows = 0
    cols = 0
    if type(dat) in [type([]),type(np.asarray([]))] and len(dat) > 0:
        ret  = np.asarray(dat).copy()
        # check which rows have any null values and remove them
        #
        # doing rows first since a null row will eliminate all columns
        rows = [i for i in range(0,len(ret)) if any(a is None or len(a) == 0 for a in ret[i])]
        ret  = ret[[i for i in range(0,len(ret)) if i not in rows],:] if len(rows) > 0 else ret
        # have to check that we still have rows of data
        if not (len(ret) == 0 or len(ret[0]) == 0):
            # check which columns have any null values and remove them
            d1   = ret.transpose()
            cols = [i for i in range(0,len(d1)) if any(a is None or len(a) == 0 for a in d1[i])]
            ret  = ret[:,[i for i in range(0,len(ret[0])) if i not in cols]] if len(cols) > 0 else ret
    return ret,rows,cols

############################################################################
##
## Purpose:   Process the data
##
############################################################################
def fixdata(inst=0,dat=[]):
    ret  = None
    if inst > const.constants.BVAL and type(dat) in [type([]),type(np.asarray([]))] and len(dat) > 0:
        # check which rows/columns have any null values and remove them
        d,rows,cols = checkdata(dat)
        # have to check that we still have rows/columns of data
        if not (len(d) == 0 or len(d[0]) == 0):
            # for those remaining rows, we want to keep track of any columns
            # that have missing values, as we will only model with completely
            # full rows/columns
            #
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
                    if dat[[k for k in range(0,len(dat)) if k not in rows],i] == ipt[:,j]:
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
                ndat = np.vstack((dat[[k for k in range(0,len(dat)) if k not in rows],i],ipt))
                # create column names (normally obtained by var.dtype.names)
                coln = {"col"+str(k):(k-1) for k in range(1,len(ndat[0])+1)}
                # create the knowledge graph that holds the "brains"
                kgdat= create_kg(inst,ndat,permu=[tuple(list(range(len(coln))))],limit=True)
                # write the knowledge graph
                dump = [data.write_kg(const.constants.V,inst,list(coln.items()),k,g,False) for k in kgdat]
                dump = [data.write_kg(const.constants.E,inst,list(coln.items()),k,g,False) for k in kgdat]
                # thought function will give us the predictions for replacement in original data set
                for j in rows:
                    dat[j,i] = thought(inst,list(coln.items()),dat[j,ncols]).values()[0] if dat[j,i] is None else dat[j,i]
    return ret

############################################################################
##
## Purpose:   wikilabel support function
##
############################################################################
def wikidocs(inst=0,dat=[]):
    ret  = []
    if inst > const.constants.BVAL and type(dat) in [type([]),type(np.asarray([]))] and len(dat) > 0:
        # for each text "document" in the list we will tokenize
        # and get a Wikipedia for each word, then concatenate the docs
        #
        # keras tokenizer will also remove punctuation
        tok  = Tokenizer()
        tok.fit_on_texts(dat)
        items= np.asarray(list(tok.word_index.items()))
        # grab the wikipedia data with image object detection
        # and the testing flags both set to False
        wiki = cognitive(const.constants.WIK,items[:,0],inst,False,False,True)
        # append together all wikis and do some topic modeling
        # that will hopefully give more uniform values, as it seems
        # that some text values started out being similar but not
        # exactly the same, which can affect labeling
        ret.append(" ".join(wiki))
    return ret

############################################################################
##
## Purpose:   Use Wikipedia to expand text data and assign labels
##
############################################################################
def wikilabel(inst=0,dat=[],wik=False):
    ret  = None
    if inst > const.constants.BVAL and type(dat) in [type([]),type(np.asarray([]))] and len(dat) > 0:
        d    = np.asarray(dat).copy()
        if wik:
            # number of cpu cores
            nc   = mp.cpu_count()
            # for each text "document" in the list we will tokenize
            # and get a Wikipedia for each word, then concatenate the docs
            wikis= [wikidocs(inst,doc.split(" ")) for doc in d]
            wikis= np.asarray(wikis).flatten()
        else:
            wikis= d
        # now we will do some topic modeling
        topic= img2txt(const.constants.KP,wikis,inst,False,True)
        # when using the open source (local=True) method, the return
        # should contain the numeric label and the topic, separated by "_"
        ret  = list(map(lambda s: s.split("_",1), topic))
    return ret

############################################################################
##
## Purpose:   Process the data
##
############################################################################

# an instance number corresponds to the set of constants in one object
# of the JSON configuration array associated to the set of modules for this app
inst = 0

# getting constants from the JSON config file
src  = cfg["instances"][inst]["src"]["index"]
typ  = cfg["instances"][inst]["src"]["types"]["main"]
fls  = cfg["instances"][inst]["sources"][src][typ]["connection"]["files"]
fold = cfg["instances"][inst]["sources"][src][typ]["connection"]["fold"]

# we will get the subset of data rows/columns that are all non-null
# and make a determination of string vs. numeric
#
# strings will be labeled using wikilabel
if type(fls) in [type([]),type(np.asarray([]))] and len(fls) > 0:
    for fl in fls:
        with open(fl) as f:
            dat  = np.asarray([line for line in csv.reader(f)])
            f.close()
            # check which rows/columns have any null values and remove them
            d,rows,cols = checkdata(dat)
            # have to check that we still have rows/columns of data
            if not (len(d) == 0 or len(d[0]) == 0):
                # string columns will be labeled using wikilabel
                ncols= list(range(0,len(dat[0])))
                ncols= ncols[[i for i in range(0,len(ncols)) if i not in cols]] if len(cols) > 0 else ncols
                for i in range(0,len(d[0])):
                    if not bool(np.char.isnumeric(d[0,i])):
                        dat[[j for j in range(0,len(dat)) if j not in rows],ncols[i]] = wikilabel(inst,d[:,i],True)
                # fix the data by intelligently filling missing values
                dat  = fixdata(inst,dat)
