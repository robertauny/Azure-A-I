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

from ai                                             import create_kg,extendglove,thought,cognitive,img2txt,checkdata,fixdata,wikidocs,wikilabel,importance
from nn                                             import dbn,calcC

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
## Purpose:   Use A-I to expand and fix a data set before processing
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
                # indices that actually had data originally
                indx = [j for j in range(0,len(dat)) if j not in rows]
                # string columns will be labeled using wikilabel
                ncols= list(range(0,len(dat[0])))
                ncols= ncols[[i for i in range(0,len(ncols)) if i not in cols]] if len(cols) > 0 else ncols
                for i in range(0,len(d[0])):
                    if not bool(np.char.isnumeric(d[0,i])):
                        wiki = wikilabel(inst,d[:,i],True)
                        for k in indx:
                            # when using the open source (local=True) method, the return
                            # should contain the numeric label and the topic, separated by const.constants.SEP
                            #
                            # we only want the integer label for now, with the rest being used later
                            dat[k,ncols[i]] = wiki[indx.index(k)].split(const.constants.SEP)[0]
                # fix the data by intelligently filling missing values
                dat  = fixdata(inst,dat)
