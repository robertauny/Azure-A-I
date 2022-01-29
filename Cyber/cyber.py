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
## File:      cyber.py
##
## Purpose:   Knowledge graph encoding with a deep belief network
##            followed by data being written to Cosmos (or other graph) DB.
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Dec. 20, 2019
##
############################################################################

import nn
import ai
import data
import constants as const

import csv

import multiprocessing as mp

import numpy  as np
import pandas as pd

# *************** TESTING *****************

def cyber_testing(M=500,N=2):
    # number of data points, properties and splits
    m    = M
    p    = N
    if p > const.MAX_FEATURES:
        p    = const.MAX_FEATURES
    #s    = p + 1
    s    = p
    # uniformly sample values between 0 and 1
    #ivals= np.random.sample(size=(500,3))
    ivals= np.random.sample(size=(m,p))
    # create the data for the sample knowledge graph
    kg   = ai.create_kg(0,ivals,s)
    print(kg[const.E])
    # we need values to turn into labels when training
    # one-hot encode the integer labels as its required for the softmax
    ovals= nn.categoricals(M,s,p)
    # generate the model for using the test values for training
    model = nn.dbn(ivals,ovals,splits=s,props=p)
    if not (model == None):
        # generate some test data for predicting using the model
        ovals= np.random.sample(size=(m/10,p))
        # encode and decode values
        pvals= model.predict(ovals)
        # look at the original values and the predicted values
    else:
        print("Cyber Security model is null.")
    # test the corrections function on some kaggle data
    rows = []
    with open("/home/robert/data/food-inspections.csv") as f:
        dat1 = csv.reader(f,delimiter=",")
        flds = dat1.next()
        cnt  = 0
        for row in dat1:
            rows.append(row)
            cnt  = cnt + 1
            if (cnt == M):
                break
        dat  = np.asarray(rows)[:,4]
        print(dat)
        out  = ai.correction(dat)
        print(out)
        f.close()
    # test cyberglove
    a=ai.cyberglove(["/mnt/code/cyber/cyber.py","/mnt/code/cyber/ai.py","/mnt/code/cyber/nn.py"],20)
    print(a)
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
    #
    # when making predictions for reports, gather the data by cluster label
    # then train the regression model using the data ... finally make predictions
    # using new data passed to the prediction method of the trained model
