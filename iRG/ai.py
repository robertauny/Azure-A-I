#!/usr/bin/python

############################################################################
##
## File:      ai.py
##
## Purpose:   Other A-I and machine learning functions needed for iRG and cyber security.
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Dec. 28, 2019
##
############################################################################

from joblib                       import Parallel,delayed
from itertools                    import combinations,combinations_with_replacement
from string                       import punctuation
from math                         import ceil,log,exp

from keras.utils                  import to_categorical
from keras.models                 import load_model

from keras.preprocessing.text     import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from pdf2image                    import convert_from_bytes,convert_from_path
from PIL                          import Image

import requests
import io
import json

import numpy  as np
import pandas as pd

import multiprocessing as mp
import os
import sys

import config
import constants as const
import nn
import data

from nn import dbn,categoricals

############################################################################
##
## Purpose:   Identify the cluster for the data row
##
############################################################################
def store(dat=[]):
    ret  = []
    sz   = len(dat)
    if not (sz == 0):
        # which cluster has been identified for storing the data
        tdat = [j for j,x in enumerate(dat) if x == max(dat)]
        ret  = tdat[0]
    return ret

############################################################################
##
## Purpose:   Cluster number
##
############################################################################
def prefix(i=const.BVAL):
    ret  = const.BVAL
    if not (i <= const.BVAL):
        ret  = i + 1
    return ret

############################################################################
##
## Purpose:   Cluster label and row number appended together
##
############################################################################
def append(i=const.BVAL,n=const.BVAL,m=const.BVAL):
    ret  = None
    if not (i <= const.BVAL or n <= const.BVAL or m <= const.BVAL):
        n   += 1 # note that this modifies the passed in value for all to see
        m   += (i+1) # note that this modifies the passed in value for all to see
        ret  = str(n) + "-" + str(m)
        #ret  = str(n+1) + "-" + str(m)
    return ret

############################################################################
##
## Purpose:   Extend a dictionary
##
############################################################################
def dextend(dat1={},k=None,v=None):
    ret  = dat1
    if not (k == None):
        ret[k] = v
    return ret

############################################################################
##
## Purpose:   Extend a dictionary
##
############################################################################
def dextend1(dat1={},dat2={}):
    # number of cpu cores
    nc   = mp.cpu_count()
    ret  = Parallel(n_jobs=nc)(delayed(dextend)(dat1,k,dat2[k]) for k in dat2.keys())
    return ret

############################################################################
##
## Purpose:   Extend an array
##
############################################################################
def extend(dat1=[],dat2=[]):
    ret  = []
    ret.append(dat1)
    ret.extend(dat2)
    return ret

############################################################################
##
## Purpose:   Extend an array
##
############################################################################
def extend1(dat1=[],dat2=[]):
    # number of cpu cores
    nc   = mp.cpu_count()
    ret  = Parallel(n_jobs=nc)(delayed(extend)(dat1,dat2[i]) for i in range(0,len(dat2)))
    return ret

############################################################################
##
## Purpose:   Split a string
##
############################################################################
def split(dat=[],ind=0):
    ret  = None
    if not (len(dat) == 0 or ind < 0 or ind > len(dat)-1):
        ret  = dat.split("-")[ind]
    return ret

############################################################################
##
## Purpose:   Check if a string is actually a number
##
############################################################################
def is_number(dat=None):
    ret  = False
    if not (dat == None):
        if (type(dat) in [type([]),type(np.asarray([]))]):
            ret  = map(is_number,dat)
        else:
            try:
                dump = float(dat)
                ret  = True
            except ValueError:
                ret  = False
    return ret

############################################################################
##
## Purpose:   Unique label for each row of data with cluster number and row number
##
############################################################################
def label(dat=[]):
    ret  = []
    # number of data points in all clusters to label
    sz   = len(dat)
    if not (sz == 0):
        num_cores = mp.cpu_count()
        # which cluster has been identified for storing the data
        ret  = Parallel(n_jobs=num_cores)(delayed(store)(dat[i]) for i in range(0,sz))
        # initialize a 2-D array for the counts of elements in a cluster
        tret = np.zeros((len(dat[0]),2),dtype=int)
        # set the cluster label prefix from 1 to length of dat[0]
        # which should be a binary word with exactly one 1 and a
        # number of zeros ... total length matches number of clusters
        tret[:,0] = Parallel(n_jobs=num_cores)(delayed(prefix)(i) for i in range(0,len(dat[0])))
        # append a unique number to the cluster label to identify
        # each unique point in the cluster
        #
        # also keeping track of the original data point ordering, marking which data points
        # went into which cluster ... this is done in the append function by adding (i+1) to tret
        ret  = Parallel(n_jobs=num_cores)(delayed(append)(i,ret[i],tret[ret[i]][1]) for i in range(0,len(ret)))
    return ret

############################################################################
##
## Purpose:   Unique list of labels for hierarchies depending upon features used
##
############################################################################
def unique(dat):
    ret  = []
    sz   = len(dat)
    if not (sz == 0):
        for i in range(0,sz):
            rsz  = len(ret)
            if rsz == 0:
                ret.append(dat[i])
            else:
                for j in range(0,rsz):
                    if ret[j] == dat[i]:
                        break
                    else:
                        if j == rsz - 1:
                            if not (ret[j] == dat[i]):
                                ret.append(dat[i])
    return ret

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
## Purpose:   Knowledge brain of all data hierarchies and neural networks
##
############################################################################
def brain(dat=[],splits=2):
    ret  = []
    # number of data points in all clusters to label
    sz   = len(dat)
    if not (sz == 0 or splits <= 0):
        # get all permutations of the 
        perms= permute(range(0,len(dat[0])))
        # add all models to the return
        for perm in perms:
            # model label
            lbl  = '-'.join(map(str,perm))
            # number of data points, properties and splits
            m    = len(dat)
            p    = len(perm)
            # we need values to turn into labels when training
            # one-hot encode the integer labels as its required for the softmax
            #
            # the outputs force the hierarchical classification model to 
            # produce regressors to segregate the data that amount to
            # projections onto the last feature of the data, as we are only using
            # the last of the input features in the definition of the outputs
            odat = to_categorical(dat[:,len(perm)-1],num_classes=splits**(2*p))
            # generate the cluster model
            mdl  = dbn(dat[:,perm],odat,splits=splits,props=p)
            # save the cluster model to a local file
            fl   = "models/" + lbl + ".h5"
            mdl.save(fl)
            # generate the regression model whose genesis is to learn the
            # model of the distribution of the highly correlated set of data
            # points that are assumed to have the markov property, meaning
            # that the most recent data point (or the last data point, in this case)
            # carries with it all information about the data points that preceded it
            # in the sequence of sampled output ... the idea for this model is simple
            # ... learn the distribution of the previous data points that will predict
            # the last data point ... then we can make predictions about the next values
            # to be be seen in the input sequence, as a distribution of this type will
            # be needed by the thought function
            #
            # the thought function will infer the next data points, which can be used to
            # infer the cluster to which the next data points belong using the clustering
            # model ... then we can predict the one-dimensional output as a function of the
            # multi-dimensional inputs using the specific cluster regression models that are
            # produced elsewhere ... the idea is this
            #
            # we have inputs from which we infer the next set of inputs, which will allow
            # us to make inferences about the set of outputs of interest
            #
            # we use almost all inputs and reason about the next set of inputs to get to the desired outputs
            # as this is the normal (human) thought process that we are attempting to model
            #
            # the way that we will find this model is to have the (k-1)-th data point to predict the k-th data point
            # since the markov property guarantees that only the (k-1)-th data point is significant in predicting
            # its successor k-th data point
            mdl  = dbn(dat[0:(len(dat)-1),perm]
                      ,dat[1: len(dat)   ,perm]
                      ,loss="mean_squared_error"
                      ,optimizer="sgd"
                      ,rbmact="selu"
                      ,dbnact="tanh"
                      ,dbnout=len(perm))
            # regression model
            rfl  = "models/" + "".join(lbl.split("-")) + ".h5"
            mdl.save(rfl)
            # add the current label and model to the return
            ret.append({"label":lbl,"model":fl,"rmodel":rfl})
    return ret

############################################################################
##
## Purpose:   Correct model for predicting future data as a service
##
############################################################################
def thought(inst=0,coln=[],preds=3):
    ret  = []
    if not (inst == None or len(coln) == 0 or preds <= 0):
        # assume that models are built so we will search the models
        # in the DB and return the one corresponding to the label 
        #
        # neural networks obtained from the DB later ... None for now
        df   = data.read_kg(inst,coln)
        # labels for each cluster
        lbls = df["labels"]
        # beginning and end indices in the label string that need to be removed to find the brain
        b    = lbls[0].find("-") + 1
        e    = lbls[0].rfind("-")
        # file for the clustering model
        fl   = str("models/"+lbls[0][b:e]+".h5")
        if os.path.exists(fl) and os.path.getsize(fl) > 0:
            # the data set for each cluster
            dat1 = df["dat"]
            # we could take the original data set that has been broken into clusters and merge it,
            # then shuffle it to randomize it (the original ordering doesn't matter much
            # in our case as the next point in the markov process only depends upon the last
            # one and the last one defines the first point used to find our predicted inputs)
            # instead we will just take the medians of the individual columns to make a new point
            #
            # so what we will do is to manufacture a data point by taking the median of each
            # column of the data defining the brain then predict the cluster for that data point
            if not (len(dat1) <= 1):
                dat  = dat1[0]
                if len(dat1) > 1:
                    for d in dat1[1:len(dat1)]:
                        dat  = np.vstack(dat,d)
                pt   = np.resize(np.median([row for row in dat if not (row == None)],axis=0),(1,len(dat[0])))
            else:
                if not (len(dat1) == 0):
                    pt   = np.resize(dat1,(1,len(dat1[0])))
            # load the clustering model and make the cluster prediction for this data point
            mdl  = load_model(fl)
            pred = store(mdl.predict(pt)[0])
            # regression neural networks for the predicted cluster
            nnet = df["nns"][pred]
            # what we want to do is to use the regression network for this brain
            # where the brain is obtained as the set of values in the dictionary
            # coln and make the number of requested predictions preds ... this network
            # is also defined by the labels when we remove the instance found before
            # the first "-" in the label and the cluster, which is found when removing characters
            # after the last "-" in the label ... then we just squeeze the remaining
            # characters and prepend with the "models/" directory and add the ".h5" file ext
            #
            # then get the column indices of the original data set that defines this brain
            cols = lbls[0][b:e].split("-")
            # load the regression model for this brain so we can make the predictions
            # of the next input data points that will be seen beyond the current data set
            rfl  = str("models/"+"".join(cols)+".h5")
            if os.path.exists(rfl) and os.path.getsize(rfl) > 0:
                # load the particular regression model for the chosen cluster
                rmdl = load_model(rfl)
                # so we load the model and make the predictions preds ... after making the
                # requested number of predictions, we use the main clustering model that has
                # almost the same name as the regression model, except we don't squeeze the remaining
                # characters since we leave the "-" in between ... we predict the cluster of the
                # resulting predictions from the regression model so that we know which cluster model
                # to use when making the final (more specific) regression predictions preds
                #
                # making the requested number of predictions
                rdat = []
                for i in range(0,preds):
                    npt  = rmdl.predict(pt)[0]
                    rdat.append(npt)
                    pt   = np.resize(npt,(1,len(npt)))
                # using these data points, make the predictions using nnet
                ret  = rmdl.predict(np.asarray(rdat))
    return ret

############################################################################
##
## Purpose:   Convert a list of strings into their ordinal representation
##
############################################################################
def chars(dat=[],pre=0,just=0):
    ret  = []
    sz   = len(dat)
    if not (pre < 0):
        if not (sz == 0):
            if not (just == 0):
                e    = dat.rjust(pre)
            else:
                e    = dat.ljust(pre)
        else:
            e    = " " * pre
        ret.extend(e)
    return ret

############################################################################
##
## Purpose:   Convert a list of strings into their ordinal representation
##
############################################################################
def numbers(dat=[],pre=0):
    ret  = None
    sz   = len(dat)
    if not (pre < 0):
        if not (sz == 0):
            d    = chars(dat,pre)
        else:
            d    = [" "] * pre
        ret  = [1.0/float(ord(x)) for i,x in enumerate(d)]
    return ret

############################################################################
##
## Purpose:  Implementation of Global Vectors for Word Representation (GloVe)
##
############################################################################
def glove(tok=None,words=0):
    ret  = {}
    # expecting that the corpus is already tokenized and integers are fit to the text
    if not (tok == None or words <= const.BVAL):
        # we will make each marginal distribution a function of uwrd words
        uwrd = len(tok.word_index.keys())
        if (0 < words and words < uwrd):
            uwrd = words
        # add the prior as the calculation of the probabilities of the top uwrd words
        #
        # calculation of the conditional specification elements carries all information about the distribution
        #
        # prior and the last element to make the dot products be the log probability of co-occurrence
        probs= []
        # dictionary of all words in the corpus
        ditem= dict(tok.word_index.items())
        # items of the top uwrd words from the dictionary
        items= [item for item in tok.word_counts.items() if ditem[item[0]] in range(1,uwrd+1)]
        for i,item in enumerate(items):
            # count associated to current word
            ccnt = item[1]
            # count associated to next word
            if not (i == uwrd-1):
                ncnt = items[i+1][1]
                # compute log probability of word-word co-occurrence
                lprob= np.log((ccnt+ncnt)/ccnt)
            else:
                # ensure that the last value is not zero, which necessarily means
                # it will be one if all other values are zero
                lprob= exp(max(probs))
            # add the last value to the dictionary
            probs.append(lprob)
        # each marginal is defined by a set of constants in such a way that the inner product of one set of
        # constants with another gives the probability of word-word co-occurrence of the words defining the marginals
        #
        # for our data files, we could compute similar numbers by fitting the texts to sequences of integers and
        # first computing the probability of word1 co-occurrence word2 as the product of the number of times word1
        # appears out of all words times number of times word2 appears out of all words in the corpus
        #
        # note that this leads to symmetry
        #
        # let's say that each row of the glove data file has N constants, preceded by an associated word so
        # that our goal for the combined data files is to obtain N constants for each word, such that their
        # mutual inner products result in the log probability that the words co-occur together
        #
        # to get the N constants, take a unique lising of the words and the first word from the list and randomly
        # generate N constants to associate with the first word ... then take the second word and randomly generate
        # N-1 constants while choosing the last constants so that the dot product of both sets of constants gives
        # the log probability of co-occurrence that was already computed ... then take the 3rd word and randomly
        # generate N-2 constants while choosing the other 2 constants so that the dot products of this line and the
        # previous 2 lines gives the associated 2 log probabilities of co-occurrence that were previously computed ...
        # note that this is a system of 2 equations in 2 unknowns resulting in a unique solution or no solution ...
        # continue in this fashion until all M <= N words have the needed constants ... if there are only M words, then
        # we have the needed glove constants for our data files ... otherwise we can truncate our data set in the Tokenizer
        # call by setting the argument "num_words=M" ... or we can simply use the constants and probabilities from the first
        # M words in the corpus to build a supervised neural network that will essentially give an "average" set of
        # constants that work for the other N-M words that were left ... see the comments below for more insights
        #
        # instead of randomly generating the constants with replacement at the end, instead we will use the priors of
        # each of the top uwrd words and the reasoning follows from a simple Bayesian theory ... note first that
        # the posterior probability $P(C|x) = P(C and x) / p(x) = P(C)p(x|C) / p(x)$ ... If we assume that the sample
        # space is partitioned into a certain collection of subsets, say $\{C_1,C_2,...,C_N\}$ with $C=C_K$, for some
        # $1 \le K \le N$, then clearly $p(x) = \sum_{K=1}^N P(x and C_K) = \sum_{K=1}^N P(C_K)p(x|C_K)$ so that the
        # denominator is the same for all calculations of posteriors, given $x$ ... Now, $p(x|C)$ can be estimated from
        # the data using the sample mean and variance as an approximately normal density (the inputs) so that our
        # modeling constants are an estimate of $P(C_K)$ for each $K$, which in this case are the constants associated
        # to each of the top uwrd words in our corpus
        #
        # Two notes for what comes next are that the constants can be reversed, as the last constant is associated to
        # the first word, next-to-last to the second, etc. ... and so that we are able to identify the word associated
        # to a given marginal, the most weight should be given to the constant associated with the word ... i.e. the last
        # constant is 1 when the associated word is the first in our top uwrd words, next-to-last is 1 for the 2nd, etc.
        #
        # note that if we want to control the weights by adding on an extra penalty term (so to speak) to the model by
        # performing L-normalization (necessarily meaning that our weights are in the unit interval), then we are allowing
        # that $\|w\| <= 1$ where $w = (C_K)_{K=1}^N$ when normally in such a case $\|w\| < 1$
        #
        # have to do this sequentially for each word in the top uwrd of the list because each set of constants are
        # derived from the previous sets of constants that are back solved using linear systems theory
        #
        # total number of appearances for all words
        tot  = sum(tok.word_counts.values())
        # start the process of generating glove marginals for the data set that's been tokenized
        for word,ind in tok.word_index.items():
            wcnt = ditem[word]
            if ret == {}:
                # initialize our first set of constants with the priors
                ret[word] = probs
                # if the word is in our top uwrd words, then weight it the most
                if wcnt in range(1,uwrd+1):
                    ret[word][wcnt-1] = 1.0
            else:
                # use all but cnt+1 probs values, as the rest are predetermined
                #
                # note that new entries into the dictionary will be first
                ret[word] = probs[:uwrd-cnt]
                # if the word is in our top uwrd-cnt words, then weight it the most
                if wcnt in range(1,uwrd-cnt+1):
                    ret[word][wcnt-1] = 1.0
                # extend the list of glove values for word after the random values
                # with all values in the last column, between the 2nd row and next to current row
                ret[word].extend(vals[0][range(uwrd-cnt,uwrd-1)])
                # append the final value which is determined as such, supposing uwrd = 3, giving a 3x3
                # matrix of glove constants to be used for the weights of each marginal we have
                # 
                #               a   b   c
                #               d   e   x1
                #               f   x2  x3
                # 
                # where a,b,c,d,e,f are randomly generated and x1,x2,x3 are unknowns that have to be calculated
                # so that the dot product of row1 with row2 gives the log probability of word-word co-occurrence
                # of word1 with word2 and so on ... now let y1 and y2 denote the probability of word-word co-occurrence
                # between row1 & row2 and row1 & row3, respectively, then we can formulate the set up
                # 
                #               a   b   c         a        a^2 + b^2 + c^2
                #               d   e   x1   *    b    =       log y1
                #               f   x2  x3        c            log y2
                # 
                # now we let log(y1) = z1, then we can solve for x1 = [z1-((a*d)+(b*e))]/c ... letting
                # log(y2) = z2 and x2 = x1, we can solve for x3 = [z2-((d*f)+(e*x1))]/x1 ... this same setup
                # generalizes to uwrd = n, for arbitrary n, with x2 being replaced by a vector of values previously
                # computed (now in the last column of the matrix) between the first and current row ... the dot product
                # of constants in the previous row together with the current row, not including computed values in the last column
                # of each row, is subtracted from the log probability of word-word co-occurrence
                #
                # previous word, which will be in the first position, as keys does not contain the present word
                pword= keys[0]
                # previous values, which will be in the first position, as values does not contain the present values
                pvals= vals[0]
                # compute log probability of word-word co-occurrence
                lprob= probs[cnt]
                # compute the dot product of values from previous row with what's currently in ret[word] to get final value
                dp   = np.dot(np.asarray(pvals)[range(0,len(pvals)-1)],np.asarray(ret[word])[range(0,len(ret[word]))])
                # compute the last value in this row of modeling constants for the marginals
                lval = (lprob-dp)/pvals[len(pvals)-1]
                # append the last value to the end of what's currently specified for this word
                ret[word].append(lval)
            # current keys
            keys = np.asarray(list(ret.keys()))
            # current values
            vals = np.asarray(list(ret.values()))
            # current count of words in the dictionary
            #
            # if uwrd < len(tok.word_index.keys()), then we need to recycle and restart the count after uwrd words
            cnt  = len(keys)
            ncnt = cnt % uwrd
            if not (ncnt == 0):
                cnt  = ncnt
            else:
                cnt  = 1
    return ret

############################################################################
##
## Purpose:   Which string in a list appears most (by whole words)
##
############################################################################
def almost(dat=[]):
    ret  = None
    if not (len(dat) == 0):
        ret  = max(set(dat),key=dat.count)
    return ret

############################################################################
##
## Purpose:   Which string in a list appears most (by whole words)
##
############################################################################
def glovemost(dat=[]):
    ret  = None
    sz   = len(dat)
    # need at least 2 rows for the modeling
    if not (sz <= 1):
        s    = 2
        # we want to find the value p such that s**(2*p) gives the same number
        # of potential clusters as there are found in glove, as indicated by the number of constants
        # found in a row of gdat (minus one accounts for the word) ... only need to use the first line for counting
        p    = 2
        # recall that the global distribution carries information about its marginals so that certain inputs will give a word
        # used to define an element of its conditional specification ... i.e. we get the values of the glove data set used
        # to build the global distribution ... this is exactly what we want ... start with a generated glove data set and
        # map this data set to elements of the conditional specification used to the build the global
        #
        # generate a "corpus" from the data that is passed in dat
        #
        # instantiate a default Tokenizer object without limiting the number of tokens
        tok  = Tokenizer()
        # create "texts" from the data
        txts = [word.translate(None,punctuation).replace(" ","").lower() for word in dat]
        # tokenize the data
        tok.fit_on_texts(txts)
        # tokenized keys and values with values corresponding to key rank in the corpus
        items= np.asarray(list(tok.word_index.items()))
        # get the glove keys and values
        keys = items[:,0]
        uks  = np.unique(keys)
        # number of clusters is same as the default defined by the splits and properties
        clust= len(uks)
        # must have more than one key, otherwise just return the one key
        if not (clust <= 1):
            # generate the glove data set
            gd   = glove(tok,clust)
            # we need values to turn into labels when training
            # one-hot encode the integer labels as its required for the softmax
            #
            # the goal is to build a model with clust=clust outputs such that when clust inputs are passed in, we get a
            # word in the glove data set associated to the clust constants ... furthermore, when the model constants are
            # used in dot product with the constants of any word from the glove data set, we get the average log probability
            # of the word from glove with any other arbitrary word from glove
            # 
            # now since every set of input constants is associated to a unique word from the glove data set, then our outputs
            # can consist of one unique value for each row of constants
            #
            # input values to the model are the values associated to the words that appear in our corpus
            ivals= np.asarray([gd[word] for word in txts])
            # generate the output values
            ukeys= {x:i for i,x in enumerate(uks)}
            ovals= to_categorical([ukeys[word] for word in txts],num_classes=clust)
            # create the model using the inputs and outputs
            model= dbn(ivals,ovals,splits=s,props=p,clust=clust)
            # after generating the model, we should have a set of constants (weights) that number the same as those
            # in the glove data file ... since we predict the word when using the associated constants of a word in the glove
            # data file, then we only need for the dot product of the constants from the model with those of the
            # constants in the data files to give us the log of the probability of the word-word co-occurrence
            #
            # however, the realizing distribution is a model of all words that gives you one particular word, when
            # a certain set of constants are used as inputs ... thus we can use the output of the model as the
            # global realizing distribution ... and the global distribution gives the mean log probability of co-occurrence
            # of any arbitrary word in the glove data set when dot product with the constants associated to another word
            #
            # in addition, we can predict any word, using its values, simply by supplying them as input to the model
            # and finding the word that corresponds to the most probable entry in the return, as indicated by
            # the highest value in the set of constants ... this is the word that we will return
            #
            # sample one word, get the prediction and return the word associated to the highest value
            nvals= np.median(ivals,axis=0).reshape((1,clust))
            preds= model.predict(nvals)
            # for each prediction, we want to identify the word associated with the prediction
            # where each prediction is a set of constants defining probabilities for each cluster
            #
            # we will return the most probable of all predictions
            ret  = []
            for row in preds:
                wrds = [word for word in dat if word.translate(None,punctuation).replace(" ","").lower() in \
                                                uks[[i for i,val in enumerate(row) if val == max(row)]]]
                if not (len(wrds) == 0):
                    ret.extend(wrds)
            if not (len(ret) == 0):
                ret  = almost(ret)
        else:
            ret  = dat[0]
    else:
        if not (sz == 0):
            ret  = dat[0]
    return ret

############################################################################
##
## Purpose:   Which string in a list appears most (by whole words)
##
############################################################################
def mostly(dat=[],rdat=[],cols=[],preds=[],pre=0):
    ret  = None
    sz   = len(dat)
    rsz  = len(rdat)
    csz  = len(cols)
    psz  = len(preds)
    if not (sz == 0 or rsz == 0 or csz == 0 or psz == 0 or pre <= 0):
        # number of cpu cores
        nc   = mp.cpu_count()
        # character-wise append of most frequent characters in each feature of the list of strings
        # the data are strings that have been converted into character arrays
        # with each character further being converted into its inverted ordinal representation
        #
        # since the data are clusters that should consist of related elements, then we can
        # create a model to find constants such that a combination of the converted chars
        # should all be the same value ... i.e. if the words are all 3 chars, then
        # for each related (possibly equal) word, ax + by + cz = 1 for each word in the
        # cluster where the characters of the word are [x,y,z] and the modeling constants
        # are [a,b,c] ... this can be done because the words in the cluster are all equal
        # or close in some statistical sense (probably differing only by some randomly
        # misplaced char)
        #
        # we predict the same data that is used to create the model and take the mean
        # of each column if there are more than one data row that gives the closest
        # prediction to 1 ... we do this because the goal is to denoise the inputs
        # and take the averaged rows of data coming closest to the expected output 1
        inds = [j for j,x in enumerate(preds) if x == max(preds)]
        # mean of all rows giving predictions close to the plane with bias 1
        mn   = np.mean(np.asarray(rdat)[inds],axis=0)
        # character-wise append of characters in each feature of the list of strings
        cdat = Parallel(n_jobs=nc)(delayed(chars)(dat[i],pre) for i in range(0,sz))
        cdat = np.asarray(cdat)
        # for the final return, take the characters closest to the mean
        rret = [" "] * pre
        for i in range(0,pre):
            if i in cols:
                udat    = "".join(unique(cdat[:,i]))
                ndat    = numbers(udat,len(udat))
                mmn     = abs(ndat-np.full(len(ndat),mn[i]))
                j       = [k for k,x in enumerate(mmn) if x == min(mmn)][0]
                rret[i] = udat[j]
            else:
                rret[i] = almost(list(cdat[:,i]))
        ret  = "".join(rret).strip()
    return ret

############################################################################
##
## Purpose:   Which string in a list appears most (by character)
##
############################################################################
def most(dat=[],rdat=[],cols=[],preds=[],pre=0):
    ret  = None
    sz   = len(dat)
    if not (sz == 0 or pre < 0):
        # number of cpu cores
        nc   = mp.cpu_count()
        # character-wise append of most frequent characters in each feature of the list of strings
        cdat = Parallel(n_jobs=nc)(delayed(chars)(dat[i],pre) for i in range(0,sz))
        cdat = np.asarray(cdat)
        # change this to use all 4 predictions combined someway
        ret  = glovemost(dat)
        if not (ret in dat):
            ret  = mostly(dat,rdat,cols,preds,pre)
            if not (ret in dat):
                mdat = Parallel(n_jobs=nc)(delayed(max)(set(cdat[:,i].tolist()),key=cdat[:,i].tolist().count) for i in range(0,pre))
                ret  = "".join(mdat).strip()
                if not (ret in dat):
                    ret  = almost(dat)
    return ret

############################################################################
##
## Purpose:   Use DBN to correct a list of corrupt or misclassified strings
##
############################################################################
def correction(dat=[],mval=1000,pcnt=0.1,lo=2):
    ret  = None
    sz   = len(dat)
    if not (sz == 0):
        # number of cpu cores
        nc   = mp.cpu_count()
        # length of longest string in the list
        ssz  = max([len(x) for i,x in enumerate(dat)])
        # numeric representations of the strings in the list
        ndat = Parallel(n_jobs=nc)(delayed(numbers)(str(dat[i]).lower(),ssz) for i in range(0,sz))
        ndat = np.asarray(ndat)
        #
        # compute the values
        #
        # the outputs force the hierarchical classification model to 
        # produce regressors to segregate the data that amount to
        # auto-encoders, as we are using all of the input data elements
        # in the definition of the outputs
        perms= permute(range(0,ssz),False)
        lmax = sys.maxint
        # initial entropy is something higher than otherwise possible
        cdt  = np.asarray(Parallel(n_jobs=nc)(delayed(chars)(dat[i],ssz) for i in range(0,sz)))
        cdat = [cdt[i,0].lower() for i in range(0,len(cdt)) if cdt[i,0].isalpha()]
        # lower bound on the number of clusters to seek has to be >= 2 as at least one error is assumed
        #lo   = len(np.unique(cdat))
        # calculate the means of each column, as we will use permutations of subsets of all columns
        # and the mean in those columns that are not included in the permutation to test if entropy
        # is increased or decreased as a result ... idea is to find the permutation of columns
        # that leaves us with the fewest errors in the data set (lowest entropic state)
        mns  = Parallel(n_jobs=nc)(delayed(np.mean)(ndat[:,i]) for i in range(0,len(ndat[0])))
        # calculate some parameters for the main dbn
        pdat = ndat
        ptdat= [sum(x)/(len(x)*max(x)) for x in pdat]
        pydat= np.asarray(ptdat)
        # we should sample at least as many data elements as there are clusters
        if len(ndat) > mval:
            ind  = [np.random.randint(0,len(ndat)) for i in range(0,max(lo,int(ceil(pcnt*len(ndat)))))]
        else:
            ind  = range(0,len(ndat))
        # generate the model of the data for smoothing errors
        #
        # the idea is to smooth out the errors in the data set
        # and use the data set that generates the model
        # that does the best job of smoothing, resulting in
        # fewer unique values, as fewer unique values are the result
        # of the values with erroneous characters being classified with
        # their correct counterparts
        model= dbn(pdat[ind]
                  ,pydat[ind]
                  ,loss='mean_squared_error'
                  ,optimizer='sgd'
                  ,rbmact='sigmoid'
                  ,dbnact='sigmoid'
                  ,dbnout=1)
        for perm in perms:
            # we only want permutations of columns that are sorted correctly
            # as we need to pass the columns to the predictor in the right order
            if (np.array_equal(perm,np.sort(perm))):
                # add a column of means if the column is not in the current permutation
                # otherwise add the data corresponding to the column in the permutation
                pdat = []
                for i in range(0,len(ndat[0])):
                    if not (i in perm):
                        pdat.append(np.full(len(ndat[:,i]),mns[i]))
                    else:
                        pdat.append(ndat[:,i])
                # columns were added as rows so we need to take the transpose
                pdat = np.asarray(pdat).transpose()
                # make the predictions
                psdat= model.predict(pdat)
                # record the updated predictions
                updat= unique(psdat)
                hi   = len(updat)
                # is this the first time through the loop of perms
                if np.array_equal(perm,perms[0]):
                    cols = perm
                    lmax = hi
                    tdat = psdat
                    udat = {max(x):i for i,x in enumerate(updat)}
                else:
                # is there a reduction in current entropy
                # if so, then record the current conditions
                    if (lo <= hi and hi < lmax):
                        cols = perm
                        lmax = hi
                        tdat = psdat
                        udat = {max(x):i for i,x in enumerate(updat)}
        # we need values to turn into labels when training
        # one-hot encode the numeric data as its required for the softmax
        #
        # splits
        s    = 2
        #
        # properties
        #
        # we want to find the value p such that s**(2*p) gives the same number
        # of potential clusters as there are found in udat
        p    = int(ceil(log(len(udat),s)/2.0))
        #
        # number of classes, one-hot encodings
        ncls = s**(2*p)
        cdat = [to_categorical(udat.get(x[0]),num_classes=min(lmax,ncls)) for x in tdat]
        odat = np.asarray(cdat)
        # get the labels
        lbls = label(odat)
        # split the labels to know the available clusters
        slbl = Parallel(n_jobs=nc)(delayed(lbls[i].split)("-") for i in range(0,len(lbls)))
        slbl = np.asarray(slbl)
        # cluster labels
        clus = slbl[:,0]
        ucls = np.unique(clus)
        # row numbers
        rows = slbl[:,1]
        # collect all data for each cluster and assign most numerously appearing value
        ret  = np.asarray([" "*ssz]*sz)
        for cls in ucls:
            # all row indices associated with the current cluster
            ind      = [j for j,x in enumerate(clus) if x == cls]
            # all data elements associated with the current cluster
            idat     = [ dat[x] for j,x in enumerate(ind)]
            ipdat    = [pdat[x] for j,x in enumerate(ind)]
            itdat    = [tdat[x] for j,x in enumerate(ind)]
            # select the label (data element) that appears most in this cluster
            ret[ind] = most(idat,ipdat,cols,itdat,ssz)
    return ret

############################################################################
##
## Purpose:   Convert Python image library (PIL) image to an array
##
############################################################################
def pil2array(pil=None):
    ret  = None
    if not (pil == None):
        # get addressable memory for writing the image
        imgb = io.BytesIO()
        # save the passed image to the addressable memory
        pil.save(imgb,format='PNG')
        # get a reference to the addressable memory for the return
        ret  = imgb.getvalue()
    return ret

############################################################################
##
## Purpose:   Convert Python image library (PIL) image data to text
##            Other text analytics
##
############################################################################
def img2txt(wtyp=const.OCR,docs=[],inst=const.BVAL,testing=True):
    ret  = []
    if not (wtyp == None or len(docs) == 0 or inst <= const.BVAL):
        # get the default configuration
        cfg  = config.cfg()
        # ordering of the data elements in the JSON file
        src  = cfg["instances"][inst]["src"]["index"]
        typ  = cfg["instances"][inst]["src"]["types"][wtyp]
        # azure subscription key
        key  = cfg["instances"][inst]["sources"][src][typ]["connection"]["key"]
        # azure vision api
        host = cfg["instances"][inst]["sources"][src][typ]["connection"]["host"]
        # api
        api  = cfg["instances"][inst]["sources"][src][typ]["connection"]["api"]
        # version
        ver  = cfg["instances"][inst]["sources"][src][typ]["connection"]["ver"]
        # app
        app  = cfg["instances"][inst]["sources"][src][typ]["connection"]["app"]
        # url
        url  = "https://" + host + "/" + api + "/" + ver + "/" + app
        # request headers. Important: content should be bytestream as we are sending an image from local
        hdrs = {"Ocp-Apim-Subscription-Key":key}
        parms= {"language":"unk","detectOrientation":"true"}
        if not testing:
            ftext= []
            if wtyp == const.OCR:
                hdrs["Content-Type"] = "application/octet-stream"
                for i in docs:
                    try:
                        # get response from the server
                        resp = requests.post(url,headers=hdrs,params=parms,data=i)
                        resp.raise_for_status()
                        # get json data to parse it later
                        js   = resp.json()
                        # all the lines from a page, including noise
                        for reg in js["regions"]:
                            line = reg["lines"]
                            for elem in line:
                                ltext = " ".join([word["text"] for word in elem["words"]])
                                ftext.append(ltext.lower())
                    except Exception as err:
                        ftext.append(str(err))
            else:
                # request headers. Important: content should be json as we are sending an array of json objects
                hdrs["Content-Type"] = "application/json"
                if wtyp == const.EE:
                    try:
                        ijson= { "documents": [{"language":"en","id":i,"text":docs[i-1]} for i in range(1,len(docs)+1)] }
                        # get response from the server
                        resp = requests.post(url,headers=hdrs,json=ijson)
                        resp.raise_for_status()
                        # get json data to parse it later
                        js   = resp.json()
                        # all the lines from a page, including noise
                        for doc in js["documents"]:
                            keys = doc["keyPhrases"]
                            if not (len(keys) == 0):
                                ftext.append(keys)
                    except Exception as err:
                        ftext.append(str(err))
                else:
                    ftext.append("ERR: WRONG TYPE IN FIRST ARGUMENT")
            # clean array containing only important data
            for line in ftext:
                ret.append(line)
        else:
            ret  = [src,typ,key,host,url,hdrs,parms]
    return ret

############################################################################
##
## Purpose:  Read data from an array of PDF files
##
############################################################################
def ocre(imgs=[]):
    ret  = None
    if not (len(imgs) == 0):
        # number of cpu cores
        nc   = mp.cpu_count()
        # converted images
        ret  = Parallel(n_jobs=nc)(delayed(pil2array)(imgs[i]) for i in range(0,len(imgs)))
    return ret

############################################################################
##
## Purpose:  Read data from an array of PDF files
##
############################################################################
def cognitive(wtyp=const.OCR,pdfs=[],inst=const.BVAL,testing=True):
    ret  = None
    if not (wtyp == None or len(pdfs) == 0 or inst <= const.BVAL):
        # number of cpu cores
        nc   = mp.cpu_count()
        if wtyp == const.OCR:
            # converted images
            imgs =     Parallel(n_jobs=nc)(delayed(convert_from_path)(pdfs[i]                   ) for i in range(0,len(pdfs )))
            pimgs=     Parallel(n_jobs=nc)(delayed(ocre             )(imgs[i]                   ) for i in range(0,len(imgs )))
            oimgs=     Parallel(n_jobs=nc)(delayed(img2txt          )(wtyp,pimgs[i],inst,testing) for i in range(0,len(pimgs)))
            if not (len(oimgs) <= 1):
                ret  = Parallel(n_jobs=nc)(delayed(oimgs[0].append  )(oimgs[i]                  ) for i in range(1,len(oimgs)))
            else:
                if not (len(oimgs) == 0):
                    ret  = oimgs[0]
            ret  = [ret,cognitive(const.EE,ret,inst,testing)]
        else:
            ret  = img2txt(wtyp,pdfs,inst,testing)
    return ret

############################################################################
##
## Purpose:  Append together the text of a doc to an array
##
############################################################################
def dappend(doc=None):
    ret  = None
    if not (doc == None):
        if os.path.exists(doc) and os.path.getsize(doc) > 0:
            f    = open(doc)
            # should just be a line of text
            ret  = f.read()
            f.close()
    return ret

############################################################################
##
## Purpose:  Get the data from a pretrained GloVe model
##
############################################################################
def wvec(line=None):
    ret  = {}
    if not (line == None):
        # split the line into word and glove values ... return the result
        vals      = line.split()
        word      = vals[0]
        coefs     = np.asarray(vals[1:],dtype=np.dtype(float))
        ret[word] = coefs
    return ret

############################################################################
##
## Purpose:  Aid in parallel expansion of the data from a pretrained GloVe model
##
############################################################################
def expand(gm=[],pm=[],ind=const.BVAL):
    ret  = gm
    if not (len(pm) == 0):
        ret.append(pm)
    return ret

############################################################################
##
## Purpose:  Identify potential code block identifiers in a tokenized data set
##
############################################################################
def blocks(tok=None):
    ret  = {}
    # expecting that the corpus is already tokenized and integers are fit to the text
    if not (tok == None):
        # the idea is simple ... we will calculate some pseudo parameters for a
        # distribution and use them to identify potential code block demarcations
        #
        # we will use the median in place of the mean of the distribution of
        # word frequencies in a calculation of the standard deviation ... then
        # we will use >=4std, <4std & >=3std, <3std & >=2std, as the ranges to
        # demarcate potential code blocks in the tokenized data sets
        #
        # word counts
        cnts       = tok.word_counts.items()
        keys       = list(dict(cnts).keys())
        vals       = list(dict(cnts).values())
        # median of the counts
        med        = np.median(vals)
        # pseudo squared error
        serr       = [(vals[i]-med)**2 for i in range(0,len(keys))]
        # pseudo variance of the counts distribution
        varn       = sum(serr)/len(keys)
        # standard deviation of the counts distribution
        sd         = np.sqrt(varn)
        # attempt to identify class, interface, package blocks and
        # method definitions, as they should appear most infrequently
        ret["top"] = [word for word,cnt in cnts if                        cnt <= med - (4*sd)]
        # attempt to identify loops and conditionals
        ret["mid"] = [word for word,cnt in cnts if med - (4*sd) < cnt and cnt <= med - (3*sd)]
        # attempt to identify what's inside the blocks
        ret["bot"] = [word for word,cnt in cnts if med - (3*sd) < cnt and cnt <= med         ]
        # everything else
        ret["ete"] = [word for word,cnt in cnts if                        cnt >  med         ]
    return ret

############################################################################
##
## Purpose:  Implementation of Global Vectors for Word Representation (GloVe)
##           that will be used to extend a sparse data set of words
##
############################################################################
def extendglove(docs=[],gdoc=None,splits=2,props=2):
    gdat = {}
    if not (len(docs) == 0 or gdoc == None):
        if os.path.exists(gdoc) and os.path.getsize(gdoc) > 0:
            # number of cpu cores
            nc   = mp.cpu_count()
            # append the text of all docs together into a string
            txts = [dappend(d) for d in docs]
            # tokenize the lines in the array and convert to sequences of integers
            #
            # instantiate a default Tokenizer object without limiting the number of tokens
            tok  = Tokenizer()
            # tokenize the data
            tok.fit_on_texts(txts)
            # get the data from the pretrained GloVe word vectors
            # the data will consist of a word in the first position
            # which is followed by a sequence of integers
            f    = open(gdoc)
            gdat = {wvec(line).keys()[0]:wvec(line).values()[0] for line in f.readlines()}
            f.close()
            # we will pad the original sequences in a certain way so as to make them align with
            # the expanded set of words from the passed in GloVe data set then
            # use the word index to embed the document data into the pretrained GloVe word vectors
            #
            # each line in the glove data set is a set of constants such that when one line is dot product with
            # another line, we get the log of the probability that the second word appears in word-word
            # co-occurrence relationship with the first
            #
            # for our data files, we could compute similar numbers by fitting the texts to sequences of integers and
            # first computing the probability of word1 co-occurrence word2 as the product of the number of times word1
            # appears out of all words times number of times word2 appears out of all words in the corpus
            #
            # note that this leads to symmetry
            #
            # let's say that each row of the glove data file has N constants, preceded by an associated word so
            # that our goal for the combined data files is to obtain N constants for each word, such that their
            # mutual inner products result in the log probability that the words co-occur
            #
            # to get the N constants, take a unique lising of the words and the first word from the list and randomly
            # generate N constants to associate with the first word ... then take the second word and randomly generate
            # N-1 constants while choosing the last constants so that the dot product of both sets of constants gives
            # the log probability of co-occurrence that was already computed ... then take the 3rd word and randomly
            # generate N-2 constants while choosing the other 2 constants so that the dot products of this line and the
            # previous 2 lines gives the associated 2 log probabilities of co-occurrence that were previously computed ...
            # note that this is a system of 2 equations in 2 unknowns resulting in a unique solution or no solution ...
            # continue in this fashion until all M <= N words have the needed constants ... if there are only M words, then
            # we have the needed glove constants for our data files ... otherwise we can truncate our data set in the Tokenizer
            # call by setting the argument "num_words=M" ... or we can simply use the constants and probabilities from the first
            # M words in the corpus to build a supervised neural network that will essentially give an "average" set of
            # constants that work for the other N-M words that were left ... see the comments below for more insights
            #
            # see the glove function above
            #
            # instead, we will just rely upon the fact that the words that do appear in both our corpus and the glove data
            # carry information about those words that only appear in our corpus so that the average model being returned
            # still allows us to extend our original data set beyond the original corpus
            gmat = {word:gdat[word] for word,i in tok.word_index.items() if word in gdat.keys()}
            if not (gmat == {}):
                # gmat keys and values
                gkeys= list(gmat.keys())
                gvals= list(gmat.values())
                # number of clusters is different than the default defined by the splits and properties
                clust= len(gmat[list(gmat.keys())[0]])-1
                #
                s    = splits
                # we want to find the value p such that s**(2*p) gives the same number
                # of potential clusters as there are found in glove, as indicated by the number of constants
                # found in a row of gdat (minus one accounts for the word) ... only need to use the first line for counting
                p    = int(ceil(log(clust,s)/2.0))
                # we need values to turn into labels when training
                # one-hot encode the integer labels as its required for the softmax
                #
                # the goal is to build a model with clust=clust outputs such that when clust inputs are passed in, we get a
                # word in the glove data set associated to the clust constants ... furthermore, when the model constants are
                # used in dot product with the constants of any word from the glove data set, we get the average log probability
                # of the word from glove with any other arbitrary word from glove
                # 
                # now since every set of input constants is associated to a unique word from the glove data set, then our outputs
                # can consist of one unique value for each row of constants
                ovals= to_categorical(np.sum(vals,axis=1),num_classes=clust)
                # for each word in the glove files, we have a conditional distribution defined by constants as the parameters
                # of a plane ... i.e. each row of constants defines an element of a conditional specification
                #
                # words are associated with constants that can be dot product with constants of other words to obtain word-word
                # co-occurrence probabilities
                #
                # recall from the theory of random fields, that given a conditional specification, we seek a realizing
                # global probability distribution such that each of the global's conditionals, given a word in the glove
                # data set, is an element in the specification
                #
                # the DBN neural network can be used to find the global realizing distribution
                #
                # at this point, we have one-hot encoded each of the unique words in the glove data set that matched a word from
                # our corpus ... we use the constants of the conditional distributions in the glove data set as inputs to the DBN
                # that will find a distribution that has as its output the set of words that have been one-hot encoded ... i.e.
                # we will have found constants that can be used in dot product with constants of the conditionals to find the
                # average log probability of co-occurrence with any word appearing in both the corpus and glove data sets
                #
                # the result being that we will have found a set of constants (synaptic weights) of a network that identifies
                # the word used to define a conditional distribution in the specification of the realizing distribution .. or
                # more succinctly stated, with the neural network, we can identify if the inputs define
                # a set of synaptic weights for a distribution in the conditional specification
                #
                # after generating the model, we should have a set of constants (weights) that number the same as those
                # in the glove data file ... since we predict the word when using the associated constants of a word in the glove
                # data file, then we only need for the dot product of the constants from the model with those of the
                # constants in the data files to give us the log of the probability of the word-word co-occurrence
                #
                # however, the realizing distribution is a model of all words that gives you one particular word, when
                # a certain set of constants are used as inputs ... thus we can use the output of the model as the
                # global realizing distribution ... and the global distribution gives the mean log probability of co-occurrence
                # of any arbitrary word in the glove data set when dot product with the constants associated to another word
                #
                # input values to the model are the values associated to the words that appear in both our corpus and the glove data set
                ivals= np.asarray(vals)
                # create the model using the inputs
                model= dbn(ivals,ovals,splits=s,props=p,clust=clust)
                # for all words that don't appear in the glove data, we could just take the mean of the glove data for words that appear in
                # the corpus ... these words (and by extension, their values) carry information about the words that don't appear
                # in the glove data ... the mean will carry all information about these words as well
                #
                # however, we will instead generate a glove data set for these words, then use the global distribution (random field),
                # found from values of elements in the conditional specification associated to words that do appear in the corpus,
                # to predict the generated glove data set
                #
                # recall that the global distribution carries information about its marginals so that certain inputs will give a word
                # used to define an element of its conditional specification ... i.e. we get the values of the glove data set used
                # to build the global distribution ... this is exactly what we want ... start with a generated glove data set and
                # map this data set to elements of the conditional specification used to the build the global
                #
                # generate the glove data set
                gd   = glove(tok,clust+1)
                # predict the right values and add them to the output
                for word,i in tok.word_index.items():
                    if not (word in gkeys):
                        gdat[word] = model.predict(np.reshape(gd[word],(1,clust+1)))
    return gdat

############################################################################
##
## Purpose:   Define edges between vertices of a knowledge graph
##
############################################################################
def edges(clus=None,rows=[]):
    ret  = None
    if not (clus == None or len(rows) == 0):
        # number of cpu cores
        nc   = mp.cpu_count()
        # append the data point row numbers of those data points that are connected to the i-th data point in the current cluster
        #
        # have to sourround [x] so that extend doesn't separate the characters
        tret = Parallel(n_jobs=nc)(delayed(extend1)(rows[i],[[x] for j,x in enumerate(rows) if rows[j] != rows[i]]) for i in range(0,len(rows)))
        # append the cluster number
        ret  = Parallel(n_jobs=nc)(delayed(extend1)(clus,tret[i]) for i in range(0,len(tret)))
    return ret

############################################################################
##
## Purpose:   Create vertices and edges of a knowledge graph
##
############################################################################
def create_kg_ve(inst=const.BVAL,dat=[],lbls=[],lbl=None,ve=None):
    ret  = None
    if not (inst <= const.BVAL or len(dat) == 0 or len(lbls) == 0 or lbl == None or ve == None):
        # number of cpu cores
        nc   = mp.cpu_count()
        if ve == const.V:
            # only need to append the unique id defined by the row label to the data row
            # this is the set of vertices for each data point in the data set
            ret  = Parallel(n_jobs=nc)(delayed(extend)(str(inst)+'-'+lbl+'-'+lbls[i],dat[i]) for i in range(0,len(lbls)))
        else:
            # which cluster has been identified for storing the data
            clus = Parallel(n_jobs=nc)(delayed(split)(lbls[i]  ) for i in range(0,len(lbls)))
            ucs  = np.unique(clus)
            # get the row number of the original data point
            rows = Parallel(n_jobs=nc)(delayed(split)(lbls[i],1) for i in range(0,len(lbls)))
            # only need to extract the cluster label to go along with the brain label that was passed in
            # the edges will consist of the cluster label, brain label and connected data point pairs
            # this shows which data points are connected to form a cluster under the model in the current brain
            ret  = Parallel(n_jobs=nc)(delayed(edges)(ucs[i],[x for j,x in enumerate(rows) if clus[j] == ucs[i]]) for i in range(0,len(ucs)))
    return ret

############################################################################
##
## Purpose:   Heavy lift of creating a knowledge graph
##
############################################################################
def build_kg(inst,dat=[],brn={},splits=2):
    ret  = {const.V:[],const.E:[]}
    if not (inst == None  or
            inst < 0      or
            len(dat) == 0 or
            brn == {}     or
            splits < 2):
        # get the nn model for this brain
        mdl  = brn[const.MDL]
        # get the nn label for this brain
        lbl  = brn[const.LBL]
        # make the predictions using this model
        model= load_model(mdl)
        # make sure to get the right subset of the data
        l    = list(map(int,lbl.split("-")))
        # number of cpu cores
        nc   = mp.cpu_count()
        # make the predictions
        prds = model.predict(dat[:,l])
        #preds= to_categorical(np.sum(prds,axis=1),num_classes=splits**(2*len(l)))
        preds= Parallel(n_jobs=nc)(delayed(to_categorical)([j for j,x in enumerate(prds[i]) if x == max(prds[i])][0],num_classes=splits**(2*len(l))) for i in range(0,len(prds)))
        # generate the labels for the data
        lbls = label(preds)
        # create the vertices
        v    = create_kg_ve(inst,dat,lbls,lbl,const.V)
        # create the edges
        e    = create_kg_ve(inst,dat,lbls,lbl,const.E)
        ret[const.V] = v
        ret[const.E] = e
    return ret 

############################################################################
##
## Purpose:   Append vertices and edges to a knowledge graph
##
############################################################################
def append_kg(ret={},dat={}):
    v    = []
    e    = []
    if not (ret == {} or dat == {} or len(dat[const.V]) == 0 or len(dat[const.E]) == 0):
        # vertices
        rv   = ret[const.V]
        dv   = dat[const.V]
        if not (len(rv) == 0):
            v    = extend(rv,dv)
        else:
            v    = dv
        # edges
        re   = ret[const.E]
        de   = dat[const.E]
        if not (len(re) == 0):
            e    = extend(re,de)
        else:
            e    = de
    return {const.V:v,const.E:e}

############################################################################
##
## Purpose:   Create a knowledge graph
##
############################################################################
def create_kg(inst,dat=[],splits=2):
    ret  = {const.V:[],const.E:[]}
    if not (inst == None or inst < 0 or len(dat) == 0 or splits < 2):
        # number of cpu cores
        nc   = mp.cpu_count()
        # generate the brains
        brns = brain(dat)
        # generate the vertices and edges
        bret = Parallel(n_jobs=nc)(delayed(build_kg)(inst,dat,brn,splits) for brn in brns)
        rret = ret
        ret  = Parallel(n_jobs=nc)(delayed(append_kg)(rret,bret[i]) for i in range(0,len(bret)))
    return ret

# *************** TESTING *****************

def ai_testing(M=500,N=2):
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
    kg   = create_kg(0,ivals,s)
    print(kg[0])
    # test ocr
    o    = ocr(["/home/robert/data/files/kg.pdf"],0)
    print(o)
    # create column names (normally obtained by var.dtype.names)
    coln = {"col"+str(i):(i-1) for i in range(1,len(ivals[0])+1)}
    # test the thought function with the default number of predictions 3
    print(thought(inst,coln))
    # get the default configuration
    cfg  = config.cfg()
    # ordering of the data elements in the JSON file
    inst = 0
    src  = cfg["instances"][inst]["src"]["index"]
    typ  = cfg["instances"][inst]["src"]["types"]["glove"]
    # glove file
    gfl  = cfg["instances"][inst]["sources"][src][typ]["connection"]["file"]
    # test glove output
    g    = extendglove(["README.txt","README.txt"],gfl)
    leng = len(g)
    if leng <= 1000:
        print(g)
    else:
        print("GloVe: "+str(leng))
    print(permute(range(0,len(ivals[0]))))
    print(brain(ivals))
    imgs = convert_from_path("/home/robert/data/files/kg.pdf")
    print(imgs)
    for img in imgs:
        pil2 = pil2array(img)
        #print(pil2)
    src,typ,key,host,url,hdrs,parms = cognitive(docs=imgs,inst=0)
    print(src)
    print(typ)
    print(key)
    print(host)
    print(url)
    print(hdrs)
    print(parms)
    # we need values to turn into labels when training
    # one-hot encode the integer labels as its required for the softmax
    ovals= nn.categoricals(M,s,p)
    # generate the model for using the test values for training
    model = dbn(ivals,ovals,splits=s,props=p)
    if not (model == None):
        # generate some test data for predicting using the model
        ovals= np.random.sample(size=(m/10,p))
        # encode and decode values
        pvals= model.predict(ovals)
        # look at the original values and the predicted values
        print(ovals)
        print(pvals)
        print(label(pvals))
    else:
        print("Label model is null.")
    # test the data correction neural network function
    # output should be "robert", overwriting corrupt data and misclassifications
    bdat = ['robert','robert','robert','r0bert','rob3rt','r0b3rt','andre','murphy','murphy','Robert','R0bert','R0b3rt']
    rnd  = [np.random.randint(0,len(bdat))]
    for i in range(0,len(bdat)-1):
        rnd.append(np.random.randint(0,len(bdat)))
    bdat = np.asarray(bdat)[rnd]
    print(bdat)
    corr = correction(bdat)
    print(corr)
    # generate some random errors in my name to test the correction function
    bdat = ['robert' for i in range(0,m)]
    name = ['r','o','b','e','r','t']
    punc = [i for i in punctuation]
    for i in range(0,max(int(ceil(m/100)),1)):
        j    = np.random.randint(0,len(name))
        nm   = ''
        for k in range(0,len(name)):
            if not (j == k):
                nm   = nm + name[k]
            else:
                nm   = nm + punc[np.random.randint(0,len(punc))]
        bdat.append(nm)
    punc = ['q','w','e','r','t','y','u','i','o','p','a','s','d','f','g','h','j','k','l','z','x','c','v','b','n','m','1','2','3','4','5','6','7','8','9','0','Q','W','E','R','T','Y','U','I','O','P','A','S','D','F','G','H','J','K','L','Z','X','C','V','B','N','M']
    for i in range(0,max(int(ceil(m/100)),1)):
        j    = np.random.randint(0,len(name))
        nm   = ''
        for k in range(0,len(name)):
            if not (j == k):
                nm   = nm + name[k]
            else:
                nm   = nm + punc[np.random.randint(0,len(punc))]
        bdat.append(nm)
    rnd  = [np.random.randint(0,len(bdat))]
    for i in range(0,len(bdat)-1):
        rnd.append(np.random.randint(0,len(bdat)))
    bdat = np.asarray(bdat)[rnd]
    print(bdat)
    corr = correction(bdat)
    print(corr)
    bdat = bdat.tolist()
    # generate some random errors in my name to test the correction function
    for i in range(0,m*2):
        bdat.append('andre')
    #bdat = ['andre' for i in range(0,m)]
    name = ['a','n','d','r','e']
    punc = [i for i in punctuation]
    for i in range(0,max(int(ceil(m/50)),2)):
        j    = np.random.randint(0,len(name))
        nm   = ''
        for k in range(0,len(name)):
            if not (j == k):
                nm   = nm + name[k]
            else:
                nm   = nm + punc[np.random.randint(0,len(punc))]
        bdat.append(nm)
    punc = ['q','w','e','r','t','y','u','i','o','p','a','s','d','f','g','h','j','k','l','z','x','c','v','b','n','m','1','2','3','4','5','6','7','8','9','0','Q','W','E','R','T','Y','U','I','O','P','A','S','D','F','G','H','J','K','L','Z','X','C','V','B','N','M']
    for i in range(0,max(int(ceil(m/50)),2)):
        j    = np.random.randint(0,len(name))
        nm   = ''
        for k in range(0,len(name)):
            if not (j == k):
                nm   = nm + name[k]
            else:
                nm   = nm + punc[np.random.randint(0,len(punc))]
        bdat.append(nm)
    rnd  = [np.random.randint(0,len(bdat))]
    for i in range(0,len(bdat)-1):
        rnd.append(np.random.randint(0,len(bdat)))
    bdat = np.asarray(bdat)[rnd]
    print(bdat)
    corr = correction(bdat)
    print(corr)
    bdat = bdat.tolist()
    # generate some random errors in my name to test the correction function
    for i in range(0,m*4):
        bdat.append('murphy')
    #bdat = ['murphy' for i in range(0,m)]
    name = ['m','u','r','p','h','y']
    punc = [i for i in punctuation]
    for i in range(0,max(int(ceil(m/25)),5)):
        j    = np.random.randint(0,len(name))
        nm   = ''
        for k in range(0,len(name)):
            if not (j == k):
                nm   = nm + name[k]
            else:
                nm   = nm + punc[np.random.randint(0,len(punc))]
        bdat.append(nm)
    punc = ['q','w','e','r','t','y','u','i','o','p','a','s','d','f','g','h','j','k','l','z','x','c','v','b','n','m','1','2','3','4','5','6','7','8','9','0','Q','W','E','R','T','Y','U','I','O','P','A','S','D','F','G','H','J','K','L','Z','X','C','V','B','N','M']
    for i in range(0,max(int(ceil(m/25)),5)):
        j    = np.random.randint(0,len(name))
        nm   = ''
        for k in range(0,len(name)):
            if not (j == k):
                nm   = nm + name[k]
            else:
                nm   = nm + punc[np.random.randint(0,len(punc))]
        bdat.append(nm)
    rnd  = [np.random.randint(0,len(bdat))]
    for i in range(0,len(bdat)-1):
        rnd.append(np.random.randint(0,len(bdat)))
    bdat = np.asarray(bdat)[rnd]
    print(bdat)
    corr = correction(bdat)
    print(corr)
    bdat = bdat.tolist()
