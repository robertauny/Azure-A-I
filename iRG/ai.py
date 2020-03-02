#!/usr/bin/python

############################################################################
##
## File:      ai.py
##
## Purpose:   Other A-I and machine learning functions needed for iRG.
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Dec. 28, 2019
##
############################################################################

from joblib                       import Parallel, delayed
from itertools                    import combinations
from string                       import punctuation
from math                         import ceil, log

from keras.utils                  import to_categorical
from keras.models                 import load_model

from keras.preprocessing.text     import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from pdf2image                    import convert_from_bytes,convert_from_path
from PIL                          import Image

import requests
import io

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
        tdat = [j for j, x in enumerate(dat) if x == max(dat)]
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
def extend(dat1={},k=None,v=None):
    ret  = dat1
    if not (k == None):
        ret[k] = v
    return ret

############################################################################
##
## Purpose:   Extend a dictionary
##
############################################################################
def extend1(dat1={},dat2={}):
    # number of cpu cores
    nc   = mp.cpu_count()
    ret  = Parallel(n_jobs=nc)(delayed(extend)(dat1,k,dat2[k]) for k in dat2.keys())
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
def permute(dat):
    ret  = []
    sz   = len(dat)
    if not (sz == 0):
        # permute the array of indices beginning with the first element
        for j in range(0,sz+1):
            # all permutations of the array of indices
            jdat = dat[j:] + dat[:j]
            for i in range(0,sz):
                # only retain the sub arrays that are length >= 2
                tmp = [list(x) for x in combinations(jdat,i+2)]
                if len(tmp) > 0:
                    ret.extend(tmp)
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
            # generate the model
            mdl  = dbn(dat[:,perm],odat,splits=splits,props=p)
            # save the model to a local file
            fl   = "models/" + lbl + ".h5"
            mdl.save(fl)
            # add the current label and model to the return
            ret.append({"label":lbl,"model":fl})
    return ret

############################################################################
##
## Purpose:   Correct model for predicting future data as a service
##
############################################################################
def thought(inst=const.BVAL,lbl=None):
    ret  = []
    if not (inst <= const.BVAL or lbl == None):
        # assume that models are built so we will search the models
        # in the DB and return the one corresponding to the label 
        #
        # neural networks obtained from the DB later ... None for now
        df   = rw_kg(inst)
        nns  = df["nns"]
        for nn in nns:
            if lbl == nn["label"]:
                # load the model from the file path that is returned
                ret  = nn["model"]
                break
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
##
############################################################################
def img2txt(img=[],inst=const.BVAL,testing=True):
    ret  = []
    if not (len(img) == 0 or inst <= const.BVAL):
        # get the default configuration
        cfg  = config.cfg()
        # ordering of the data elements in the JSON file
        src  = cfg["instances"][inst]["src"]["index"]
        typ  = cfg["instances"][inst]["src"]["types"]["ocr"]
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
        hdrs = {"Ocp-Apim-Subscription-Key":key,"Content-Type":"application/octet-stream"}
        # request parameters: language is unknown, and we do detect orientation
        parms= {"language":"unk","detectOrientation":"true"}
        if not testing:
            # get response from the server
            resp = requests.post(url,headers=hdrs,params=parms,data=img)
            resp.raise_for_status()
            # get json data to parse it later
            json = resp.json()
            # all the lines from a page, including noise
            ftext= []
            for reg in json["regions"]:
                line = reg["lines"]
                for elem in line:
                    ltext = " ".join([word["text"] for word in elem["words"]])
                    ftext.append(ltext.lower())
            # clean array containing only important data
            for line in ftext:
                ret.append(line)
        else:
            ret  = [src,typ,key,host,url,hdrs,parms]
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
        ret  = [1.0/float(ord(x)) for i, x in enumerate(d)]
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
        rret = [" "] * csz
        for i in range(0,csz):
            udat    = "".join(unique(cdat[:,cols[i]]))
            ndat    = numbers(udat,len(udat))
            mmn     = abs(ndat-np.full(len(ndat),mn[i]))
            j       = [k for k,x in enumerate(mmn) if x == min(mmn)][0]
            rret[i] = udat[j]
        ret  = "".join(rret).lstrip()
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
        # change this to use all 3 predictions combined someway
        mdat = mostly(dat,rdat,cols,preds,pre)
        ret  = "".join(mdat).lstrip()
        if not (ret in dat):
            mdat = Parallel(n_jobs=nc)(delayed(max)(set(cdat[:,i].tolist()),key=cdat[:,i].tolist().count) for i in range(0,pre))
            ret  = "".join(mdat).lstrip()
            if not (ret in dat):
                ret  = almost(dat)
    return ret

############################################################################
##
## Purpose:   Use DBN to correct a list of corrupt or misclassified strings
##
############################################################################
def correction(dat=[]):
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
        perms= permute(range(0,ssz))
        lmax = sys.maxint
        # initial entropy is something higher than otherwise possible
        cdt  = np.asarray(Parallel(n_jobs=nc)(delayed(chars)(dat[i],ssz) for i in range(0,sz)))
        cdat = [cdt[i,0].lower() for i in range(0,len(cdt)) if cdt[i,0].isalpha()]
        # lower bound on the number of clusters to seek has to be >= 2 as at least one error is assumed
        #lo   = len(np.unique(cdat))
        lo   = 2
        # we should sample at least as many data elements as there are clusters
        ind  = [np.random.randint(0,len(ndat)) for i in range(0,max(lo,int(ceil(0.1*len(ndat)))))]
        for perm in perms:
            beg  = perm[0]
            end  = perm[len(perm)-1]
            if (beg < end and np.array_equal(range(beg,end+1),perm)):
                pdat = ndat[:,perm]
                ptdat= [sum(x)/(len(x)*max(x)) for x in pdat]
                pydat= np.asarray(ptdat)
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
                psdat= model.predict(pdat)
                updat= unique(psdat)
                hi   = len(updat)
                if np.array_equal(perm,perms[0]):
                    cols = perm
                    lmax = hi
                    tdat = psdat
                    udat = {max(x):i for i,x in enumerate(updat)}
                else:
                # ********************* compute current entropy somehow and make it a condition
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
def ocr(pdfs=[],inst=const.BVAL,testing=True):
    ret  = None
    if not (len(pdfs) == 0 or inst <= const.BVAL):
        # number of cpu cores
        nc   = mp.cpu_count()
        # converted images
        imgs =     Parallel(n_jobs=nc)(delayed(convert_from_path)( pdfs[i]             ) for i in range(0,len(pdfs )))
        pimgs=     Parallel(n_jobs=nc)(delayed(ocre             )( imgs[i]             ) for i in range(0,len(imgs )))
        oimgs=     Parallel(n_jobs=nc)(delayed(img2txt          )(pimgs[i],inst,testing) for i in range(0,len(pimgs)))
        if not (len(oimgs) <= 1):
            ret  = Parallel(n_jobs=nc)(delayed(oimgs[0].append  )(oimgs[i]             ) for i in range(1,len(oimgs)))
        else:
            ret  = oimgs
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
## Purpose:  Implementation of Global Vectors for Word Representation (GloVe)
##           that will be used to extend a sparse data set of words
##
############################################################################
def glove(docs=[],gdoc=None,splits=2,props=2):
    model= None
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
            # generate sequences of integers from the texts
            seqs = tok.texts_to_sequences(txts)
            # the sequences will be of different lengths because the original docs
            # were almost surely of different lengths ... so pad the sequences, which
            # essentially appends the sequences together and tries to produce a square matrix
            pdat = pad_sequences(seqs)[0]
            # get the data from the pretrained GloVe word vectors
            # the data will consist of a word in the first position
            # which is followed by a sequence of integers
            gd   = {}
            f    = open(gdoc)
            gdat = {wvec(line).keys()[0]:wvec(line).values()[0] for line in f.readlines()}
            #gdat = Parallel(n_jobs=nc)(delayed(extend1)(gd,wvec(line)) for line in f.readlines())
            f.close()
            # we will pad the original sequences in a certain way so as to make them align with
            # the expanded set of words from the passed in GloVe data set then
            # use the word index to embed the document data into the pretrained GloVe word vectors
            #
            # each line in the glove data set is a set of constants such that when one line is dot product with
            # another line, we get the log of the probability that the second word appears in word-word
            # co-occurrence relationship with the first
            #
            # for our data files, we could compute similar numbers using pdat by first computing the probability of word1
            # co-occurrence word2 as the product of the number of times word1 appears out of all words times
            # number of times word2 appears out of all words in the corpus ... note that this leads to symmetry
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
            # instead, we will just rely upon the fact that the words that do appear in both our corpus and the glove data
            # carry information about those words that only appear in our corpus so that the average model being returned
            # still allows us to extend our original data set beyond the original corpus
            gmat = {word:gdat[word] for word,i in tok.word_index.items() if word in gdat.keys()}
            if not (gmat == {}):
                # number of clusters is different than the default defined by the splits and properties
                clust= len(gmat[gmat.keys()[0]])-1
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
                ovals= to_categorical(np.sum(gmat.values(),axis=1),num_classes=clust)
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
                ivals= np.asarray(gmat.values())
                # create the model using the inputs
                model= dbn(ivals,ovals,splits=s,props=p,clust=clust)
                # for all words that don't appear in the glove data, just take the mean of the glove data for words that appear in
                # the corpus ... these words (and by extension, their values) carry information about the words that don't appear
                # in the glove data ... the mean will carry all information about these words as well
                mvals= np.mean(ivals,axis=0).reshape((1,len(ivals[0])))
                # get the values associated with the mean
                mmat = model.predict(mvals)
                # add to our glove dictionary all words that appear in the corpus but not currently in the dictionary
                for word,i in tok.word_index.items():
                    if not (word in gmat.keys()):
                        gdat[word] = mmat
    return gdat

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
    # test ocr
    o    = ocr(["files/kg.pdf"],0)
    print(o)
    # test glove output
    g    = glove(["README.txt","README.txt"],"data/glove.6B.50d.txt")
    #print(g)
    print(permute(range(0,len(ivals[0]))))
    print(brain(ivals))
    imgs = convert_from_path("files/kg.pdf")
    print(imgs)
    for img in imgs:
        pil2 = pil2array(img)
        #print(pil2)
    src,typ,key,host,url,hdrs,parms = img2txt(imgs,0)
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