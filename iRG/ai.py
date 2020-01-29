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
    if not (sz == 0 or pre < 0):
        if not (just == 0):
            e    = dat.rjust(pre)
        else:
            e    = dat.ljust(pre)
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
    if not (sz == 0 or pre < 0):
        d    = chars(dat,pre)
        ret  = [1.0/float(ord(x)) for i, x in enumerate(d)]
    return ret

############################################################################
##
## Purpose:   Which string in a list appears most (by whole words)
##
############################################################################
def mostly(dat=[],pre=0):
    ret  = None
    sz   = len(dat)
    if not (sz == 0 or pre < 0):
        # number of cpu cores
        nc   = mp.cpu_count()
        # numeric representations of the strings in the list
        rdat = Parallel(n_jobs=nc)(delayed(numbers)(dat[i].lower(),pre) for i in range(0,sz))
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
        rdat = np.asarray(rdat)
        model= dbn(rdat
                  ,np.full(len(rdat),1.0)
                  ,loss='mean_squared_error'
                  ,optimizer='sgd'
                  ,rbmact='sigmoid'
                  ,dbnact='sigmoid'
                  ,dbnout=1)
        preds= model.predict(rdat)
        inds = [j for j,x in enumerate(preds) if x == max(preds)]
        # mean of all rows giving predictions close to the plane with bias 1
        mn   = np.mean(rdat[inds],axis=0)
        # character-wise append of characters in each feature of the list of strings
        cdat = Parallel(n_jobs=nc)(delayed(chars)(dat[i],pre) for i in range(0,sz))
        cdat = np.asarray(cdat)
        # for the final return, take the characters closest to the mean
        rret = [" "] * pre
        for i in range(0,pre):
            udat    = "".join(unique(cdat[:,i]))
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
def most(dat=[],pre=0):
    ret  = None
    sz   = len(dat)
    if not (sz == 0 or pre < 0):
        # number of cpu cores
        nc   = mp.cpu_count()
        # character-wise append of most frequent characters in each feature of the list of strings
        cdat = Parallel(n_jobs=nc)(delayed(chars)(dat[i],pre) for i in range(0,sz))
        cdat = np.asarray(cdat)
        mdat = Parallel(n_jobs=nc)(delayed(max)(set(cdat[:,i].tolist()),key=cdat[:,i].tolist().count) for i in range(0,pre))
        ret  = "".join(mdat).lstrip()
        if not (ret in dat):
            mdat = mostly(dat,pre)
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
        #
        # compute the values
        #
        # the outputs force the hierarchical classification model to 
        # produce regressors to segregate the data that amount to
        # auto-encoders, as we are using all of the input data elements
        # in the definition of the outputs
        #tdat = [x[len(x)-1]/max(x) for i, x in enumerate(ndat)]
        #tdat = [sum(x)/(len(x)*max(x)) for i, x in enumerate(ndat)]
        #tdat = [max(x) for i,x in enumerate(ndat)]
        tdat = [sum(x[0:1]) for i,x in enumerate(ndat)]
        udat = {x:i for i,x in enumerate(unique(tdat))}
        # we need values to turn into labels when training
        # one-hot encode the numeric data as its required for the softmax
        #
        # splits
        s    = 2
        #
        # properties
        #p    = len(ndat[0])
        p    = int(ceil(log(len(udat),s)/2.0))
        #
        # number of classes, one-hot encodings
        ncls = s**(2*p)
        cdat = [to_categorical(udat.get(x),num_classes=ncls) for x in tdat]
        odat = np.asarray(cdat)
        ndat = np.asarray(ndat)
        # generate the model of the data
        #ind  = [np.random.randint(0,len(ndat)) for i in range(0,int(ceil(0.1*len(ndat))))]
        #model= dbn(ndat[ind],odat[ind],splits=s,props=p)
        model= dbn(ndat,odat,splits=s,props=p)
        # predict the same data to get the labels
        preds= model.predict(ndat)
        # get the labels
        lbls = label(preds)
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
            ind      = [j for j,x in enumerate(clus) if x == cls]
            idat     = [dat[x] for j,x in enumerate(ind)]
            ret[ind] = most(idat,ssz)
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
def dappend(ret=[],doc=None):
    if not (ret == None or doc == None):
        if os.path.exists(doc) and os.path.getsize(doc) > 0:
            f    = open(doc)
            # should just be a line of text
            ret.append(f.read())
            f.close()

############################################################################
##
## Purpose:  Get the data from a pretrained GloVe model
##
############################################################################
def wvec(ret={},line=None):
    if not (ret == None or line == None):
        vals = line.split()
        word = vals[0]
        coefs= np.asarray(vals[1:],dtype='float32')
        ret[word] = coefs

############################################################################
##
## Purpose:  Aid in expansion of the data from a pretrained GloVe model
##
############################################################################
def expand(seqi=[],seqo=None,inds=[]):
    ret  = None
    if not (seqo == None):
        ret  = seqo
    else:
        if not (len(seqi) == 0 or len(inds) == 0):
            arr             = {j:seqi[x] for j,x in enumerate(inds) if not (x == None)}
            ret             = np.zeros(len(inds))
            ret[arr.keys()] = arr.values()
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
            # append the text of all docs together into an array, one line per file
            txt  = []
            txts = Parallel(n_jobs=nc)(delayed(dappend)(txt,docs[i]) for i in range(0,len(docs)))
            # tokenize the lines in the array and convert to sequences of integers
            #
            # instantiate a default Tokenizer object without limiting the number of tokens
            tok  = Tokenizer()
            # tokenize the data
            toks = tok.fit_on_texts(txts)
            # generate sequences of integers from the texts
            seqs = tok.texts_to_sequences(txts)
            # the sequences will be of different lengths because the original docs
            # were almost surely of different lengths ... so pad the sequences
            pdat = pad_sequences(seqs)
            # get the data from the pretrained GloVe word vectors
            # the data will consist of a word in the first position
            # which is followed by a sequence of integers
            gd   = {}
            f    = open(gdoc)
            gdat = Parallel(n_jobs=nc)(delayed(wvec)(gd,line) for i,line in f.read())
            f.close()
            # we will pad the original sequences in a certain way so as to make them align with
            # the expanded set of words from the passed in GloVe data set
            #
            # the set of indices that align with the GloVe data set will dictate how to pad
            # the word index consisting of each unique word and the integer to which it is mapped
            # inds will look something like [1, 0, 0, 4, 2, 3, 3, None, 3, 2, 0, None, 0, ...]
            # where the length of inds matches the number of words in the expanded data set
            tmp  = {word:i for i,word in enumerate(tok.word_index.items())}
            inds = [tmp.get(word) for word in gdat.keys()]
            # use the word index to embed the document data into the pretrained GloVe word vectors
            gmat = Parallel(n_jobs=nc)(delayed(expand)(pdat[i],gdat.get(word),inds) for i,word in tok.word_index.items())
            # we need values to turn into labels when training
            # one-hot encode the integer labels as its required for the softmax
            ovals= categoricals(inds,splits,props)
            # for all of the indices in inds that are not None, we should have a corresponding seq of ints in gmat
            # we will substitute those values into the gdat array
            model= dbn(gmat,ovals,splits=splits,props=props)
    return model

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
    print(permute(range(0,len(ivals[0]))))
    print(brain(ivals))
    imgs = convert_from_path("files/kg.pdf")
    print(imgs)
    #for img in imgs:
        #print(pil2array(img))
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
    corr = correction(unique(bdat))
    print(corr)
    # generate some random errors in my name to test the correction function
    bdat = ['robert' for i in range(0,m)]
    name = ['r','o','b','e','r','t']
    punc = [i for i in punctuation]
    for i in range(0,min(m/10,5)):
        j    = np.random.randint(0,len(name))
        nm   = ''
        for k in range(0,len(name)):
            if not (j == k):
                nm   = nm + name[k]
            else:
                nm   = nm + punc[np.random.randint(0,len(punc))]
        bdat.append(nm)
    punc = ['q','w','e','r','t','y','u','i','o','p','a','s','d','f','g','h','j','k','l','z','x','c','v','b','n','m','1','2','3','4','5','6','7','8','9','0','Q','W','E','R','T','Y','U','I','O','P','A','S','D','F','G','H','J','K','L','Z','X','C','V','B','N','M']
    for i in range(0,min(m/10,5)):
        j    = np.random.randint(0,len(name))
        nm   = ''
        for k in range(0,len(name)):
            if not (j == k):
                nm   = nm + name[k]
            else:
                nm   = nm + punc[np.random.randint(0,len(punc))]
        bdat.append(nm)
    corr = correction(bdat)
    print(corr)
    # generate some random errors in my name to test the correction function
    for i in range(0,m):
        bdat.append('andre')
    #bdat = ['andre' for i in range(0,m)]
    name = ['a','n','d','r','e']
    punc = [i for i in punctuation]
    for i in range(0,min(m/10,5)):
        j    = np.random.randint(0,len(name))
        nm   = ''
        for k in range(0,len(name)):
            if not (j == k):
                nm   = nm + name[k]
            else:
                nm   = nm + punc[np.random.randint(0,len(punc))]
        bdat.append(nm)
    punc = ['q','w','e','r','t','y','u','i','o','p','a','s','d','f','g','h','j','k','l','z','x','c','v','b','n','m','1','2','3','4','5','6','7','8','9','0','Q','W','E','R','T','Y','U','I','O','P','A','S','D','F','G','H','J','K','L','Z','X','C','V','B','N','M']
    for i in range(0,min(m/10,5)):
        j    = np.random.randint(0,len(name))
        nm   = ''
        for k in range(0,len(name)):
            if not (j == k):
                nm   = nm + name[k]
            else:
                nm   = nm + punc[np.random.randint(0,len(punc))]
        bdat.append(nm)
    corr = correction(bdat)
    print(corr)
    # generate some random errors in my name to test the correction function
    for i in range(0,m):
        bdat.append('murphy')
    #bdat = ['murphy' for i in range(0,m)]
    name = ['m','u','r','p','h','y']
    punc = [i for i in punctuation]
    for i in range(0,min(m/10,5)):
        j    = np.random.randint(0,len(name))
        nm   = ''
        for k in range(0,len(name)):
            if not (j == k):
                nm   = nm + name[k]
            else:
                nm   = nm + punc[np.random.randint(0,len(punc))]
        bdat.append(nm)
    punc = ['q','w','e','r','t','y','u','i','o','p','a','s','d','f','g','h','j','k','l','z','x','c','v','b','n','m','1','2','3','4','5','6','7','8','9','0','Q','W','E','R','T','Y','U','I','O','P','A','S','D','F','G','H','J','K','L','Z','X','C','V','B','N','M']
    for i in range(0,min(m/10,5)):
        j    = np.random.randint(0,len(name))
        nm   = ''
        for k in range(0,len(name)):
            if not (j == k):
                nm   = nm + name[k]
            else:
                nm   = nm + punc[np.random.randint(0,len(punc))]
        bdat.append(nm)
    corr = correction(bdat)
    print(corr)
