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
## File:      nn.py
##
## Purpose:   Neural network logic.
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Nov. 22, 2021
##
############################################################################

import sys

import constants as const

import utils

ver  = sys.version.split()[0]

if ver == const.constants.VER:
    from            keras.layers                            import Dense,BatchNormalization,Activation,Conv2D,Conv2DTranspose,Add,Input,Subtract,ReLU,Layer,Flatten
    from            keras.models                            import Sequential,load_model,Model,clone_model
    from            keras.utils.np_utils                    import to_categorical
    from            keras.callbacks                         import ModelCheckpoint,Callback
    from            keras.preprocessing.text                import Tokenizer
    from            keras                                   import backend
    from            keras.applications                      import DenseNet121
else:
    from tensorflow.keras.layers                            import Dense,BatchNormalization,Activation,Conv2D,Conv2DTranspose,Add,Input,Subtract,ReLU,Layer,Flatten
    from tensorflow.keras.models                            import Sequential,load_model,Model,clone_model
    from tensorflow.keras.utils                             import to_categorical
    from tensorflow.keras.callbacks                         import ModelCheckpoint,Callback
    from tensorflow.keras.preprocessing.text                import Tokenizer
    from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
    from tensorflow.keras                                   import backend
    from tensorflow.keras.applications                      import DenseNet121

from joblib                      import Parallel,delayed
from math                        import log,ceil,floor,sqrt,inf
from itertools                   import combinations
from contextlib                  import closing
from matplotlib.pyplot           import imread,savefig
from datetime                    import datetime
from PIL                         import Image
from scipy                       import stats
from glob                        import glob 

import multiprocessing               as mp
import numpy                         as np
import pandas                        as pd
import tensorflow                    as tf
import tensorflow_model_optimization as tfmot
import xml.etree.ElementTree         as ET

import os

# current datetime
dt   = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

np.random.seed(const.constants.SEED)
tf.random.set_seed(const.constants.SEED)

############################################################################
##
## Purpose:   Define the outputs for a classification
##
############################################################################
def categoricals(rows=const.constants.BVAL,splits=const.constants.SPLITS,props=const.constants.PROPS,clust=const.constants.BVAL):
    ret  = []
    if not (rows <= const.constants.BVAL or ((splits < const.constants.SPLITS or props < const.constants.PROPS) and clust <= const.constants.BVAL)):
        # we need values to turn into labels when training
        # one-hot encode the integer labels as its required for the softmax
        s    = splits
        p    = props
        if not (clust <= const.constants.BVAL):
            nc   = clust
        else:
            nc   = s**(2*p)
        ret  = [np.random.randint(1,nc) for i in range(0,rows)]
        ret  = to_categorical(ret,num_classes=nc)
    return ret

############################################################################
##
## Purpose:   Calculate the number of clusters to form using random clusters
##
############################################################################
def calcN(pts=None):
    ret = 2**2
    if (pts != None and type(pts) == type(0)):
        M    = ceil(sqrt(pts))
        for i in range(2,const.constants.MAX_FEATURES+1):
            N    = max(floor(sqrt(i)),2)
            if pts/(pts+(2*M*((N-1)^2))) <= 0.5:
                ret  = N**2
            else:
                break
    return ret

############################################################################
##
## Purpose:   Random cluster theory for formation
##
############################################################################
def calcC(vals=None,clust=None,keys=None):
    ret  = None
    # can't use categoricals for the labels here
    # throws an error for the list of things in the first column being longer than the number of labels
    # yet this is ok for this application since all words in our corpus
    # that's captured by the GloVe model bave non-zero probability of co-occurrence
    # thus all words connect in one cluster (random field theory and markov property)
    # which will be the case for floating pint labels sent to to_categorical
    #
    # because of the single brain (single model) forced by the last parameter to create_kg
    # all other columns in the data set will be used to model the single first column
    #
    # the only other things is to change the behavior of which data points are connected
    # to which and this will involve a little higher order probability theory which
    # states that we have full connectivity, i.e. one cluster if each node in a network
    # connects to O(logN) of its neighbors, where N is the total number of nodes
    #
    # the first column can be turned into categoricals by considering the theory that allows for choosing the number of clusters.
    # Then, by separability, the marginal of the first column's cluster distribution can be learned using a DBN where categorical
    # labels (using the number of clusters) as the output
    #
    # order of the categoricals are found as such
    #
    # sort the first column to obtain the sort order indices ... calculate the number of clusters and divide the data uniformly
    # with labels represented in the right proportion ... uniformity is legitimate by the Central Limit Theorem giving an
    # assumption of normality of the first column, but we order it, which allows us to assume a beta distribution whose parameters
    # give uniformity by other arguments from the same theory that allows calculation of the number of clusters ... then, get the reverse
    # sort order and apply it to the labels and this is the label ordering for use as the outputs with original first column as inputs to the DBN
    if not (type(vals) == type(None) or len(vals) == 0):
        svals= np.argsort(vals)
        #rints= list(range(1,clust if clust is not None and clust >= 2 else calcN(len(vals))+1))
        rints= list(range(0,clust if clust is not None and clust >= 2 else calcN(len(vals))))
        sints= int(ceil(len(vals)/len(rints)))
        tvals= []
        for i in rints:
            tvals.extend([[i] for j in range((i-1)*sints,min(i*sints,len(vals)))])
        ret  = np.asarray([[int(i[0])] for i in np.asarray(tvals)[np.argsort(svals)]])
        # if the caller wants the key/value pairs then return them
        if keys is not None and type(keys) == type({}):
            # the integer values for the categoricals
            v    = np.asarray(tvals)[np.argsort(svals)]
            # vals and svals are the same length, but vals are in the original ordering
            for i in range(0,len(vals)):
                # dictionary is unique
                if vals[i] not in keys:
                    keys[vals[i]] = int(v[i][0])
    return ret

############################################################################
##
## Purpose:   Deep belief network support class for residuals
##
############################################################################
class ResLayer(Layer):
    def __init__(self,enc,**kwargs):
        super(ResLayer,self).__init__(**kwargs)
        self.enc = enc

    def call(self):
        nenc = Dense(list(self.enc.shape)[1],input_shape=self.enc.shape,activation='relu' if ver == const.constants.VER else 'selu')(self.enc)
        model= Model(inputs=self.enc,outputs=nenc)
        return model.layers[len(model.layers)-1]

    def compute_output_shape(self,input_shape):
        return input_shape

############################################################################
##
## Purpose:   Deep belief network support function for clustering
##
############################################################################
def clustering(mdl=None,clust=0):
    model= mdl
    if not (type(mdl) == type(None) or clust <= 0):
        wghts= tfmot.clustering.keras.cluster_weights
        cent = tfmot.clustering.keras.CentroidInitialization
        parms= {'number_of_clusters': clust, 'cluster_centroids_init': cent.LINEAR}
        # cluster a model
        model= wghts(mdl,**parms)
        # use smaller learning rate for fine-tuning clustered model
        opt  = tf.keras.optimizers.Adam(learning_rate=const.constants.BASE_LR if hasattr(const.constants,"BASE_LR") else 1e-5)
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                     ,optimizer=opt
                     ,metrics=['accuracy'])
    return model

############################################################################
##
## Purpose:   Deep belief network support function
##
############################################################################
def dbnlayers(model=None,outp=const.constants.OUTP,shape=None,act=None,useact=False):
    if not (type(model) == type(None) or outp < const.constants.OUTP or shape == None or type(act) == type(None)):
        if type(shape) in [type(0),type([]),type(np.asarray([])),type(())]:
            shp  = (shape,) if type(shape) == type(0) else shape
            # auto-encode the input data using the rectified linear unit
            enc1 = Dense(outp,input_shape=shp,activation='relu')
            # add the input layer to the model
            model.add(enc1)
            if not useact:
                # encode the input data using the rectified linear unit
                enc  = Dense(outp,input_shape=shp,activation=act)
                # add the input layer to the model
                model.add(enc)
            else:
                # encode the input data using the rectified linear unit
                enc  = Dense(outp,input_shape=shp,use_bias=False)
                # add the input layer to the model
                model.add(enc)
                # add batch normalization
                enc  = BatchNormalization()
                # add the input layer to the model
                model.add(enc)
                # add the activation
                enc  = Activation(act)
                # add the input layer to the model
                model.add(enc)
            # calculate the residuals
            inp1 = Subtract()([enc1.output,enc.output])
            # auto-encode the input data using the linear unit
            enc1 = Dense(outp,input_shape=shp,activation='relu' if ver == const.constants.VER else 'selu')
            # add the input layer to the model
            model.add(enc1)
            # add the residuals to the filtered output from before we started
            enc  = ResLayer(Add()([inp1,enc1.output])).call()
            # add the input layer to the model
            model.add(enc)
    return

############################################################################
##
## Purpose:   Deep belief network for classification or regression
##
############################################################################
def dbn(inputs=[]
       ,outputs=[]
       ,sfl=None#const.constants.SFL
       ,splits=const.constants.SPLITS
       ,props=const.constants.PROPS
       ,clust=const.constants.BVAL
       ,loss=const.constants.LOSS
       ,optimizer=const.constants.OPTI
       ,rbmact=const.constants.RBMA
       ,dbnact=const.constants.DBNA
       ,dbnout=const.constants.DBNO
       ,epochs=const.constants.EPO
       ,embed=const.constants.EMB
       ,encs=const.constants.ENCS
       ,useact=const.constants.USEA
       ,verbose=const.constants.VERB
       ,callbacks=None):
    model= None
    if inputs.any():
        # linear stack of layers in the neural network (NN)
        model= Sequential()
        # add dense layers which are just densely connected typical artificial NN (ANN) layers
        #
        # at the input layer, there is a tendency for all NNs to act as Gabor filters
        # where there's an analysis of the content of the inputs to determine whether there is
        # any specific content in any direction of the multi-dimensional inputs ... i.e. a Gabor filter
        # is a feature selector
        #
        # at the onset, the input level and second level match
        # first sibling level matches the number of features
        # all are deemed important and we will not allow feature selection as a result
        # this follows the writings in "Auto-encoding a Knowledge Graph Using a Deep Belief Network"
        #
        # the number of features, M, should be a function of len(inputs[0,0]) and the size of the kernel
        # as we will be building the model using the flattened kernels as inputs and reconstructing the input
        # data sets after retrieving the output from the model
        #
        # the kernel size will be determined using random cluster theory that won't be explained here, but
        # simply involves the projection of multidimensional data to a 2D space, while preserving class membership
        # of each data point ... the projection results in uniformly sized partitions in the 2D space, giving
        # the dimension of the kernel ... these kernels define the maximal region of disjoint subsets of
        # of data points, given the number of rows in the data set, such that all points in each disjoint
        # set are determinative of each other's states, according to random field theory ... this is precisely
        # what we want to model when filtering the data set to reveal its underlying equilibrium distribution
        #
        # recall that a distribution of data which exhibits the Markov property can be modeled as the sum of
        # a deterministic lower dimensional subspace plus additive noise ... it is the noise that we seek to filter
        # thereby revealing the underlying subspace
        ip   = inputs
        op   = outputs
        S    = splits
        # add dense layers which are just densely connected typical artificial NN (ANN) layers
        #
        # at the input layer, there is a tendency for all NNs to act as Gabor filters
        # where there's an analysis of the content of the inputs to determine whether there is
        # any specific content in any direction of the multi-dimensional inputs ... i.e. a Gabor filter
        # is a feature selector
        #
        # at the onset, the input level and second level match
        # first sibling level matches the number of features
        # all are deemed important and we will not allow feature selection as a result
        # this follows the writings in "Auto-encoding a Knowledge Graph Using a Deep Belief Network"
        #
        # control the size of the network's inner layers
        #
        # if all possible levels, then M  = len(inputs[0])
        if type(ip[0]) in [type([]),type(np.asarray([]))]:
            if type(ip[0,0]) in [type([]),type(np.asarray([]))]:
                M    = min(len(ip[0,0]),props)
            else:
                M    = min(len(ip[0  ]),props)
        else:
            M    = props
        # add other encodings that are being passed in the encs array
        if not (type(encs) == type(None)):
            if not (len(encs) == 0):
                for enc in encs:
                    model.add(enc)
        else:
            # inputs have M columns and any number of rows, while output has M columns and any number of rows
            #
            # encode the input data using the rectified linear unit
            dbnlayers(model,M,ip.shape[1:],'relu',useact)
        # if M > const.constants.MAX_FEATURES, then we will embed the inputs in a lower dimensional space of dimension const.constants.MAX_FEATURES
        #
        # embed the inputs into a lower dimensional space if M > min(const.constants.MAX_FEATURES,props)
        if embed:
            p    = min(const.constants.MAX_FEATURES,props)#if not op.any() else min(const.constants.MAX_FEATURES,min(props,op.shape[len(op.shape)-1]))
            if M > p:
                dbnlayers(model,p,M,'relu' if ver == const.constants.VER else 'selu',useact)
                M    = min(max(1,int(ceil(log(p,S)/2.0))),p)
                # project the outputs into the lower dimensional subspace
                #op   = np.asarray(op)[:,:(S**(2*p))]
        # add the rest of the layers according to the writings
        #
        # output dimension (odim) is initially S (number of splits in first level of hierarchy)
        odim = S
        # the loop variable counts the number of restricted boltzmann machines (RBM)
        # that are defined by the 2 extra layers that are added to the model
        # note that we should always have M RBMs, one for each property of the data
        #
        # calculation of a solution to clust = S ** (2 * M) for (2 * M)
        #for J in range(0,M):
        itr  = max(1,2*M-1 if hasattr(const.constants,"BVAL") and clust == const.constants.BVAL else int(ceil(log(clust,S)))-1)
        for J in range(0,itr):
            # the dimensionality will be computed in the loop as what's specified in
            # the writings works well for binary splits and 2 properties, but the loop
            # can be generalized to make the computation a bit better
            #
            # S^((J-2)/2)*M^(J/2)     for sibling levels in the writings
            # S^((J-1)/2)*M^((J-1)/2) for binary  levels in the writings
            #
            # for the sibling layers, the input is the old output dimension
            if J == 0:
                dim  = odim
            else:
                #dim  = const.constants.MAX_DIM if hasattr(const.constants,"MAX_DIM") and S * dim > const.constants.MAX_DIM else S * dim
                dim  = S * dim
            # next layers using the scaled exponential linear unit (Gibbs distribution) are siblings
            dbnlayers(model,dim,odim,'relu' if ver == const.constants.VER else 'selu',useact)
            # redefine the input and output dimensions for the binary layers
            #odim = const.constants.MAX_DIM if hasattr(const.constants,"MAX_DIM") and S * odim > const.constants.MAX_DIM else S * odim
            odim = S * odim
            # next layers using the softmax are binary layers
            #
            # note that by using dense layers of ANNs we can use back propagation within each RBM and across
            # all RBMs to tune the weights in each layer so that the hidden layers have a structure
            # that results in the correct number of clusters in the output layer giving rise to
            # the hierarchy that is detailed in the writings and the slide deck ... without this tuning
            # through back propagation, there is no guarantee that the hidden layers' structure will match
            # what is expected
            #
            # also, Gibbs sampling is being performed in the hidden layers so that edges are open
            # and closed to give rise to the correct cluster structure in the hidden layers defined above
            # so that the tuning through back propagation leads to the equilibrium distribution that can be color
            # coded into distinct regions of connected clusters ... see the writings for an example
            if not (J == itr - 1):
                #dbnlayers(model,odim,dim,'sigmoid',useact)
                dbnlayers(model,odim,dim,rbmact,useact)
            else:
                if not (type(dbnact) == type(None) or dbnout <= 0):
                    dbnlayers(model,odim,dim,'relu' if ver == const.constants.VER else 'selu',useact)
                else:
                    # add another layer to change the structure of the network if needed based on clusters
                    if not (clust <= 0):
                        dbnlayers(model,clust,dim,rbmact,useact)
                    else:
                        dbnlayers(model,odim ,dim,rbmact,useact)
        # add another layer for a different kind of model, such as a regression model
        if not (type(dbnact) == type(None) or dbnout <= 0):
            # preceding layers plus this layer now perform auto encoding
            dbnlayers(model,M,odim,'relu' if ver == const.constants.VER else 'selu',useact)
            # requested model at the output layer of this RBM
            dbnlayers(model,dbnout,M   ,dbnact   ,useact)
        # optimize using the typical categorical cross entropy loss function with root mean square optimizer to find weights
        model.compile(loss=loss,optimizer=optimizer)
        x    = ip.astype(np.single)
        y    = op.astype(np.single)
        if not (type(sfl) == type(None)):
            # construct the relaxed file name
            #
            # file name minus the type extension
            fln  = sfl[:sfl.rfind(".") ]
            # file type extension
            flt  = ".hdf5"
            # model checkpoint call back
            nsfl = fln + const.constants.SEP + dt + flt
            chkpt= ModelCheckpoint(filepath=nsfl,save_weights_only=False,monitor='val_acc',mode='max')#,save_best_only=True)
            # we will allow for 100 iterations through the training data set to find the best sets of weights for the layers
            # fit the model using the flattened inputs and outputs
            vpct = 1.0 - (const.constants.TRAIN_PCT if hasattr(const.constants,"TRAIN_PCT") else 0.8)
            if vpct * len(x) <= 1:
                # we will allow for 100 iterations through the training data set to find the best sets of weights for the layers
                # fit the model using the flattened inputs and outputs
                if ver == const.constants.VER:
                    model.fit(x=x,y=y,nb_epoch =epochs,verbose=verbose)
                else:
                    model.fit(x=x,y=y,   epochs=epochs,verbose=verbose)
            else:
                if ver == const.constants.VER:
                    model.fit(x=x,y=y,nb_epoch =epochs,verbose=verbose,callbacks=[chkpt],validation_split=vpct)
                else:
                    model.fit(x=x,y=y,   epochs=epochs,verbose=verbose,callbacks=[chkpt],validation_split=vpct)
                # file type extension
                flt  = sfl[ sfl.rfind("."):]
                # save the model
                nsfl = fln + const.constants.SEP + dt + flt
                model.save(nsfl)
        else:
            # we will allow for 100 iterations through the training data set to find the best sets of weights for the layers
            # fit the model using the flattened inputs and outputs
            if ver == const.constants.VER:
                model.fit(x=x,y=y,nb_epoch =epochs,verbose=verbose,callbacks=callbacks)
            else:
                model.fit(x=x,y=y,   epochs=epochs,verbose=verbose,callbacks=callbacks)
    # return the model to the caller
    return model

############################################################################
##
## Purpose:   Generative network composed of DBNs
##
############################################################################
def gan(inputs=[]
       ,outputs=[]
       ,sfl=None#const.constants.SFL
       ,splits=const.constants.SPLITS
       ,props=const.constants.PROPS
       ,clust=const.constants.BVAL
       ,loss=const.constants.LOSS
       ,optimizer=const.constants.OPTI
       ,rbmact=const.constants.RBMA
       ,dbnact=const.constants.DBNA
       ,dbnout=const.constants.DBNO
       ,epochs=const.constants.EPO
       ,embed=const.constants.EMB
       ,encs=const.constants.ENCS
       ,useact=const.constants.USEA
       ,verbose=const.constants.VERB):
    class LossHistory(Callback):
        def on_train_begin(self,logs={}):
            self.losses = []
        def on_batch_end(self,batch,logs={}):
            self.losses.append(logs.get('loss'))
    hist = LossHistory()
    # placeholder for now until the methodology of the GAN is defined
    model= dbn(inputs
              ,outputs
              ,sfl
              ,clust=clust
              ,callbacks=[hist])
    return model

############################################################################
##
## Purpose:   Deconstruct the data
##
############################################################################
def nn_decon(ivals=None,tvals=None,m=const.constants.MAX_ROWS,p=const.constants.PROPS,s=const.constants.SPLITS):
    ret  = None,None,0,0,0
    if ((type(ivals) == type([]) or type(ivals) == type(np.asarray([]))) and len(ivals) > 1) and \
       ((type(tvals) == type([]) or type(tvals) == type(np.asarray([]))) and len(tvals) > 1) and \
       (m >= const.constants.MAX_ROWS and s >= const.constants.SPLITS and p >= const.constants.PROPS):
        # later the permutations in the colums that are taken together by separability
        # due to the Markov property, will allow us a way to stochastically relax the entire dataset
        # while also not overly taxing the resources of the local machine
        #
        # keep the data at a manageable size for processing
        #
        # handle columns first
        #
        # the number of datasets from excessive columns
        blksc= int(ceil(p/const.constants.MAX_COLS))
        kp   = p
        remc = kp % const.constants.MAX_COLS
        p    = const.constants.MAX_COLS
        s    = p
        imgs = []
        timgs= []
        for blk in range(0,blksc):
            if blk < blksc - 1:
                imgs.append(np.asarray(ivals)[:,range(blk*p,(blk+1)*p)])
            else:
                e    = (blk+1)*p if blk < blksc-1 else min((blk+1)*p,kp)
                if remc == 0:
                    imgs.append(np.asarray(ivals)[:,range(blk*p,(blk+1)*p)])
                else:
                    # pad the datasets with a zero strip to make the dimensions match ... cols in 2nd arg, no rows
                    imgs.append(np.pad(np.asarray(ivals)[:,range(blk*p,e)],[(0,0),(0,const.constants.MAX_COLS-remc)],'constant',constant_values=const.constants.FILL_COLOR))
            timgs.append(list(tvals))
        ivals= np.asarray( imgs)
        tvals= np.asarray(timgs)
        # now handle rows
        #
        # the number of datasets from excessive rows
        blksr= int(ceil(m/const.constants.MAX_ROWS))
        km   = m
        remr = km % const.constants.MAX_ROWS
        m    = const.constants.MAX_ROWS
        imgs = []
        timgs= []
        for i in range(0,len(ivals)):
            for blk in range(0,blksr):
                if blk < blksr - 1:
                    imgs.append(np.asarray(ivals[i])[range(blk*m,(blk+1)*m),:])
                    timgs.append(tvals[i][range(blk*m,(blk+1)*m)])
                else:
                    e    = (blk+1)*m if blk < blksr-1 else min((blk+1)*m,km)
                    if remr == 0:
                        imgs.append(np.asarray(ivals[i])[range(blk*m,(blk+1)*m),:])
                        timgs.append(tvals[i][range(blk*m,(blk+1)*m)])
                    else:
                        # adding rows
                        imgs.append(np.pad(np.asarray(ivals[i])[range(blk*m,e),:],[(0,const.constants.MAX_ROWS-remr),(0,0)],'constant',constant_values=const.constants.FILL_COLOR))
                        # one-dimensional so adding characters
                        timgs.append(tvals[i][range(blk*m,e)])
                        timgs[-1] = np.append(timgs[-1],np.full(const.constants.MAX_ROWS-remr,timgs[-1][len(timgs[-1])-1]))
                timgs[-1] = list(timgs[-1])
        ivals= np.asarray( imgs)
        # need to reshape for the network to use these outputs
        shp  = list(np.asarray(timgs).shape)
        shp.extend([1])
        tvals= np.reshape(timgs,shp)
        ret  = ivals,tvals,m,p,s
    return ret

############################################################################
##
## Purpose:   Reconstruct the data
##
############################################################################
def nn_recon(vals=None,m=const.constants.MAX_ROWS,p=const.constants.PROPS):
    ret  = None
    if ((type(vals) == type([]) or type(vals) == type(np.asarray([]))) and len(vals) > 1) and \
       (m >= const.constants.MAX_ROWS and p >= 1):
        # first do the blocks of rows
        #
        # the number of datasets from excessive rows
        blksr= int(ceil(m/const.constants.MAX_ROWS))
        km   = m
        ivals= list(vals)
        if not (blksr < 1):
            rblks= int(len(ivals)/blksr)
            rblks= max(rblks,1)
            for blk in range(0,rblks):
                ivals[blk*blksr] = list(ivals[blk*blksr])
                for i in range(1,blksr):
                    ival = ivals[blk*blksr]
                    ival.extend(ivals[(blk*blksr)+i])
                    if not (i < blksr - 1):
                        if not (km % const.constants.MAX_ROWS == 0):
                            ival = np.asarray(ival)[range(0,len(ival)-(const.constants.MAX_ROWS-(km%const.constants.MAX_ROWS))),:]
                    ivals[blk*blksr] = ival
            ival = []
            for i in range(0,int(len(ivals)/blksr)):
                if len(ivals) % blksr == 0:
                    ival.append(ivals[i*blksr])
            ivals= np.asarray(ival)
        # now do the blocks of cols
        #
        # the number of datasets from excessive columns
        blksc= int(ceil(p/const.constants.MAX_COLS))
        kp   = p
        ivals= list(ivals)
        if not (blksc < 1):
            cblks= int(len(ivals)/blksc)
            cblks= max(cblks,1)
            for blk in range(0,cblks):
                ivals[blk*blksc] = list(ivals[blk*blksc])
                for i in range(1,blksc):
                    ival = np.append(ivals[blk*blksc],ivals[(blk*blksc)+i],axis=1)
                    if not (i < blksc - 1):
                        if not (kp % const.constants.MAX_COLS == 0):
                            ival = np.asarray(ival)[:,range(0,len(ival[0])-(const.constants.MAX_COLS-(kp%const.constants.MAX_COLS)))]
                    ivals[blk*blksc] = ival
            ret  = []
            for i in range(0,int(len(ivals)/blksc)):
                if len(ivals) % blksc == 0:
                    ret.append(ivals[i*blksc])
            ret  = np.asarray(ret)
    return ret

############################################################################
##
## Purpose:   Main data wrapper for the neural network logic
##
############################################################################
def nn_dat(dat=None,tgt=1):
    # process a single dataset or a list of datasets being passed in
    ret  = {"ivals":None,"rivals":None,"tvals":None,"m":0,"p":0,"s":0}
    # number of cpu cores for multiprocessing
    nc   = const.constants.CPU_COUNT if hasattr(const.constants,"CPU_COUNT") else mp.cpu_count()
    if ((type(dat) == type([]) or type(dat) == type(np.asarray([]))) and len(dat) > 1) and (tgt > 0):
        # inputs assume that the last T columns are targets
        #
        # data is scaled to represent black/white pixels
        ivals= dat[:,:-tgt] * 255
        shape= list(ivals.shape)
        # the targets
        tvals= dat[:, -tgt]
        # without going into a lot of detail here, using a result based upon the random cluster model
        # we can estimate the number of classes to form as by assuming that we have N^2 classes containing
        # a total of M^2 data points, uniformly and evenly distributed throughout a bounded uniformly
        # partitioned unit area in the 2D plane
        #
        # then the probability of connection for any 2 data points in the bounded region is given by
        # M^2 / (M^2 + (2M * (N-1)^2)) and by the random cluster model, this probability has to equate
        # to 1/2, which gives a way to solve for N, the minimum number of clusters to form for a
        # data set consistinng of M^2 Gaussian distributed data points projected into 2D space
        #
        # each class contains M^2/N^2 data points ... the sqrt of this number is the size of the kernel we want
        #
        # note that the number of data points is M^2 in the calculations, different than M below
        kM   = max(2,int(floor(np.float32(min(shape))/np.float32(calcN(min(shape))))))
        # replace kMxkM portions of the data set with uniformly distributed image values
        #
        # generate the kM^2 values
        vals = np.random.randint(0,255,kM*kM).reshape((kM,kM))
        # all kMxKM regions will have the same set of values represented
        # while the boundary of the region will be the original image values
        # this is the new data set that will be passed to nn_dat for smoothing
        # and recovery of the image to test the theory
        rows = int(floor(np.float32(shape[0])/np.float32(kM+1)))
        cols = int(floor(np.float32(shape[1])/np.float32(kM+1)))
        for i in range(0,rows):
            r    = range(i*(kM+1),i*(kM+1)+kM)
            for j in range(0,cols):
                c    = range(j*(kM+1),j*(kM+1)+kM)
                for k in range(0,kM):
                    for m in range(0,kM):
                        ivals[r[0]+k,c[0]+m] = vals[k,m]
        # number of data points, properties and splits
        m    = len(ivals)
        om   = m
        p    = len(ivals[0])
        op   = p
        # only one feature or property for the analysis
        # just return None ... can't do anything
        if not (m >= 2 or p >= 2):
            return ret
        s    = p
        os   = s
        # deconstruct the data into blocks for management
        # of the memory in lean times
        ivals,tvals,m,p,s = nn_decon(ivals,tvals,m,p,s)
        # the next steps are allowable as a result of the random cluster model and separable stochastic processes
        #
        # Use the permutations function to get sequential permutations of length MAX_FEATURES
        # Create a new data structure to hold the data for each feature indicated by the permutations.
        # Train the model using the created data structure then predict using the same data structure.
        # Modify the original data structure using whats been predicted.
        #
        # permutations of integer representations of the features in the dataset
        perm = utils.utils._permute(list(range(0,p)),False,min(p,const.constants.MAX_FEATURES))
        perms= []
        for j in range(0,len(perm)):
            if len(perm[j]) == min(len(ivals[0,0]),const.constants.MAX_FEATURES) and \
               list(perm[j]) == list(np.sort(list(range(min(perm[j]),max(perm[j])+1)))):
                if j == 0:
                    perms.append(list(perm[j]))
                else:
                    if not list(perm[j]) in list(perms):
                        perms.append(list(perm[j]))
        # the new data structure to hold the dataset in permutations of length MAX_FEATURES
        #
        # the idea is that the local structure determined by the Markov process and separability
        # allows us to only consider neighbors of each pixel ... as such, we will learn the local
        # structure as determined by the permutations of features and predict the most probable color of each center pixel
        # by considering the probability of each neighboring color ... here, we only create the structure
        # and train the model ... the probabilities will be obtained in a prediction by auto encoding the inputs
        # with the proviso that we are not looking to obtain outputs from the underlying subspace of the inputs
        # rather we will have the same number of outputs as inputs, only the outputs will be probabilities
        # indicating the likelihood of the color of the center pixel based upon the probabilities of colors of its neighbors
        nivals  = []
        for i in range(0,len(ivals)):
            for j in range(0,len(perms)):
                # length of the input feature space has to be consistent
                # also, we want only contiguous integer feature columns, e.g. 0,1,2 or 3,4,5, etc.
                nivals.append(ivals[i][:,perms[j]])
        nivals= np.asarray(nivals)
        # don't want to use a linear model for dbnact as that assumes that the
        # data are linearly separable, which results in 2 classes by default
        # as such, the resulting linear subspace will present values fairly close
        # to one another (we are filtering noise after all) and that is not desirable
        # in the long run ... thus we will use the sigmoid instead
        #
        # may also want to try a softmax with categorical cross entropy
        #
        # did some testing with other loss/optimizer/dbnact combinations
        # turns out that softmax and sigmoid in the final layer force
        # normalized values between 0 and 1 for the output
        #
        # returning to linear and will use the greatest changes between values in
        # the smoothed data and the original data as areas to concentrate upon
        # when identifying regions of interest
        #
        # after more testing it looks like a classification configuration will work best
        # in that we will get a probability for each color in each row of pixels which
        # is an indication of which color is more probable ... this is useful as we can
        # relax the data by changing all colors to the most probable one
        #
        # also, we can make use of permuations on each column sequence to make predictions
        # on the whole set of columns using the Markov property and separable processes
        #
        # construct the relaxed data name
        sfl  = const.constants.SFL if hasattr(const.constants,"SFL") else "models/obj.h5"
        #model = dbn(np.asarray([ivals]),np.asarray([ivals]),splits=s,props=p)
        model= dbn(np.asarray(nivals)
                  ,np.asarray(nivals)
                  ,sfl=sfl
                  ,splits=s
                  ,props=p
                  ,loss=const.constants.LOSS
                  ,optimizer=const.constants.OPTI
                  #,rbmact='relu'
                  #,dbnact='relu' if ver == const.constants.VER else const.constants.RBMA
                  ,rbmact='selu'
                  ,dbnact='sigmoid'
                  ,dbnout=p)
        assert(type(model) != type(None))
        pvals= model.predict(np.asarray(nivals).astype(np.single))
        # *********************************
        # replace the original data here and compute the differences between
        # the modified original data and the original data to identify
        # the regions of interest
        #
        # also note that by breaking the data up into a smaller number of features and
        # loading the data into the neural netowrk we are capturing the local characteristics
        # guaranteed by the Markov property and encoding those characteristics and changed states
        # for use afterward when other parts of the bounded region undergo the same relaxation process
        # *********************************
        rivals= ivals.copy()
        def rivals_func(i):
            for j in range(0,len(perms)):
                for k in range(0,len(rivals[i])):
                    # local receptive fields will be square of size len(perms)
                    # each partition of the data will have len(perms) columns
                    # the partitions are stacked on top of each other in the ordering of the perms
                    # so dividing by the number of data, we have all of the partitions for each data in stacked groupings
                    # we need the right stack and partition for each data to get the right receptive field for replacement
                    idx                  = list(pvals[len(perms)*i+j,k,range(0,len(perms[j]))]).index(max(pvals[len(perms)*i+j,k,range(0,len(perms[j]))]))
                    # copy the most probable color to all pixels in the kth row of the local receptive field
                    rivals[i,k,perms[j]] = np.full(len(perms[j]),ivals[i,k,perms[j][idx]])
            return None
        if nc > 1:
            dump = Parallel(n_jobs=nc)(delayed(rivals_func)(i) for i in range(0,len(rivals)))
        else:
            dump = []
            for i in range(0,len(rivals)):
                dump.append(rivals_func(i))
        ret["ivals" ] = nn_recon(ivals ,om,op)
        ret["rivals"] = nn_recon(rivals,om,op) / 255
        ret["tvals" ] = tvals
        ret["m"     ] = m
        ret["p"     ] = p
        ret["s"     ] = s
    return ret

############################################################################
##
## Purpose:   Do some data management of string categoricals
##
############################################################################
def nn_mgmt(dat=None):
    ret  = dat
    if (type(dat) == type([]) or type(dat) == type(np.asarray([]))):
        ret  = dat.copy()
        # columns that are strings
        for i in [j for j in range(0,len(ret[0])) if type("") in [type(ret[k,j]) for k in range(0,len(ret[:,j]))]]:
            nret     = ["".join(key.split()) for key in np.asarray(list(ret[:,i]))]
            tok      = Tokenizer(num_words=len(nret)+1)
            tok.fit_on_texts(nret)
            items    = dict(list(tok.word_counts.items()))
            # data is scaled to be binary
            # later it will be scaled to be black/white pixels
            ret[:,i] = [int(np.round(float(items[key.lower()])/float(max(list(items.values()))))) for key in nret]
    return ret

############################################################################
##
## Purpose:   Balance the data set
##
############################################################################
def nn_balance(dat=None,tgt=1):
    ret  = dat
    if (type(dat) == type([]) or type(dat) == type(np.asarray([]))) and tgt > 0:
        ret  = pd.DataFrame(dat.copy())
        # all targets
        tgts = ret.iloc[:,-tgt].to_numpy()
        # most frequently appearing target
        mtgt = utils.utils._most(tgts)
        # list of unique targets
        utgts= list(utils.utils._unique(tgts))
        # number of rows associated to the most frequently appearing target
        atgt = [j for j,t in enumerate(tgts) if t == mtgt]
        ltgt = len(atgt)
        # balance the data
        for i in [j for j in utgts if not j == mtgt]:
            l    = [k for k,t in enumerate(tgts) if t == i]
            if len(l) < ltgt:
                ret  = ret.append(ret.iloc[list(np.random.choice(l,ltgt-len(l))),:])
        ret  = ret.to_numpy()
    return ret

############################################################################
##
## Purpose:   Define the outputs for a classification
##
############################################################################
def nn_cat(dat=None,splits=const.constants.SPLITS,props=const.constants.PROPS,clust=const.constants.BVAL):
    ret  = None
    if (type(dat) is not None) and \
       (splits >= const.constants.SPLITS and props >= const.constants.PROPS and clust > const.constants.BVAL):
        # we need values to turn into labels when training
        # one-hot encode the integer labels as its required for the softmax
        s    = splits
        p    = min(const.constants.MAX_FEATURES,props)
        if not (clust <= const.constants.BVAL):
            nc   = clust
        else:
            nc   = s**(2*p)
        ret  = to_categorical(dat,nb_classes=nc) if ver == const.constants.VER else to_categorical(dat,num_classes=nc)
    return ret

############################################################################
##
## Purpose:   Merge the data and split into train and test sets
##
############################################################################
def nn_split(pfl=None,mfl=None):
    ret  = {"train":None,"test":None,"labels":{}}
    if type(pfl) == type(pd.DataFrame([])):
        # get the first input values
        dat  = pfl.copy()
        if type(mfl) in [type([]),type(np.asarray([]))]:
            for m in mfl:
                if type(m) == type(pd.DataFrame([])):
                    # merge the data
                    dat  = dat.merge(m,how="inner").fillna(0)
        else:
            if type(mfl) == type(pd.DataFrame([])):
                # merge the data
                dat  = dat.merge(mfl,how="inner").fillna(0)
        # drop certain columns if requested
        if hasattr(const.constants,"DROP"):
            cols = [c for c in const.constants.DROP if c in list(dat.columns)]
            dat  = dat.drop(columns=cols) if not len(cols) == 0 else dat
        # move target columns to the end if requested
        if hasattr(const.constants,"TARGETS") and (type(const.constants.TARGETS) == type([]) or type(const.constants.TARGETS) == type(np.asarray([]))):
            cols         = dat.columns.tolist()
            for i in range(0,len(cols)):
                if cols[i] in const.constants.TARGETS:
                    # we will sub labels for the target values
                    ucls                   = utils.utils._unique(dat[cols[i]]).flatten()
                    cls                    = {int(ucls[i]):i for i in range(0,len(ucls))}
                    dat[cols[i]]           = [cls[dat.iloc[j,i]] for j in range(0,len(dat.iloc[:,i]))]
                    ret["labels"][cols[i]] = cls
        # and change any date formats to integers
        if hasattr(const.constants,"DATES") and (type(const.constants.DATES) == type([]) or type(const.constants.DATES) == type(np.asarray([]))):
            cols         = dat.columns.tolist()
            for i in range(0,len(cols)):
                if cols[i] in const.constants.DATES:
                    if hasattr(const.constants,"DTFORMAT"):
                        dat[cols[i]] = datetime.datetime.strptime(dat.iloc[:,i],const.constants.DTFORMAT)
                    else:
                        dat          = dat.drop(columns=cols[i])
        # get a train and test data set
        tpct         = const.constants.TRAIN_PCT if hasattr(const.constants,"TRAIN_PCT") else 0.8
        trow1        = np.random.randint(0,len(dat),int(ceil(tpct*len(dat))))
        ret["train"] = dat.iloc[trow1,:].to_numpy().copy()
        trow2        = [j for j in range(0,len(dat)) if j not in trow1]
        ret["test" ] = dat.iloc[trow2,:].to_numpy().copy()
    return ret

############################################################################
##
## Purpose:   Random field similarity measure for 2 vectors
##
############################################################################
def nn_similarity(vec1=[],vec2=[]):
    # at this point things get really interesting
    #
    # until now we have relied upon the states (labels) to
    # determine which neighboring sites are correlated with each
    # center was we proceed through the NxN bounded region but
    # without those labels we still need a way to determine correlation
    #
    # recall that there is a canonical projection of the higher dimensional
    # data into the 2-D bounded region, with each of the vectors at sites
    # in the 2-D bounded region having an orientation and spin, which is what
    # can be used to determine correlation
    #
    # given any 2 vectors, there's a plane containing those vectors and we
    # can determine their relative correlation by measuring the angle between
    # the 2 vectors
    # 
    # recall that the cosine of the angle between 2 vectors is proportional to
    # the dot product ... thus we can compute their dot product and use it as the
    # argument to the arccosine to get the angle between
    #
    # now the closer the angle is to being 0 or 180 degrees, the greater the correlation
    # as such we will consider vectors to be correlated if the angle gebtween them is
    # within 45 degrees of 0 or 180 ... i.e. between
    #
    # 0 and 45 ... 135 and 180 ... 180 and 225 ... 315 and 360
    #
    # now note also that in random field theory, the boundary of the 2-D region contains all
    # of the information about sites external to the bounded region and even tho we have exactly
    # one realizing distribution whenever all sites external to the bounded region have one state
    # from the set of all possible states (without regard to which state open or closed is used)
    # then in the case of a set of data with a dominant class where most sites external to the bounded
    # region have states from the dominant class, then we likely have a realizing global distribution
    # that can be found with a deep learner
    #
    # in addition, sites external to the bounded region have the same state but it is still
    # worth while to learn its distribution since each row of data corresponding to a site
    # has a spin and orientation that can be learned in association with a corresponding label
    # for predicting what's external to the bounded region
    #
    # furthermore, when learning the conditional distribution of states in the bounded region
    # given states on external sites, the bounded region is fully connected almost surely
    # which is not to say that all sites in the bounded region have the same state, but we are
    # able to find a path of connected sites such that we can traverse the bounded region almost surely
    #
    # as such, if all sites external to the bounded region have the same state almost surely (say 0 = closed)
    # and the bounded region can be traversed, then the interior of the bounded region contains sites with
    # states (1 = open) almost surely ... and this is what we modeled when finding the bounded region using labels
    #
    # it is now clear that to model a distribution of data with a dominant class and minority class then we
    # require 2 models, one to recognize what spin and orientation constitutues a zero state and the other to model
    # the interior of a bounded region that can contain any state (0 or 1, with predominantly 1) since we can traverse
    # the bounded region almost surely
    # the interior labels
    ret  = False
    if (type(vec1) in [type([]),type(np.asarray([]))] and len(vec1) > 0) and \
       (type(vec2) in [type([]),type(np.asarray([]))] and len(vec2) > 0):
        pi   = np.arccos(-1.0)
        sim  = np.arccos(np.dot(vec1,vec2))
        if (sim >= 0.0        and sim <      pi/4.0) or \
           (sim >= 3.0*pi/4.0 and sim <      pi    ) or \
           (sim >=     pi     and sim <  5.0*pi/4.0) or \
           (sim >  7.0*pi/4.0 and sim <= 2.0*pi    ):
            ret  = True
    return ret

############################################################################
##
## Purpose:   Trim the data using random field/cluster theory
##
############################################################################
def nn_trim(dat=[],labels=None,label=None,order=False):
    # devising a way using mathematics to limit the dominant class of an
    # imbalanced data set so that models show greater sensitivity when
    # predicting the label of an inferior class
    #
    # these methods only work if the data are ordered and assumed uniformly
    # distributed, as that is the typical set up in a random field (special
    # case of the random cluster model)
    #
    # order is certainly the case in a timeseries and with an imbalanced data
    # set, we can model the data with a poisson distribution with parameter
    # equal to the average frequency the inferior event appears in the overall
    # data set
    #
    # furthermore, since the uniform distribution is the limiting distribution
    # of a sequence of (poisson) distributed random variables, then an ordered
    # timeseries fits into the assumptions of the set up
    #
    # likewise, in the case of a more generalized data set, we can make an
    # appeal to the central limit theorem to make an assumption of normality
    # of the originally unordered data set, after which we can apply an ordering
    # by row with feature ordering unchanged ... with this ordering, we can assume
    # the beta distribution and parameters requiring the new distribution to be
    # uniform since the random field theory provides a canonical mapping into
    # the 2-D plane with preservation of q spatially uniformly distributed clusters
    #
    # in both cases, we can assume the canonical mapping into 2-D space so that
    # each higher dimensional row from the data set is represented by a 2-D vector
    # with class representation maintained
    #
    # each mapped data row has a corresponding state represented by some real value
    # but this is not our concern at the moment since the values figure into an
    # energy function that forms the argument of an exponential function that models
    # the density of the distribution, but this energy function will be learned by
    # the deep learner after the data rows have been embedded into some finite dimensional
    # euclidean space before the canonical projection is applied resulting in 2-D vectors
    #
    # rather, our concern at this point is to jump ahead in the process after embedding
    # and configure the size of the 2-D space for the following reason that comes from
    # random cluster theory
    #
    # we seek the smallest set of sites in 2-D space that encompass all positive values
    # from the inferior class so that all other values outside the bounds of the inferior
    # class are from the dominant class ... in such a case, it makes no difference if
    # the values from outside the bounds are actually all from the dominant class label
    # or the inferior class label, since the conditional distribution so defined in either
    # case leads to the unique limiting distribution (which is what we seek) when the
    # data set grows increasingly larger with the bound growing larger as well
    #
    # all 2-D vectors outside the bound can be discarded as a result of either label
    # being sufficient, which comports with the 2-D Markov property where only
    # rows mapped inside the bound are required to learn the limiting distribution
    #
    # as such, we can discard all rows from the original data set that correspond to
    # vectors outside the bound
    #
    # first we devise the entire 2-D space required for all data rows after mapping
    # which can be obtained by simply finding the smallest number whose square is not less
    # than the number of rows in the data set
    #
    # to find the bound, we compute the row containing the first and last rows corresponding
    # to the first and last labels from the inferior class
    if type(dat) == type(np.asarray([])) and len(dat) > 0:
        ret  = pd.DataFrame(dat)
        # if the data should be sorted as requested
        if order:
            ret  = ret.sort_values(by=list(ret.columns))
        # calculate the size of the NxN 2-D space
        #
        # what allows us to take the 1-D series and make it into
        # a 2-D NxN random field is the Markov property whereby
        # the 1-D Markov chain is embeddable into a Brownian motion
        # that amounts to a harmonic sum of sines and cosines with Fourier
        # coefficients ... each term in the harmonic sum has a periodicity
        #
        # this actuality is borne out by the canonical map of the assumed
        # gaussian distributed higher dimensional data set into the 2-D
        # bounded region while preserving class membership
        #
        # in addition, the number of clusters and the map forces uniformity
        # in the NxN range space of the canonical map
        #
        # calculate the size of the NxN 2-D space
        N    = int(sqrt(len(ret)))
        # perfect square
        Nlbls= N**2
        if (type(labels) == type(0) and labels >= 0 and labels <= len(ret[0])) and \
           (type(label ) == type(0) and label  >= 0                          ):
            # find the first and last label from the inferior class
            #
            # if label is provided as something that does not appear in
            # the list of labels, then the entire data set will undergo
            # this random field reduction
            albls= [j for j,x in enumerate(ret[labels]) if x == label]
            lbls = [min(albls),max(albls)] if not len(albls) == 0 else [0,len(ret)]
            # adjust the size of the bound if this conditional is satisfied
            if lbls[0] > 0 and lbls[1] < len(ret):
                # once we have the first and last labels, we need to balance
                # out the bound so that we have a square sub-region contained in the
                # original NxN 2-D space
                #
                # due to invariance (shifting the bound horizontally or vertically), we
                # don't need to be concerned whether the bound is centered in the 2-D space
                #
                # as such we can just calculate the number of rows obtained by the difference
                # in lbls values and add 1 to the square root if the difference is not a
                # perfect square
                N    = int(sqrt(lbls[1]-lbls[0]))
                # the number of rows corresponding to vectors in our 2-D bound is the square of Nlbls
                Nlbls= N**2
                # make an adjustment to the label ranges to form our data set range of rows
                #
                # note ... by uniformity and translation invariance we can limit the size even further
                # the ending labels corresponding rows that should be included in our data set
                # reset the lbls start and end row numbers for creating the bound
                lbls[1] = min(lbls[0]+Nlbls,len(ret))
            # now we will use random field theory to reduce the size of the data set
            # in essence, we will find and remove all labels in the dominant class
            # that are fully connected to each of its neighbors while paying close
            # attention to those labels that exist in the boundary of the bounded region
            #
            # for each label not in the inferior class identified by "label", check its
            # neighbors in the NxN bounded region to see if the same state exists on those
            # neighboring sites ... if yes, then we can delete the row of data corresponding
            # to this 2-D vector
            #
            # normally in the random field set up we are concerned with fully connected clusters
            # but note that we have an abundance of data having labels not belonging to the inferior
            # class and we want to limit this data set
            #
            # thus we will find connections amongst data rows with labels not in the inferior
            # class and remove them, while keeping all rows with labels from the inferior class
            # and any neighbors with labels from all other classes that form closed edges with
            # its neighbors from the inferior class ... these will be "transitions" that we 
            # want to maintain in our data set
            #
            # at the end of this process we will have separated clusters of common labeled rows
            #
            # another thing to notice in kind of a thought experiment is making the analogy of
            # rows of data corresponding to the inferior class label as being visible matter
            # with all other labels corresponding to other matter ... then as we make connections
            # and eliminate fully connected rows of data as determined by its label and those of its
            # neighbors we are actually drawing other areas of "space" together (kind of creating a
            # singularity) whereby clusters of inferior labels are closer together with separation
            # being provided by the remaining other labels in between
            #
            # start considering labels in the boundary of the NxN 2-D region then work to the interior
            # we need to keep the labels in the boundary of the 2-D region contains information about
            # all of the labels external to the bounded region ... this is by the Markov property
            # thus we move on to interior labels
            nrgs = list(np.zeros(len(ret)))
            rmv  = []
            rng  = list(range(lbls[0],lbls[1]))
            for j,x in enumerate(rng):
                # these are boundary elements to the left and right of the interior
                if (j+1) % N in [0,1]:
                    continue
                else:
                    # these are the boundary elements at the top and bottom
                    # not including those elements that are also to the left
                    # and right (in the corners)
                    if j in range(0,N) or j in range(len(rng)-N,len(rng)):
                        continue
                    else:
                        # the interior labels
                        if ret[labels][x] == ret[labels][x+1] and \
                           ret[labels][x] == ret[labels][x-1] and \
                           ret[labels][x] == ret[labels][x+N] and \
                           ret[labels][x] == ret[labels][x-N]:
                            # computed energy in the clique of neighbors
                            nrg  = 0
                            notl = [i for i in range(len(ret.columns)) if not i == labels]
                            for i in [-1,1,-N,N]:
                                nrg  += np.dot(ret.to_numpy()[x,notl],ret.to_numpy()[x+i,notl])
                            nrgs.append(nrg)
                            # completely connected site so remove the label
                            rmv.append(x)
        # for any values to be removed, then remove them and keep the rest
        lbls = [i for i in rng if i not in rmv] if not len(rmv) == 0 else rng
        # get everything not in lbls for building the learner that will
        # predict the boundary and interior elements for other data sets
        nlbls= [i for i in range(len(dat)) if i not in lbls]
        # everything in the boundary and the interior will be labeled 1
        bint       = np.zeros(len(dat))
        bint[lbls] = np.ones(len(lbls))
        # build the model that will predict the boundary and interior
        # elements for other data sets and add it to the return
        #
        # in the future this will be a generative model based on the base dbn
        model= gan(dat[:,[i for i in range(len(dat[0])) if not i == labels]]
                  ,bint
                  ,sfl=None
                  ,clust=2)
    return lbls,nrgs,model

############################################################################
##
## Purpose:   Calculate an energy function for use with random field/cluster theory
##
############################################################################
def nn_energy(dat=[],labels=None,label=None,order=False,seqn=True):
    nrgs = None
    if type(dat) == type(np.asarray([])) and len(dat) > 0:
        nrgs = np.zeros(len(dat))
        ret  = pd.DataFrame(dat)
        # if the data should be sorted as requested
        if order:
            ret  = ret.sort_values(by=list(ret.columns))
        if (type(labels) == type(0) and labels >= 0 and labels <= len(ret[0])) and \
           (type(label ) == type(0) and label  >= 0                          ):
            if seqn:
                rng  = range(len(ret))
                for j in rng:
                    # range of values containing the jth row
                    if j-2 in rng and j+2 in rng:
                        b    = j-2
                        e    = j+2
                    else:
                        if j-1 in rng and j+3 in rng:
                            b    = j-1
                            e    = j+3
                        else:
                            if j+4 in rng:
                                b    = 0
                                e    = j+4
                            else:
                                if j-3 in rng and j+1 in rng:
                                    b    = j-3
                                    e    = j+1
                                else:
                                    if j-4 in rng:
                                        b    = j-4
                                        e    = len(ret)-1
                    # add the contribution of each projected neighboring vector
                    notl = [i for i in range(len(ret.columns)) if not i == labels]
                    for k in [i for i in range(b,e+1) if not i == j]:
                        nrgs[j] += np.dot(ret.to_numpy()[j,notl],ret.to_numpy()[k,notl])
            else:
                _,nrgs = nn_trim(dat,labels,label,order)
        else:
            _,nrgs = nn_trim(dat,labels,label,order)
    return nrgs

############################################################################
##
## Purpose:   Return the subset of data rows that are full
##
############################################################################
def checkdata(dat=[]):
    ret  = []
    rows = []
    cols = []
    if type(dat) in [type([]),type(np.asarray([]))] and len(dat) > 0:
        ret  = np.asarray(dat).copy()
        # check which rows have any null values and remove them
        #
        # doing rows first since a null row will eliminate all columns
        rows = [i for i in range(0,len(ret)) if type("") in utils.sif(ret[i])]
        # have to check that we still have rows of data
        if not (len(ret) == len(rows)):
            # for the return, we will remove all rows/cols that have empty (null) strings
            ret  = ret[  [i for i in range(0,len(ret   )) if i not in rows],:] if len(rows) > 0 else ret
        # check which columns have any null values and remove them
        d1   = ret.transpose()
        cols = [i for i in range(0,len(d1)) if type("") in utils.sif(d1[i])]
        if not (len(ret[0]) == len(cols)):
            ret  = ret[:,[i for i in range(0,len(ret[0])) if i not in cols]  ] if len(cols) > 0 else ret
    return ret,rows,cols

############################################################################
##
## Purpose:   Fix a data set using simulated thought in a knowledge brain
##
############################################################################
def fixdata(inst=0,dat=[],coln={}):
    ret  = dat
    if inst > const.constants.BVAL                                     and \
       type(dat ) in [type([]),type(np.asarray([]))] and len(dat ) > 0 and \
       type(coln) ==  type({})                       and len(coln) > 0:
        # check which rows/columns have any null values and remove them
        d,rows,cols = checkdata(dat)
        # have to check that we still have rows/columns of data
        if not (len(d) == 0 or len(d[0]) == 0):
            # if there is something to fix
            if len(rows) in range(1,len(dat)) or len(cols) in range(1,len(dat[0])):
                # set of columns that don't need fixing
                nrows= [i for i in range(0,len(dat   )) if i not in  rows]
                nrow = [j for j in range(0,len(dat   )) if j not in nrows]
                # set of columns that don't need fixing
                ncols= [i for i in range(0,len(dat[0])) if i not in  cols]
                # all rows with values
                ndat =  dat[nrows,    :] if not (len(nrows) == 0                  ) else []
                # inputs for importance are the subset of rows that have values
                ip   = ndat[    :,ncols] if not (len(ncols) == 0 or len(ndat) == 0) else []
                # outputs for importance calculated as categorical labels
                op   = ndat[    :, cols] if not (len( cols) == 0 or len(ndat) == 0) else []
                # do we have data to build models
                #
                # only numeric columns should need fixing at this point
                #
                # floating point column and regression prediction
                if not (len(ip) == 0 or len(op) == 0):
                    model= dbn(ip
                              ,op
                              ,sfl=None
                              ,loss="mean_squared_error"
                              ,optimizer="adam"
                              #,rbmact="relu"
                              #,dbnact='relu' if ver == const.constants.VER else 'selu'
                              ,rbmact="selu"
                              ,dbnact='sigmoid'
                              ,dbnout=len(cols))
                    if not (type(model) == type(None)):
                        ip1  = dat[nrow ,    :] if not (len(nrow ) == 0                 ) else []
                        ip1  = ip1[    :,ncols] if not (len(ncols) == 0 or len(ip1) == 0) else []
                        dat[nrow,cols] = np.asarray(model.predict(ip1.astype(np.single))).reshape((len(nrow),len(cols)))
                    else:
                        for col in cols:
                            if type("") in utils.sif(dat[:,col]):
                                #dat[nrow,col] = [ceil(np.median(dat[nrows,col].astype(np.single)))] * len(nrow)
                                dat[nrow,col] = [ceil(np.mean(dat[nrows,col].astype(np.single)))] * len(nrow)
                else:
                    if len(ndat) == 0:
                        for col in cols:
                            nrows= [i for i in range(0,len(dat[:,col])) if     utils.sif(dat[i,col]) in [type(0),type(0.0)]]
                            nrow = [j for j in range(0,len(dat       )) if not j                     in nrows              ]
                            # all rows with values
                            ndat =  dat[nrows,    :] if not  len(nrows) == 0                    else []
                            # inputs for importance are the subset of rows that have values
                            ip   = ndat[    :,ncols] if not (len(ncols) == 0 or len(ndat) == 0) else []
                            # outputs for importance calculated as categorical labels
                            op   = ndat[    :, col ] if not (                   len(ndat) == 0) else []
                            # do we have data to build models
                            #
                            # only numeric columns should need fixing at this point
                            #
                            # floating point column and regression prediction
                            if not (len(ip) == 0 or len(op) == 0):
                                model= dbn(ip
                                          ,op
                                          ,sfl=None
                                          ,loss="mean_squared_error"
                                          ,optimizer="adam"
                                          #,rbmact="relu"
                                          #,dbnact='relu' if ver == const.constants.VER else 'selu'
                                          ,rbmact="selu"
                                          ,dbnact='linear'
                                          ,dbnout=1)
                                if not (type(model) == type(None)):
                                    ip1  = dat[nrow ,    :] if not (len(nrow ) == 0                 ) else []
                                    ip1  = ip1[    :,ncols] if not (                   len(ip1) == 0) else []
                                    if not (len(ip1) == 0):
                                        dat[nrow,col] = np.asarray(model.predict(ip1.astype(np.single))).flatten()
                                    else:
                                        if type("") in utils.sif(dat[:,col]):
                                            #dat[nrow,col] = [ceil(np.median(dat[nrows,col].astype(np.single)))] * len(nrow)
                                            dat[nrow,col] = [ceil(np.mean(dat[nrows,col].astype(np.single)))] * len(nrow)
                                else:
                                    if type("") in utils.sif(dat[:,col]):
                                        #dat[nrow,col] = [ceil(np.median(dat[nrows,col].astype(np.single)))] * len(nrow)
                                        dat[nrow,col] = [ceil(np.mean(dat[nrows,col].astype(np.single)))] * len(nrow)
                            else:
                                if type("") in utils.sif(dat[:,col]):
                                    #dat[nrow,col] = [ceil(np.median(dat[nrows,col].astype(np.single)))] * len(nrow)
                                    dat[nrow,col] = [ceil(np.mean(dat[nrows,col].astype(np.single)))] * len(nrow)
                    else:
                        for col in cols:
                            if type("") in utils.sif(dat[:,col]):
                                #dat[nrow,col] = [ceil(np.median(dat[nrows,col].astype(np.single)))] * len(nrow)
                                dat[nrow,col] = [ceil(np.mean(dat[nrows,col].astype(np.single)))] * len(nrow)
    return ret

############################################################################
##
## Purpose:   Data cleansing activities
##
############################################################################
def nn_cleanse(inst=0,d=None):
    nhdr = None
    dat  = None
    sifs = None
    keys = None
    if type(d) == type(pd.DataFrame()):
        dat  = d.copy()
        # going to capture the header and data so it can be replaced
        hdr  = list(dat.columns)
        nhdr = list(np.asarray(hdr).copy())
        # drop any columns that have been configured to be dropped in constants.py
        if hasattr(const.constants,"DROP" )                                                                  and \
           type(const.constants.DROP ) in [type([]),type(np.asarray([]))] and len(const.constants.DROP ) > 0:
            cols = [c for c in const.constants.DROP  if c in nhdr]
            if not len(cols) == 0:
                #drop = dat.iloc[:,[hdr.index(i) for i in const.constants.DROP ]].to_numpy().copy()
                dat  = dat.drop(columns=cols)
                dhdr = list(np.asarray(hdr)[[hdr.index(i) for i in cols]])
                nhdr = [i for i in  hdr if i not in dhdr]
        # drop any dates that have been configured to be dropped in constants.py
        #if hasattr(const.constants,"DATES")                                                                  and \
           #type(const.constants.DATES) in [type([]),type(np.asarray([]))] and len(const.constants.DATES) > 0:
            #cols = [c for c in const.constants.DATES if c in nhdr]
            #if not len(cols) == 0:
                #dts  = dat.iloc[:,[hdr.index(i) for i in const.constants.DATES]].to_numpy().copy()
                #dat  = dat.drop(columns=cols)
                #thdr = list(np.asarray(hdr)[[hdr.index(i) for i in cols]])
                #nhdr = [i for i in nhdr if i not in thdr]
        # replace any NaNs
        dat  = np.asarray([[str(x).lower().replace("nan","") for x in row] for row in dat.to_numpy()])
        dat  = pd.DataFrame(dat,columns=nhdr)
        # remove any completely null columns
        cols = list(dat.columns)
        chdr = []
        for i in range(0,len(cols)):
            d    = dat.iloc[:,i].to_list()
            if d == [""] * len(d):
                dat  = dat.drop(columns=cols[i])
                chdr.append(i)
        nhdr = [i for i in nhdr if i not in chdr]
        # now continue on
        dat  = dat.to_numpy()
        # the remaining columns turned into a dictionary
        coln = {h:i for i,h in enumerate(nhdr)}
        # check which rows/columns have any null values and remove them
        d,rows,cols = checkdata(dat)
        # have to check that we still have rows/columns of data
        if not (len(d) == 0 or len(d[0]) == 0):
            keys = []
            for i in range(0,len(dat[0])):
                key  = None
                # string columns will be labeled using wikilabel
                # rows of sifs are the actual columns so transpose later
                vec  = utils.sif(dat[:,i])
                if not type(sifs) == type(None):
                    sifs = np.vstack((sifs,vec))
                    b    = type(0) in sifs[-1] or type(0.0) in sifs[-1]
                else:
                    sifs = vec
                    b    = type(0) == sifs     or type(0.0) == sifs
                if not b:
                    #wiki = wikilabel(inst,dat[indxr,i],True,True)
                    key  = {}
                    ccd  = np.asarray(list(calcC(dat[:,i],len(utils.utils._unique(dat[:,i])),key)))
                    # if there is something to fix
                    if i in cols:
                        dat[rows,i] = ccd.flatten()[rows]
                keys.append(key)
            sifs = sifs.transpose()
        # fix the data by intelligently filling missing values
        dat  = fixdata(inst,dat,coln)
        dat  = pd.DataFrame(dat,columns=nhdr)
        dat  = dat.to_numpy()
        dat[dat == ""] = "0"
    return {"nhdr":nhdr if nhdr is not None else np.asarray([])
           ,"dat" : dat if dat  is not None else np.asarray([])
           ,"sifs":sifs if sifs is not None else np.asarray([])
           ,"keys":keys if keys is not None else np.asarray([])}

############################################################################
##
## Purpose:   Build the Gibbs sampler
##
############################################################################
def nn_gibbs(dat=None):
    gibbs= {"orig":None,"mod":None,"smth":None}
    if (type(dat) == type([]) or type(dat) == type(np.asarray([]))):
        gibbs["orig"] = dat.copy()
        # we have the original image, the modified image, and the smoothed image
        #
        # make the input values better for processing in parallel
        ret           = nn_dat(dat)
        gibbs["mod" ] = ret["ivals" ]
        gibbs["smth"] = ret["rivals"]
    return gibbs

############################################################################
##
## Purpose:   Main class
##
############################################################################
class NN():
    @classmethod
    def _split(self,pfl=None,mfl=None):
        return nn_split(pfl,mfl)
    @classmethod
    def _gibbs(self,dat=None):
        return nn_gibbs(dat)

############################################################################
##
## Purpose:   Testing
##
############################################################################
def nn_testing(M=500,N=3,pfl="/home/robert/data/csv/patients.csv",mfl="/home/robert/data/csv/medications.csv"):
    ret  = {"orig":None,"mod":None,"smth":None}
    if (os.path.exists(pfl) and os.stat(pfl).st_size > 0) and \
       (os.path.exists(mfl) and os.stat(mfl).st_size > 0):
        nn   = NN()
        # split the dataset into train and test sets
        split= nn._split(pfl,mfl)
        # during the model build we will output validation accuracy
        #
        # as one more output, we will plot the accuracy vs
        # the number of epochs for tuning the learner
        ret  = nn._gibbs(split["train"])
        # test the base NN
        #
        # uniformly sample values between 0 and 1
        m    = M
        p    = min(const.constants.MAX_FEATURES,N)
        s    = p
        ivals= np.random.sample(size=(m,p))
        ovals= categoricals(m,s,p)
        # generate the clustering model for using the test values for training
        model = dbn(ivals,ovals,splits=s,props=p)
        if not (type(model) == type(None)):
            # generate some test data for predicting using the model
            ovals= np.random.sample(size=(np.uint16(m/10),p))
            # encode and decode values
            pvals= model.predict(ovals)
            # look at the original values and the predicted values
            print(ovals)
            print(pvals)
        else:
            print("Model 1 is null.")
        ovals= np.random.sample(size=(m,1))
        # generate the regression model for using the test values for training
        #model = dbn(ivals
                    #,ovals
                    #,splits=s
                    #,props=p
                    #,loss='mean_squared_error'
                    #,optimizer='sgd'
                    #,rbmact='sigmoid'
                    #,dbnact='linear'
                    #,dbnout=1)
        model = dbn(ivals
                   ,ovals
                   ,splits=s
                   ,props=p
                   ,loss='mean_squared_error'
                   ,optimizer='sgd'
                   #,rbmact='relu'
                   ,rbmact='selu'
                   ,dbnact='linear'
                   ,dbnout=1)
    return [ret,model]
