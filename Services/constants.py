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
## File:      constants.py
##
## Purpose:   General configuration constants
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Nov. 22, 2021
##
############################################################################

import sys

ver  = sys.version.split()[0]

import tensorflow as tf

class constants():
    def custom_loss(y_true,y_pred):
        CSTFN= 0.85
        CSTTP= 0.0
        CSTFP= 0.15
        CSTTN= 0.0
        cost =    y_true  * np.log(  y_pred) * CSTFN \
             +    y_true  * np.log(  y_pred) * CSTTP \
             + (1-y_true) * np.log(1-y_pred) * CSTFP \
             + (1-y_true) * np.log(1-y_pred) * CSTTN
        return -np.mean(cost,axis=-1)
    # miscellaneous constants
    CONFIG         = "services.json"
    TEST           = "test.json"
    BVAL           = -1
    MAX_CLUSTERS   = 100#10
    MAX_FEATURES   = 100 #>= 3
    MAX_SPLITS     = 10#MAX_FEATURES
    MAX_COLS       = 100#MAX_FEATURES
    MAX_ROWS       = 100
    MAX_DIM        = 81
    V              = "vertices"
    E              = "edges"
    MDL            = "model"
    LBL            = "label"
    SEP            = "_"
    SEED           = 12345
    INST           = 0
    FILL_COLOR     = 0
    ERR            = "ERROR"
    VER            = "3.5.2"
    DEV_CPU        = "/cpu:0"
    DEV_GPU        = "/gpu:0"
    DEV            = DEV_CPU
    DIFFA          = 2
    DIFFB          = 1
    IMS_PER_BATCH  = 2#0
    MASK_INTENSITY = 20
    TMP_DIR        = "/tmp/"
    EVAL_TYPE      = None
    PMO            = False
    RO             = True
    SHIFT          = 200#150
    PRED_L_SHIFT   = 10
    PRED_H_SHIFT   = 15
    VSIZE          = 8
    HOUN_OFFSET    = 10
    CPU_COUNT      = 1
    #PERMS          = [[0,1,2,3]]#2#11# either a constant used in range or the column numbers of the inputs (outputs found automatically)
    PERMS          = [[0,1,2]]#2#11# either a constant used in range or the column numbers of the inputs (outputs found automatically)
    COLUMNS        = [
                      "Fraud"
                     ,"ORIGIN_IP"
                     ,"DEST_IP"
                     ,"CLICKS"
                     #,"dt"
                     ]
    DROP           = [
                      "dt"
                     #,"ORIGIN_IP"
                     #,"DEST_IP"
                     #,"CLICKS"
                     #,"Fraud"
                     ]
    XLABEL         = "Event Number"#"Storm Event Number"
    #TARGETS        = ["REASONCODE"]
    #DATES          = ["dt"]
    #DTFORMAT       = "%Y-%m-%dT%H:%M:%SZ"
    #DTFORMAT       = "%s"
    #MATCH_ON       = ["Id","PATIENT"]
    # neural network constants
    OUTP           = 1
    SHAPE          = 3
    SPLITS         = 3
    PROPS          = 3
    #LOSS           = custom_loss
    #LOSS           = "categorical_crossentropy"
    LOSS           = "binary_crossentropy"
    #OPTI           = "rmsprop"
    OPTI           = "adam"
    RBMA           = "softmax"
    #RBMA           = "relu"
    METRICS        = ["accuracy"]
    DBNA           = None
    DBNO           = 0
    EPO            = 10
    EMB            = False
    ENCS           = None
    USEA           = False
    VERB           = 0
    BSZ            = 64
    LDIM           = 256
    VSPLIT         = 0.2
    SFL            = "models/mrc.h5"
    TRAIN_PCT      = 0.8
    MAX_PREDS      = 100#0
    BASE_LR        = 0.0002
    MAX_ITER       = 30000
    KFOLD          = 5
    THRESH         = 0.05
    POIS           = 0.01
    # global stem variables for some NLP
    #
    # tokenization
    TOKS         = "toks"
    # entity extraction
    ENTS         = "ents"
    # concepts
    CONS         = "cons"
    # part of speech
    POST         = "post"
    # stemming
    STEM         = "stem"
    # all stems but only using vertices for now
    STEMS        = [V]
    # extension used when storing the knowledge graph files
    EXT          = ".xml"
    # optical character recognition
    OCR          = "ocr"
    # image processing
    IMG          = "ocri"
    # object detection
    OBJ          = "objd"
    # shape detection
    SHP          = "objs"
    # shape detection
    WIK          = "summary"
    # entity extraction
    EE           = "keyPhrases"
    # sentiment
    SENT         = "sentiment"
    # keyPhrases
    KP           = "keyPhrases"
