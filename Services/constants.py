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

BVAL         = -1
MAX_CLUSTERS = 50
MAX_FEATURES = 2
MAX_SPLITS   = 3
V            = "vertices"
E            = "edges"
MDL          = "model"
LBL          = "label"
SEP          = "-"
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
WIK          = "wiki"
# entity extraction
EE           = "keyPhrases"
# sentiment
SENT         = "sentiment"
