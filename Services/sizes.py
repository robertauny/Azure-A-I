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

#!/usr/bin/python3
# /tmp/sizes.py

import os
import requests
import math

import numpy as np

from PIL import Image

import constants as const

import config

cfg  = config.cfg()

def sizes(wtyp=const.OCR,inst=0,dr=None):
    ret  = {"rows":0,"cols":0,"bbh":math.inf,"bbl":math.inf,"bb":[math.inf,math.inf,math.inf,math.inf]}
    if not (wtyp == None or inst <= const.BVAL or dr == None):
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
        hdrs = {"Ocp-Apim-Subscription-Key":key,"Content-Type":"application/octet-stream"}
        parms= {"language":"unk","detectOrientation":"true"}
        for r,d,fls in os.walk(dr):
            for fl in fls:
                img         = Image.open(dr+"/"+fl)
                shp         = np.shape(img)
                ret["rows"] = shp[0] if shp[0] > ret["rows"] else ret["rows"]
                ret["cols"] = shp[1] if shp[1] > ret["cols"] else ret["cols"]
                img.close()
                try:
                    f    = open(dr+"/"+fl,"rb")
                    # get response from the server
                    resp = requests.post(url,headers=hdrs,params=parms,data=f)
                    resp.raise_for_status()
                    # get json data to parse it later
                    js   = resp.json()
                    # all the lines from a page, including noise
                    for reg in js["regions"]:
                        line = reg["lines"]
                        for elem in line:
                            bbs  = [word["boundingBox"] for word in elem["words"]]
                            for bb in bbs:
                                bbp        = bb.split(",")
                                bbh        = abs(int(bbp[0])-int(bbp[1]))
                                bbl        = abs(int(bbp[2])-int(bbp[3]))
                                ret["bb" ] = bbp if (0 < bbh and bbh < ret["bbh"]) and (0 < bbl and bbl < ret["bbl"]) and bbh * bbl < ret["bbh"] * ret["bbl"] else ret["bb" ]
                                ret["bbh"] = bbh if  0 < bbh and bbh < ret["bbh"]                                                                             else ret["bbh"]
                                ret["bbl"] = bbl                                     if 0 < bbl and bbl < ret["bbl"]                                          else ret["bbl"]
                    f.close()
                except Exception as err:
                    print(str(err))
    return ret
