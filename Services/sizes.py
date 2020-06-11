#!/usr/bin/python3
# /tmp/test3.py

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
                                ret["bb" ] = [ret["bb"][0],ret["bb"][1],bbp[2]      ,bbp[3]      ] if 0 < bbl and bbl < ret["bbl"] else ret["bb" ]
                                ret["bb" ] = [bbp[0]      ,bbp[1]      ,ret["bb"][2],ret["bb"][3]] if 0 < bbh and bbh < ret["bbh"] else ret["bb" ]
                                ret["bbh"] =  bbh                                                  if 0 < bbh and bbh < ret["bbh"] else ret["bbh"]
                                ret["bbl"] =  bbl                                                  if 0 < bbl and bbl < ret["bbl"] else ret["bbl"]
                    f.close()
                except Exception as err:
                    print(str(err))
    return ret
