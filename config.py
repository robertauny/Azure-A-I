#!/usr/bin/python

############################################################################
##
## File:      config.py
##
## Purpose:   Read and parse the JSON config file.
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Dec. 23, 2019
##
############################################################################

import json
import os

############################################################################
##
## Purpose:   Read a configuration file
##
############################################################################
def cfg(fl='irg.json'):
    ret= None
    if( os.path.exists(fl) and os.stat(fl).st_size > 0 ):
        with open(fl) as f:
        	ret  = json.load(f)
        try:
            with open(fl) as f:
                ret  = json.load(f)
        except:
            ret  = ret.msg
    # return the parsed config to the caller
    return ret

############################################################################
##
## Purpose:   Write a configuration file
##
############################################################################
def wcfg(dat=None,fl='/tmp/irg.json'):
    ret= None
    if( len(dat) > 0 ):
        try:
            with open(fl,'w') as f:
                json.dump(dat,f)
        except:
            ret  = 'Unable to write to ' + dat
    # return the parsed config to the caller as a check
    return ret

# *************** TESTING *****************

def config_testing():
    ret  = cfg('test.json')
    print(ret)
    f    = open('test.json')
    ret  = wcfg(f.read())
    print(ret)
