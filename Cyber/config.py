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
def cfg(fl='cyber.json'):
    ret= None
    if( os.path.exists(fl) and os.stat(fl).st_size > 0 ):
        try:
            with open(fl) as f:
                ret  = json.load(f)
                f.close()
        except:
            ret  = ret.msg
    # return the parsed config to the caller
    return ret

############################################################################
##
## Purpose:   Write a configuration file
##
############################################################################
def wcfg(dat=None,fl='/tmp/cyber.json'):
    ret= None
    if( len(dat) > 0 ):
        try:
            with open(fl,'w') as f:
                json.dump(dat,f)
                f.close()
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
    f.close()
    print(ret)
