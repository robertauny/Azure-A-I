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
## File:      includes.py
##
## Purpose:   Include a few required packages and such
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Nov. 22, 2021
##
############################################################################

import os
import sys
import subprocess

import config
import constants       as const

import numpy           as np

# read the default json config file
cfg  = config.cfg()

############################################################################
# SNOWFLAKE BEGIN
############################################################################

# snowflake installers
def get_all_rows(sql=""):
    return None
if hasattr(const.constants,"INCLUDES")                                  and \
   type(const.constants.INCLUDES)    in [type([]),type(np.asarray([]))] and \
   type(const.constants.INCLUDES[0]) in [type([]),type(np.asarray([]))]:
    for inc in const.constants.INCLUDES:
        if len(inc) == 3:
            if inc[0] == "pip":
                subprocess.check_call([sys.executable,"-m",inc[0],inc[1],inc[2]])
            else:
                subprocess.run(inc)
    if "snowflake.sqlalchemy==1.2.4" in const.constants.INCLUDES:
        from sqlalchemy                       import create_engine
        from sqlalchemy                       import *
        from snowflake.sqlalchemy             import URL
        from sqlalchemy.dialects              import registry
        registry.register('snowflake', 'snowflake.sqlalchemy', 'dialect')
    if "pydb" in const.constants.INCLUDES:
        import pyhdb

############################################################################
# SNOWFLAKE END
############################################################################
