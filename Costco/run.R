#!/usr/bin/env Rscript

############################################################################
##
## File:      run.R
##
## Purpose:   General run utility
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Nov. 19, 2019
##
############################################################################

write(format(Sys.time(),"%Y-%m-%d %H:%M:%s"),"/tmp/time.out")
setwd('/home/robert/code/scripts/r/costco')
source('test.R')
write(format(Sys.time(),"%Y-%m-%d %H:%M:%s"),"/tmp/time.out",append=TRUE)
