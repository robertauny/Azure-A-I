#!/usr/bin/env Rscript

############################################################################
##
## File:      relu.R
##
## Purpose:   Demonstrate how relu works via picture-based example.
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Jul. 23, 2019
##
############################################################################

rm(list=ls())

library("stats"   ,warn.conflicts=FALSE,verbose=FALSE,quietly=TRUE)
library("graphics",warn.conflicts=FALSE,verbose=FALSE,quietly=TRUE)

pf   <- paste(Sys.getenv("HOME"),"/code/scripts/python/djo/images/relu.png",sep="")

png(pf,width=1000,height=600)
x    <- 1:100
y    <- rnorm(100)
plot(x,y,col='blue')
lines(c(1,100),c(-0.2,0.2),type='l',col='red'  )
lines(c(1,100),rep(0,2)   ,type='l',col='green')
dev.off()
