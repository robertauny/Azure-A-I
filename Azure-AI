#!/usr/bin/env Rscript

############################################################################
##
## File:      config.R
##
## Purpose:   Parse the XML document for config values.
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Nov. 19, 2019
##
############################################################################

library("XML"      ,warn.conflicts=FALSE,verbose=FALSE,quietly=TRUE)
library("parallel" ,warn.conflicts=FALSE,verbose=FALSE,quietly=TRUE)
library("doMC"     ,warn.conflicts=FALSE,verbose=FALSE,quietly=TRUE)

source("utils.R")

# configuration file
atest.config.constants.file           <- atest.utils.functions.inst("file")
atest.config.constants.file           <- ifelse(is.null(atest.config.constants.file)
                                               ,"/opt/atest/atest.xml"
                                               ,atest.config.constants.file)
atest.config.constants.cores          <- atest.utils.functions.inst("cores")
atest.config.constants.cores          <- ifelse(is.null(atest.config.constants.cores)
                                               ,8
                                               ,atest.config.constants.cores)

atest.config.constants.node           <- "node"
atest.config.constants.item           <- "item"
atest.config.constants.value          <- "value"
atest.config.constants.map            <- "map"

atest.config.constants.name           <- "atest"
atest.config.constants.type           <- "type"

# global file creation permissions 666
Sys.umask("000")

# detect cores for multi-core parallel processing
atest.config.constants.dc             <- detectCores()
#if( is.na(atest.config.constants.dc) )
    atest.config.constants.dc         <- 1
registerDoMC(min(atest.config.constants.cores,atest.config.constants.dc))

# parse the configuration file or
# return the name of the config file
atest.config.functions.parse <- function(f=atest.config.constants.file) {
    # config file will take the form
    #
    # <config>
    #      <node>
    #          <item>Value</item>
    #          <item>Value</item>
    #          <item>Value</item>
    #      </node>
    #      <node>
    #          <item>Value</item>
    #          <item>Value</item>
    #          <item>Value</item>
    #      </node>
    #      <node>
    #          <item>Value</item>
    #          <item>Value</item>
    #          <item>Value</item>
    #      </node>
    # </config>
    #
    # where node can be something like "db" for assigning the database items, "url", "user", "pass"
    # or node can be "ftp" for assigning the ftp items, "url", "user", "pass", etc.
    ret <- NULL
    if( is.null(f) || !file.exists(f) || file.size(f) == 0 ) f    <- atest.config.constants.file
    mat1 <- c()
    mat2 <- c()
    mat3 <- c()
    tdat <- xmlTreeParse(f)
    #dat  <- tdat[[1]]$children for multiple config blocks
    dat  <- tdat[[1]]$children$config
    for( i in 1:length(dat) ) {
        node <- dat[[i]]
        for( j in 1:length(node) ) {
            item <- node[j]
            xmln <- xmlName(node)
            xmli <- xmlName(item[[1]])
            xmlv <- xmlValue(item[[1]])
            if( !(length(xmln) == 0 || length(xmli) == 0 || length(xmlv) == 0) ) {
                mat1 <- c(mat1,xmln)
                mat2 <- c(mat2,xmli)
                mat3 <- c(mat3,xmlv)
            }
        }
    }
    ret           <- matrix(cbind(mat1,mat2,mat3),ncol=3)
    colnames(ret) <- c(atest.config.constants.node
                      ,atest.config.constants.item
                      ,atest.config.constants.value)
    return(ret)
}

# search the configuration info for
# the config parms and the value
atest.config.functions.search <- function(mat=NULL,node=NULL,item=NULL) {
    ret <- NULL
    if( !(is.null(mat) || is.null(node) || is.null(item)) ) {
        coln <- which(colnames(mat)==atest.config.constants.node)
        coli <- which(colnames(mat)==atest.config.constants.item)
        colv <- which(colnames(mat)==atest.config.constants.value)
        for( n in 1:nrow(mat) )
            if( mat[n,coln] == node && mat[n,coli] == item )
                ret  <- as.character(mat[n,colv])
    }
    return(ret)
}

# configuration information
atest.config.variables <- atest.config.functions.parse()

atest.config.functions.get <- function(var=NULL) {
    ret  <- NULL
    if( !is.null(var) ) {
        getc <- atest.config.functions.search(atest.config.variables
                                             ,atest.config.constants.name
                                             ,atest.config.constants.map)
        if( !is.null(getc) && file.exists(getc) && file.size(getc) > 0 ) {
            getf <- file(getc,"r")
            if( isOpen(getf) ) {
                getr <- as.matrix(readLines(getf))
                getu <- unlist(strsplit(getr,"=",fixed=TRUE))
                wget <- which((1:length(getu))%%2==1)
                getr <- matrix(c(getu[wget],getu[-wget]),nrow=nrow(getr),ncol=2)
                getg <- which(getr[,1]==var)
                if( length(getg) > 0 ) ret  <- getr[getg,2]
                else                   ret  <- var
                close(getf)
            }
            else
                ret  <- var
        }
        else
            ret  <- var
    }
    return(ret)
}
