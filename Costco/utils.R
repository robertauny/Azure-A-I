#!/usr/bin/env Rscript

############################################################################
##
## File:      utils.R
##
## Purpose:   General utilities
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Nov. 19, 2019
##
############################################################################

# defined constants
atest.utils.constants.name   <- "name"
atest.utils.constants.file   <- "file"

atest.utils.constants.sp.eq  <- "="
atest.utils.constants.sp.ul  <- "_"
atest.utils.constants.sp.div <- "|"
atest.utils.constants.sp.div2<- "+"
atest.utils.constants.sp.c   <- ","

atest.utils.constants.fl     <- "inst.txt"
atest.utils.constants.na     <- "N/A"

# file name utility
atest.utils.functions.file <- function(x=NULL,f=NULL) {
    ret  <- NULL
    if( !is.null(f) ) {
        sp   <- "/"
        fl   <- strsplit(f,sp,fixed=TRUE)
        if( !is.null(fl) ) {
            f1   <- unlist(fl)
            f2   <- f1[length(f1)]
            p1   <- paste(f1[1:(length(f1)-1)],collapse=sp)
            p2   <- paste(x,f2,sep="")
            p3   <- paste(p1,p2,sep="/")
            ret  <- list(nm=f2,di=p1,dr=p3)
        }
        else
            ret  <- list(nm=f,di=NULL,dr=paste(x,f,sep=""))
    }
    return(ret)
}

# unique rows
atest.utils.functions.unique <- function(dat=NULL) {
    ret  <- c()
    if( !is.null(dat) ) {
        if( is.matrix(dat) ) for( i in 1:nrow(dat) ) ret  <- c(ret,paste(dat[i,],collapse=""))
        else                                         ret  <-       paste(dat    ,collapse="")
    }
    return(ret)
}

# just return what's passed (for concurrency)
atest.utils.functions.return <- function(dat=NULL) {
    return(dat)
}

# parse instance file
atest.utils.functions.inst <- function(typ=NULL) {
    ret  <- NULL
    if( file.exists(atest.utils.constants.fl) && file.size(atest.utils.constants.fl) > 0 ) {
        if( !is.null(typ) ) {
            f    <- file(atest.utils.constants.fl)
            dat  <- readLines(f)
            close(f)
            if( !is.null(dat) ) {
                gnm  <- grep(paste(typ,atest.utils.constants.sp.eq,sep=""),dat)
                if( length(gnm) > 0 ) {
                    strc <- as.character(unlist(strsplit(dat[gnm],atest.utils.constants.sp.eq,fixed=TRUE)))
                    ret  <- strc[2]
                }
            }
        }
    }
    return(ret)
}

# categorical counts
atest.utils.functions.counts <- function(val=NULL,pct=NULL,iter=NULL) {
    if( !(is.null(val) || is.null(pct) || is.null(iter)) ) {
        cnt  <- (1.0-pct)*val
        for( i in 1:iter ) {
            cnt  <- (1.0+0.02) * (1.0-0.02) * cnt
            print(paste("CNT:",cnt))
        }
    }
}

# mimic sprintf in C
atest.utils.functions.sprintf <- function(fmt, ...) {
     MAX_NVAL <- 99L
     args <- list(...)
     if( length(args) <= MAX_NVAL ) return( sprintf(fmt,...) )
     stopifnot(length(fmt) == 1L)
     not_a_spec_at <- gregexpr("%%",fmt,fixed=TRUE)[[1L]]
     not_a_spec_at <- c(not_a_spec_at,not_a_spec_at + 1L)
     spec_at       <- setdiff(gregexpr("%",fmt,fixed=TRUE)[[1L]],not_a_spec_at)
     nspec         <- length(spec_at)
     if( length(args) < nspec ) stop( "too few arguments" )
     if( nspec <= MAX_NVAL ) break_points <- integer(0)
     else                    break_points <- seq( MAX_NVAL + 1L, nspec, by=MAX_NVAL)
     break_from   <- c(1L,break_points)
     break_to     <- c(break_points-1L, nspec)
     fmt_break_at <- spec_at[break_points]
     fmt_chunks   <- substr(rep.int(fmt,length(fmt_break_at)+1L)
                           ,c(1L,fmt_break_at)
                           ,c(fmt_break_at-1L,nchar(fmt)))
     ans_chunks <- mapply(
         function(fmt_chunk,from,to)
             do.call(sprintf,c(list(fmt_chunk),args[from:to]))
         ,fmt_chunks
         ,break_from
         ,break_to
     )
     paste(ans_chunks, collapse="")
}

# dynamic creation of encryption file name
atest.utils.functions.gname<- function(nm=NULL) {
    ret  <- NULL
    if( !is.null(nm) ) {
        syst <- as.character(format(Sys.time(),"%m%d%Y%H%M%s"))
        if( length(grep(".gpg",nm)) > 0 ) ret  <- gsub("gpg",syst,nm)
        if( length(grep(".pgp",nm)) > 0 ) ret  <- gsub("pgp",syst,nm)
    }
    return(ret)
}

# collapse strings for uniqueness comparison
atest.utils.functions.paste.collapse <- function(mat=NULL,opt=NULL) {
    ret  <- c()
    if( !(is.null(mat) || is.null(opt)) ) {
        nr   <- nrow(mat)
        if( !is.null(nr) ) for( i in 1:nr ) ret[i] <- paste(mat[i,],collapse=opt)
    }
    return(ret)
}
