#!/usr/bin/env Rscript

############################################################################
##
## File:      test.R
##
## Purpose:   Testing utilities
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Nov. 19, 2019
##
############################################################################

library("randomForest",warn.conflicts=FALSE,verbose=FALSE,quietly=TRUE)
library("mosaic"      ,warn.conflicts=FALSE,verbose=FALSE,quietly=TRUE)

source("db.R")

# get the data
#dat  <- atest.db.functions.data.mysql("coupons50000.csv")
#dat1 <- read.csv("couponsnonbillable1.csv")
#dat  <- rbind(dat,dat1[sample(1:nrow(dat1),50),])
dat  <- atest.db.functions.data.mysql("coupons100000.csv")

# engineer a binary feature that indicates if a consumer has non billable activity
nbcol<- "COUPONBILLINGCODEID"
nb   <- rep(0,nrow(dat))
wnb  <- which(as.character(dat[,nbcol])%in%c("2","12","4","6","7","11","13","14"))
if( length(wnb) > 0 ) nb[wnb] <- rep(1,length(wnb))
dat  <- cbind(dat,nb)
colnames(dat)[ncol(dat)] <- "NONBILLABLE"
fl   <- "coupons.csv"
write(t(colnames(dat)),fl,ncolumns=ncol(dat),sep=",",append=FALSE)
write(t(         dat ),fl,ncolumns=ncol(dat),sep=",",append=TRUE )

# if a consumer has non billable activity, then the
# engineered feature will be 1 for all rows for that consumer
mbcol<- "MEMBERSHIPNUMBER"
racol<- "REDEMPTIONAMOUNT"
wdat <- which(as.numeric(dat[,ncol(dat)])==1)
if( length(wdat) > 0 ) {
    mbrs <- as.character(dat[,mbcol])
    umbrs<- unique(mbrs[wdat])
    wmbrs<- which(mbrs%in%umbrs)
    dat[wmbrs,ncol(dat)] <- rep(1,length(wmbrs))
    for( mbr in umbrs ) {
        wmbr <- which(mbrs==mbr)
        wdmbr<- which(mbrs[wdat]==mbr)
        cdat <- matrix(c(median(as.numeric(dat[wmbr,racol])),length(wdmbr)/length(wmbr)),nrow=1,ncol=2)
        colnames(cdat) <- c(racol,"PCTNONBILLABLE")
        rdat <- atest.db.functions.pct.mysql(cdat)
    }
    fl   <- "pctcoupons.csv"
    write(t(colnames(rdat)),fl,ncolumns=ncol(rdat),sep=",",append=FALSE)
    write(t(         rdat ),fl,ncolumns=ncol(rdat),sep=",",append=TRUE )
    x    <- as.numeric(rdat[,1])
    y    <- as.numeric(rdat[,2])
    df   <- data.frame(x=x,y=y)
    mod  <- lm(y~.,data=df)
    df   <- data.frame(x=x,y=0)
    pred <- predict(mod,newdata=df)
    png("linear_model.png",width=1000,height=600)
    graphics::plot(y~x
                  ,xlim=c(min(x),max(x))
                  ,ylim=c(min(y),max(y))
                  ,col='blue'
                  ,xlab="Redemption Amount"
                  ,ylab="Percent Non-Billable"
                  ,main="Regression Model: Percent Non-Billable Redemptions vs. Redemption Amount")
    graphics::lines(pred~x,typ='l',col='red')
    dev.off()
    #qplot(x,y,data=df)
    png("model_residuals.png",width=1000,height=600)
    graphics::plot(mod,id.n=2,col='blue')
    dev.off()
}

# convert some binary columns to use as other inputs to the model during training
tcol <- c("ISRAINCHECK"
         ,"ISREFUND"
         ,"ISVOID"
         ,"TAXCODE"
         ,"ISRESALE")
for( col in tcol ) {
    vcol <- rep(0,nrow(dat))
    vals <- as.character(dat[,col])
    wval <- which(vals=="Y")
    if( length(wval) > 0 ) vcol[wval] <- rep(1,length(wval))
    dat[,col] <- vcol
}

# get the set of columns to use as the independent variables for the model
mcol <- c(which(colnames(dat)%in%tcol),grep("REDEMPTIONAMOUNT",colnames(dat)))

# build the model
x    <- as.matrix(dat[,mcol])
y    <- as.factor(dat[,ncol(dat)])
df   <- data.frame(x=x,y=y)
rf   <- randomForest(y~.,data=df,importance=TRUE)

# print the confusion matrix and variable importance
print(rf); print(importance(rf))
