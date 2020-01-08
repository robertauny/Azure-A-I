#!/usr/bin/env Rscript

############################################################################
##
## File:      db.R
##
## Purpose:   Database utilities
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Nov. 19, 2019
##
############################################################################

#library("RSQLServer"  ,warn.conflicts=FALSE,verbose=FALSE,quietly=TRUE)
#library("RODBC"       ,warn.conflicts=FALSE,verbose=FALSE,quietly=TRUE)
library("RRedshiftSQL",warn.conflicts=FALSE,verbose=FALSE,quietly=TRUE)
library("DBI"         ,warn.conflicts=FALSE,verbose=FALSE,quietly=TRUE)
library("RMySQL"      ,warn.conflicts=FALSE,verbose=FALSE,quietly=TRUE)
library("RPostgreSQL" ,warn.conflicts=FALSE,verbose=FALSE,quietly=TRUE)
library("RJDBC"       ,warn.conflicts=FALSE,verbose=FALSE,quietly=TRUE)
#library("ROracle"     ,warn.conflicts=FALSE,verbose=FALSE,quietly=TRUE)
library("dplyr"       ,warn.conflicts=FALSE,verbose=FALSE,quietly=TRUE)

source("config.R")

# defined constants
atest.db.constants.url                <- "url"
atest.db.constants.user               <- "user"
atest.db.constants.pass               <- "pass"
atest.db.constants.host               <- "host"
atest.db.constants.db                 <- "db"
atest.db.constants.sid                <- "sid"
atest.db.constants.port               <- "port"
atest.db.constants.class              <- "class"
atest.db.constants.jar                <- "jar"
atest.db.constants.atest              <- atest.config.constants.name

atest.db.constants.redshift.name      <- "redshift"
atest.db.constants.oracle.name        <- "oracle"
atest.db.constants.sqlserver.name     <- "sqlserver"
atest.db.constants.mysql.name         <- "mysql"

# redshift disconnect
atest.db.functions.disconnect.redshift <- function(conn=NULL) {
    if( !is.null(conn) ) {
        dbl  <- suppressWarnings(dbListResults(conn)[1][[1]])
        if( !is.null(dbl) ) dbClearResult(dbl)
        dbDisconnect(conn)
    }
}

# oracle disconnect
atest.db.functions.disconnect.oracle   <- atest.db.functions.disconnect.redshift

# sql server disconnect
atest.db.functions.disconnect.sqlserver<- atest.db.functions.disconnect.redshift

# mysql disconnect
atest.db.functions.disconnect.mysql    <- atest.db.functions.disconnect.redshift

# redshift connection
atest.db.functions.connect.redshift <- function(user=NULL,pass=NULL,host=NULL,db=NULL,port=NULL) {
    # user
    user <- atest.config.functions.search(atest.config.variables
                                         ,atest.db.constants.redshift.name
                                         ,atest.db.constants.user)
    # password
    pass <- atest.config.functions.search(atest.config.variables
                                         ,atest.db.constants.redshift.name
                                         ,atest.db.constants.pass)
    # host
    host <- atest.config.functions.search(atest.config.variables
                                         ,atest.db.constants.redshift.name
                                         ,atest.db.constants.host)
    # database
    db   <- atest.config.functions.search(atest.config.variables
                                         ,atest.db.constants.redshift.name
                                         ,atest.db.constants.db)
    # port number
    port <- atest.config.functions.search(atest.config.variables
                                         ,atest.db.constants.redshift.name
                                         ,atest.db.constants.port)
    # driver connection
    drv  <- dbDriver("PostgreSQL")
    # disconnect any lingering connections
    conns<- dbListConnections(PostgreSQL())
    if( !is.null(conns) ) for( conn in conns ) atest.db.functions.disconnect.redshift(conn)
    # connection
    conn <- dbConnect(drv
                     ,dbname=db
                     ,host=host
                     ,port=port
                     ,user=user
                     ,password=pass)
    return(conn)
}

# oracle connection
atest.db.functions.connect.oracle <- function(url=NULL
                                             ,user=NULL
                                             ,pass=NULL
                                             ,host=NULL
                                             ,sid=NULL
                                             ,port=NULL
                                             ,class=NULL
                                             ,jar=NULL) {
    # url
    if( is.null(url) )
        url  <- atest.config.functions.search(atest.config.variables
                                             ,atest.db.constants.oracle.name
                                             ,atest.db.constants.url)
    # user
    if( is.null(user) )
        user <- atest.config.functions.search(atest.config.variables
                                             ,atest.db.constants.oracle.name
                                             ,atest.db.constants.user)
    # password
    if( is.null(pass) )
        pass <- atest.config.functions.search(atest.config.variables
                                             ,atest.db.constants.oracle.name
                                             ,atest.db.constants.pass)
    # host
    if( is.null(host) )
        host <- atest.config.functions.search(atest.config.variables
                                             ,atest.db.constants.oracle.name
                                             ,atest.db.constants.host)
    # system id
    if( is.null(sid) )
        sid  <- atest.config.functions.search(atest.config.variables
                                             ,atest.db.constants.oracle.name
                                             ,atest.db.constants.sid)
    # port number
    if( is.null(port) )
        port <- atest.config.functions.search(atest.config.variables
                                             ,atest.db.constants.oracle.name
                                             ,atest.db.constants.port)
    # class name
    if( is.null(class) )
        class<- atest.config.functions.search(atest.config.variables
                                             ,atest.db.constants.oracle.name
                                             ,atest.db.constants.class)
    # jar file location
    if( is.null(jar) )
        jar  <- atest.config.functions.search(atest.config.variables
                                             ,atest.db.constants.oracle.name
                                             ,atest.db.constants.jar)
    # driver connection
    drv  <- JDBC(class,jar,identifier.quote="'")
    # url string
    url  <- paste(url
                 ,host
                 ,':'
                 ,port
                 ,'/'
                 ,sid
                 ,sep="")
    # check if we are using oracle
    orcl <- atest.config.functions.search(atest.config.variables
                                         ,atest.db.constants.oracle.name
                                         ,atest.db.constants.use)
    # connection
    conn <- NULL
    if( !(is.null(orcl) || orcl == 0) )
        conn <- dbConnect(drv,url,user,pass)
    return(conn)
}

# oracle connection
atest.db.functions.connect.oracle.odbc <- function(user=NULL
                                                  ,pass=NULL
                                                  ,host=NULL
                                                  ,sid=NULL
                                                  ,port=NULL) {
    # user
    if( !is.null(user) )
        user <- atest.config.functions.search(atest.config.variables
                                             ,atest.db.constants.oracle.name
                                             ,atest.db.constants.user)
    # password
    if( !is.null(pass) )
        pass <- atest.config.functions.search(atest.config.variables
                                             ,atest.db.constants.oracle.name
                                             ,atest.db.constants.pass)
    # host
    if( !is.null(host) )
        host <- atest.config.functions.search(atest.config.variables
                                             ,atest.db.constants.oracle.name
                                             ,atest.db.constants.host)
    # port number
    if( !is.null(port) )
        port <- atest.config.functions.search(atest.config.variables
                                             ,atest.db.constants.oracle.name
                                             ,atest.db.constants.port)
    # system id
    if( !is.null(sid) )
        sid  <- atest.config.functions.search(atest.config.variables
                                             ,atest.db.constants.oracle.name
                                             ,atest.db.constants.sid)
    # driver connection
    drv  <- dbDriver("Oracle")
    cstr <- paste("(DESCRIPTION=                                      "
                 ,"             (ADDRESS=(PROTOCOL=tcp)               "
                 ,"                      (HOST=",host,")              "
                 ,"                      (PORT=",port,"))             "
                 ,"             (CONNECT_DATA=(SERVICE_NAME=",sid,")))"
                 ,sep="")
    # connection
    conn <- dbConnect(drv
                     ,username=user
                     ,password=pass
                     ,dbname=cstr)
    return(conn)
}

# sql server connection
atest.db.functions.connect.sqlserver <- function(url=NULL
                                                ,user=NULL
                                                ,pass=NULL
                                                ,host=NULL
                                                ,sid=NULL
                                                ,port=NULL
                                                ,class=NULL
                                                ,jar=NULL) {
    # url
    if( is.null(url) )
        url  <- atest.config.functions.search(atest.config.variables
                                             ,atest.db.constants.sqlserver.name
                                             ,atest.db.constants.url)
    # user
    if( is.null(user) )
        user <- atest.config.functions.search(atest.config.variables
                                             ,atest.db.constants.sqlserver.name
                                             ,atest.db.constants.user)
    # password
    if( is.null(pass) )
        pass <- atest.config.functions.search(atest.config.variables
                                             ,atest.db.constants.sqlserver.name
                                             ,atest.db.constants.pass)
    # host
    if( is.null(host) )
        host <- atest.config.functions.search(atest.config.variables
                                             ,atest.db.constants.sqlserver.name
                                             ,atest.db.constants.host)
    # port number
    if( is.null(port) )
        port <- atest.config.functions.search(atest.config.variables
                                             ,atest.db.constants.sqlserver.name
                                             ,atest.db.constants.port)
    # system id
    if( is.null(sid) )
        sid  <- atest.config.functions.search(atest.config.variables
                                             ,atest.db.constants.sqlserver.name
                                             ,atest.db.constants.sid)
    # class name
    if( is.null(class) )
        class<- atest.config.functions.search(atest.config.variables
                                             ,atest.db.constants.sqlserver.name
                                             ,atest.db.constants.class)
    # jar file location
    if( is.null(jar) )
        jar  <- atest.config.functions.search(atest.config.variables
                                             ,atest.db.constants.sqlserver.name
                                             ,atest.db.constants.jar)
    # driver connection
    drv  <- JDBC(class,jar,identifier.quote="'")
    # url string
    url  <- paste(url,host,':',port,';'
                 ,'databaseName=',sid,';'
                 ,'user=',user,';'
                 ,'password=',pass,';'
                 ,'encrypt=true;'
                 ,sep="")
    # connection
    conn <- dbConnect(drv,url)
    return(conn)
}

# mysql connection
atest.db.functions.connect.mysql <- function(user=NULL,pass=NULL,host=NULL,db=NULL) {
    # url
    url  <- atest.config.functions.search(atest.config.variables
                                         ,atest.db.constants.mysql.name
                                         ,atest.db.constants.url)
    # user
    user <- atest.config.functions.search(atest.config.variables
                                         ,atest.db.constants.mysql.name
                                         ,atest.db.constants.user)
    # password
    pass <- atest.config.functions.search(atest.config.variables
                                         ,atest.db.constants.mysql.name
                                         ,atest.db.constants.pass)
    # host
    host <- atest.config.functions.search(atest.config.variables
                                         ,atest.db.constants.mysql.name
                                         ,atest.db.constants.host)
    # database
    db   <- atest.config.functions.search(atest.config.variables
                                         ,atest.db.constants.mysql.name
                                         ,atest.db.constants.db)
    # port number
    port <- atest.config.functions.search(atest.config.variables
                                         ,atest.db.constants.mysql.name
                                         ,atest.db.constants.port)
    # class name
    class<- atest.config.functions.search(atest.config.variables
                                         ,atest.db.constants.mysql.name
                                         ,atest.db.constants.class)
    # jar file location
    jar  <- atest.config.functions.search(atest.config.variables
                                         ,atest.db.constants.mysql.name
                                         ,atest.db.constants.jar)
    # driver connection
    drv  <- JDBC(class,jar,identifier.quote="'")
    # url string
    url  <- paste(url
                 ,host
                 ,':'
                 ,port
                 ,'/'
                 ,db
                 ,sep="")
    # disconnect any lingering connections
    conns<- dbListConnections(MySQL())
    if( !is.null(conns) ) for( conn in conns ) atest.db.functions.disconnect.mysql(conn)
    # connection
    conn <- dbConnect(drv,url,user,pass)
    return(conn)
}

# specialized dbRemoveTable
atest.db.functions.dbRemoveTable <- function(con=NULL,tbl=NULL) {
    dat  <- NULL
    if( !(is.null(con) || is.null(tbl)) ) {
        csql <- paste("DROP TABLE",tbl,";")
        dat  <- dbSendStatement(con=con,csql)
        dbGetRowsAffected(dat)
        dbClearResult(dat)
    }
    return(dat)
}

# get the redshift data
atest.db.functions.data.redshift <- function() {
    dat  <- NULL
    # redshift connection
    conn <- atest.db.functions.connect.redshift()
    print(conn)
    # disconnect from redshift
    atest.db.functions.disconnect.redshift(conn)
    # return redshift data
    return(dat)
}

# get the oracle data
atest.db.functions.data.oracle <- function() {
    dat  <- NULL
    # oracle connection
    conn <- atest.db.functions.connect.oracle()
    print(conn)
    # disconnect from oracle
    atest.db.functions.disconnect.oracle(conn)
    # return oracle data
    return(dat)
}

# get the sql server data
atest.db.functions.data.sqlserver <- function() {
    dat  <- NULL
    # sqlserver connection
    conn <- atest.db.functions.connect.sqlserver()
    print(conn)
    # disconnect from sqlserver
    atest.db.functions.disconnect.sqlserver(conn)
    # return sqlserver data
    return(dat)
}

# get the mysql server data
atest.db.functions.data.mysql <- function(f=NULL) {
    tbl  <- "COUPONS"
    dat  <- NULL
    # mysql connection
    conn <- atest.db.functions.connect.mysql()
    if( !is.null(conn) ) {
        if( !is.null(f) ) {
            dat  <- read.csv(f)
            cdat <- colnames(dat)
            if( length(cdat) > 0 ) {
                #dbRemoveTable(conn,tbl)
                #stmt <- paste("CREATE TABLE",tbl,"(",paste(paste(cdat,"VARCHAR(30)"),collapse=","),");")
                #dbSendQuery(conn,stmt)
                #dbWriteTable(conn,dat[2:nrow(dat),],"COUPONS",append=TRUE)
                dbWriteTable(conn,tbl,dat,overwrite=TRUE)
            }
            #else
                #dbWriteTable(conn,dat,"COUPONS",append=FALSE)
        }
        else {
            stmt <- paste("SELECT * FROM",tbl,";")
            dat  <- dbGetQuery(conn,stmt)
        }
        # disconnect from mysql
        atest.db.functions.disconnect.mysql(conn)
    }
    # return sqlserver data
    return(dat)
}


# create the mysql server percentage data
atest.db.functions.pct.mysql <- function(d=NULL) {
    tbl  <- "PCTCOUPONS"
    dat  <- NULL
    # mysql connection
    if( !is.null(d) ) {
        conn <- atest.db.functions.connect.mysql()
        if( !is.null(conn) ) {
            cdat <- colnames(d)
            if( length(cdat) > 0 ) {
                #stmt <- paste("CREATE TABLE",tbl,"(",paste(paste(cdat,"VARCHAR(30)"),collapse=","),");")
                #dbSendQuery(conn,stmt)
                #dbWriteTable(conn,d[2:nrow(d),],"COUPONS",append=TRUE)
                if( dbExistsTable(conn,tbl) )
                    dbWriteTable(conn,tbl,d,append=TRUE,overwrite=FALSE)
                else
                    dbWriteTable(conn,tbl,d,append=FALSE,overwrite=TRUE)
            }
            #else
                #dbWriteTable(conn,dat,"COUPONS",append=FALSE)
        }
        stmt <- paste("SELECT * FROM",tbl,";")
        dat  <- dbGetQuery(conn,stmt)
        # disconnect from mysql
        atest.db.functions.disconnect.mysql(conn)
    }
    # return sqlserver data
    return(dat)
}

# test the sql server db
atest.db.functions.test.sqlserver <- function() {
    dat  <- NULL
    # sqlserver connection
    conn <- atest.db.functions.connect.sqlserver()
    if( !is.null(conn) ) {
        csql <- "SELECT COUNT(*) FROM SalesLT.Customer;"
        dat  <- dbGetQuery(con=conn,csql)
        print(dat)
        # disconnect from sqlserver
        atest.db.functions.disconnect.sqlserver(conn)
    }
    # return sqlserver data
    return(dat)
}

# create user in the sql server db
atest.db.functions.user.sqlserver <- function(conn=NULL,user=NULL,pass=NULL) {
    dat  <- NULL
    if( !(is.null(conn) || is.null(user) || is.null(pass)) ) {
        csql <- paste("CREATE USER ",user," WITH PASSWORD = '",pass,"'; COMMIT;"
                     ,"EXEC SP_ADDROLEMEMBER 'db_owner', ",user,";"
                     ,sep="")
        dat  <- dbSendStatement(con=conn,csql)
        print(dat)
        dbGetRowsAffected(dat)
        dbClearResult(dat)
    }
    # return sqlserver data
    return(dat)
}

# setup the sql server db
atest.db.functions.setup.sqlserver <- function(user=NULL,pass=NULL) {
    dat  <- NULL
    if( !(is.null(user) || is.null(pass)) ) {
        # sql server connection
        conn <- atest.db.functions.connect.sqlserver()
        if( !is.null(conn) ) {
            # create the user
            tryCatch(atest.db.functions.user.sqlserver(conn,user,pass),error=function(e) {},finally=NULL)
            # disconnect from sqlserver
            atest.db.functions.disconnect.sqlserver(conn)
        }
    }
    # return sqlserver data
    return(dat)
}
