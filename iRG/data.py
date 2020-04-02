#!/usr/bin/python

############################################################################
##
## File:      data.py
##
## Purpose:   Data utilities for connecting and reading/writing data.
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Dec. 24, 2019
##
############################################################################

from joblib import Parallel, delayed

import config
import sys
import traceback

import numpy           as np
import multiprocessing as mp

import constants as const

# read the default json config file
cfg  = config.cfg()

############################################################################
##
## Purpose:   Heavy lift of writing a knowledge graph vertices and edges
##
############################################################################
def write_ve(stem=None,coln=[],kgdat=[],g=None):
    ret  = None
    lcol = len(coln)
    lkg0 = len(kgdat)
    # creating vertices/edges for exactly one KG
    if not (stem == None or lcol == 0 or lkg0 == 0 or g == None):
        if stem == const.V:
            # create IDs using the ID defined by the vertices
            ret  = np.asarray(kgdat[0].split("-"))
            ret  = ["-".join(ret[range(0,len(ret)-1)])]
            for i in range(0,len(kgdat)):
                if i == 0:
                    # add the ID
                    g    = g.addV(stem).property("id",kgdat[i])
                else:
                    # add all of the other properties
                    if i == len(kgdat)-1:
                        ret.append(g.property(coln[i-1],kgdat[i]).next())
                        g    = ret[1]
                    else:
                        g    = g.property(coln[i-1],kgdat[i])
    return ret

############################################################################
##
## Purpose:   Heavy lift of writing a knowledge graph with needed NLP
##
############################################################################
def write_nlp(stem=None,coln=[],kgdat=[],g=None):
    ret  = None
    if not (stem == None or len(coln) == 0 or len(kgdat) == 0 or g == None):
        if stem in [const.V,const.E]:
            # returns for the current stem, one KG row at a time
            ret  = [write_ve(const.V,coln,k,g) for k in kgdat[const.V]]
            # possibly have multiple clusters and multiple rows in that cluster
            for clus in kgdat[const.E]:
                for blk in clus:
                    for row in blk:
                        row1 = int(row[1])
                        row2 = int(row[2])
                        # create the ID
                        ids1 = ret[row1-1][0] + "-" + str(row1)
                        ids2 = ret[row2-1][0] + "-" +                   str(row2)
                        ids12= ret[row1-1][0] + "-" + str(row1) + "-" + str(row2)
                        # get vertices associated to the IDs in question
                        v1   = ret[row1-1][1]
                        v2   = ret[row2-1][1]
                        # create the edge between the vertices in question
                        g.V(v1).addE(const.E).to(v2).property("id",ids12).next()
        else:
            # just a placeholder for the moment
            # call the appropriate function in the future
            # then append things to the graph using property tags
            ret  = None
    return ret

############################################################################
##
## Purpose:   Heavy lift of writing a knowledge graph
##
############################################################################
def write_kg(coln=[],kgdat=[],g=None):
    ret  = None
    lcol = len(coln)
    lkg  = len(kgdat)
    if not (lcol == 0 or lkg == 0 or g == None):
        # returns for each stem, one KG row at a time
        ret  = [write_nlp(stem,coln,kgdat,g) for stem in const.STEMS]
    return ret

############################################################################
##
## Purpose:   Read/write a knowledge graph from/to a graph DB using spark
##
############################################################################
def rw_kg(inst=const.BVAL,df=None,testing=True):
    if not (inst <= const.BVAL):
        # ordering of the data elements in the JSON file
        kg   = cfg["instances"][inst]["kg"]
        # configuration parameters
        host = cfg["instances"][inst]["sources"][kg]["connection"]["host"]
        port = cfg["instances"][inst]["sources"][kg]["connection"]["port"]
        acc  = cfg["instances"][inst]["sources"][kg]["connection"]["acc" ]
        db   = cfg["instances"][inst]["sources"][kg]["connection"]["db"  ]
        app  = cfg["instances"][inst]["sources"][kg]["connection"]["app" ]
        key  = cfg["instances"][inst]["sources"][kg]["connection"]["key" ]
        sec  = cfg["instances"][inst]["sources"][kg]["connection"]["sec" ]
        con  = cfg["instances"][inst]["sources"][kg]["connection"]["con" ]
        dirn = cfg["instances"][inst]["sources"][kg]["connection"]["dirn"]
        tbl  = cfg["instances"][inst]["sources"][kg]["connection"]["tbl" ]
        if not (host == None or
                port == None or
                acc  == None or
                db   == None or
                app  == None or
                sec  == None or
                con  == None or
                dirn == None or
                tbl  == None):
            url  = "https://" + host + ":" + port
            if not testing:
                # Write configuration
                wc = {
                    "Endpoint"  : url,
                    "Masterkey" : key,
                    "Database"  : dirn,
                    "Collection": table,
                    "Upsert"    : "true"
                }
                # spark sqlContext should be used to create the data frame
                # of edges and vertices in the following format
                #
                # vertices
                #
                # unique id, column listing
                # unique id column name, column listing name
                #
                # df = sqlContext.createDataFrame( [ ("id1","val11","val12",...,"val1n"),
                #                                    ("id2","val21","val22",...,"val2n"),
                #                                    ...,
                #                                    ["id","col1","col2",...,"coln"] ] )
                #
                # edges
                #
                # entity 1, related to entity 2, relationship
                # source entity, destination entity, relationship heading
                #
                # df = sqlContext.createDataFrame( [ ("id1","id2","rel12"),
                #                                    ("id2","id3","rel23"),
                #                                    ...,
                #                                    ("idm","idn","relmn"),
                #                                    ...,
                #                                    ["src","dst","relationship"] ] )
                if not df == None:
                    df.write                                        \
                      .format("com.microsoft.azure.cosmosdb.spark") \
                      .options(**wc                               ) \
                      .save()
                else:
                    # Read Configuration
                    rc = {
                        "Endpoint"                       : url,
                        "Masterkey"                      : key,
                        "Database"                       : db,
                        "Collection"                     : table,
                        "ReadChangeFeed"                 : "true",
                        "ChangeFeedQueryName"            : dirn,
                        "ChangeFeedStartFromTheBeginning": "false",
                        "InferStreamSchema"              : "true",
                        "ChangeFeedCheckpointLocation"   : "dbfs:/" + dirn
                    }
                    # Open a read stream to the Cosmos DB Change Feed via azure-cosmosdb-spark to create Spark DataFrame
                    df = (spark.readStream                                                                    \
                               .format("com.microsoft.azure.cosmosdb.spark.streaming.CosmosDBSourceProvider") \
                               .options(**rc                                                                ) \
                               .load())
            else:
                print(url)
    return df

############################################################################
##
## Purpose:   Housekeeping done when logging into a graph DB.
##
############################################################################
def blob_conf_set(inst=const.BVAL,acc=None,testing=True):
    ret  = False
    if not (inst <= const.BVAL or acc == None):
        # ordering of the data elements in the JSON file
        wh   = cfg["instances"][inst]["wh"]
        # configuration parameters
        key  = cfg["instances"][inst]["sources"][wh]["connection"]["key"]
        cstr = "fs.azure.account.key." + acc + ".blob.core.windows.net"
        if not testing:
            spark.conf.set(cstr,key)
        else:
            print(key)
            print(cstr)
        ret  = True
    return ret

############################################################################
##
## Purpose:   Read/write data from/to a data warehouse using spark
##
############################################################################
def rw_warehouse(inst=const.BVAL,df=None,testing=True):
    if not (inst <= const.BVAL):
        # ordering of the data elements in the JSON file
        wh   = cfg["instances"][inst]["wh"]
        # configuration parameters
        host = cfg["instances"][inst]["sources"][wh]["connection"]["host"]
        port = cfg["instances"][inst]["sources"][wh]["connection"]["port"]
        acc  = cfg["instances"][inst]["sources"][wh]["connection"]["acc" ]
        db   = cfg["instances"][inst]["sources"][wh]["connection"]["db"  ]
        app  = cfg["instances"][inst]["sources"][wh]["connection"]["app" ]
        key  = cfg["instances"][inst]["sources"][wh]["connection"]["key" ]
        sec  = cfg["instances"][inst]["sources"][wh]["connection"]["sec" ]
        con  = cfg["instances"][inst]["sources"][wh]["connection"]["con" ]
        dirn = cfg["instances"][inst]["sources"][wh]["connection"]["dirn"]
        tbl  = cfg["instances"][inst]["sources"][wh]["connection"]["tbl" ]
        if not (host == None or
                port == None or
                acc  == None or
                db   == None or
                app  == None or
                sec  == None or
                con  == None or
                dirn == None or
                tbl  == None):
            url  = "jdbc:sqlserver://" + host + ":" + port + ";databaseName=" + db + ";integratedSecurity=" + sec + ";applicationName=" + app
            wasbs= "wasbs://" + con + "@" + acc + ".blob.core.windows.net/" + dirn
            blob_conf_set(inst,acc,testing)
            if not testing:
                if not df == None:
                    df.write                                                \
                      .format("com.databricks.spark.sqldw"                ) \
                      .option("url"                                ,url   ) \
                      .option("forwardSparkAzureStorageCredentials","true") \
                      .option("dbTable"                            ,tbl   ) \
                      .option("tempDir"                            ,wasbs ) \
                      .save()
                else:
                    qry  = "select * from " + tbl + ";"
                    df   = spark.read                                                 \
                                .format("com.databricks.spark.sqldw"                ) \
                                .option("url"                                ,url   ) \
                                .option("tempDir"                            ,wasbs ) \
                                .option("forwardSparkAzureStorageCredentials","true") \
                                .option("query"                              ,qry   ) \
                                .load()
            else:
                print(url)
                print(wasbs)
    return df

#***************** TESTING ********************

def data_testing(inst=0):
    ret  = rw_warehouse(0)
    print(ret)
    ret  = rw_warehouse(0,"testing")
    print(ret)
    ret  = rw_kg(0)
    print(ret)
    ret  = rw_kg(0,"testing")
    print(ret)
    # demo of data source info
    src  = cfg["instances"][inst]["src"]["index"]
    mysql= cfg["instances"][inst]["src"]["types"]["mysql"]
    host = cfg["instances"][inst]["sources"][src][mysql]["connection"]["host"]
    print(str(src)+"-"+str(mysql)+"-"+host)
