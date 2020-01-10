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

import config
import sys
import traceback

# read the default json config file
cfg  = config.cfg()

############################################################################
##
## Purpose:   Read/write a knowledge graph from/to a graph DB using spark
##
############################################################################
def rw_kg(inst,df=None,testing=True):
    if not (inst == None or inst < 0):
        # ordering of the data elements in the JSON file
        kg   = cfg["instances"][inst]["kg"]
        # configuration parameters
        host = cfg["instances"][inst]["sources"][kg]["connection"]["host"]
        port = cfg["instances"][inst]["sources"][kg]["connection"]["port"]
        acc  = cfg["instances"][inst]["sources"][kg]["connection"]["acc" ]
        db   = cfg["instances"][inst]["sources"][kg]["connection"]["db"  ]
        app  = cfg["instances"][inst]["sources"][kg]["connection"]["app" ]
        key  = cfg["instances"][inst]["sources"][kg]["connection"]["key"]
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
                    "Endpoint": url,
                    "Masterkey": key,
                    "Database": dirn,
                    "Collection": table,
                    "Upsert": "true"
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
                        "Endpoint": url,
                        "Masterkey": key,
                        "Database": db,
                        "Collection": table,
                        "ReadChangeFeed": "true",
                        "ChangeFeedQueryName": dirn,
                        "ChangeFeedStartFromTheBeginning": "false",
                        "InferStreamSchema": "true",
                        "ChangeFeedCheckpointLocation": "dbfs:/" + dirn
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
def blob_conf_set(inst,acc,testing=True):
    ret  = False
    if not (inst == None or inst < 0 or acc == None):
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
def rw_warehouse(inst,df=None,testing=True):
    if not (inst == None or inst < 0):
        # ordering of the data elements in the JSON file
        wh   = cfg["instances"][inst]["wh"]
        # configuration parameters
        host = cfg["instances"][inst]["sources"][wh]["connection"]["host"]
        port = cfg["instances"][inst]["sources"][wh]["connection"]["port"]
        acc  = cfg["instances"][inst]["sources"][wh]["connection"]["acc" ]
        db   = cfg["instances"][inst]["sources"][wh]["connection"]["db"  ]
        app  = cfg["instances"][inst]["sources"][wh]["connection"]["app" ]
        key  = cfg["instances"][inst]["sources"][wh]["connection"]["key"]
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
