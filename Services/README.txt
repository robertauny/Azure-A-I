1.  The services.py file is the main driver code and it picks up after the data from the different data sources has been collected and merged into the data warehouse.

2.  Within data.py, there are functions to access the data housed within the data warehouse, which relies, in large part, on the data definitions defined in the services.json configuration file.

3.  An instance number relates to a set of configuration items in services.json.  It is used to identify a set of related data elements to be used when creating the knowledge graph using the knowledge graph functions within services.py.

4.  The created knowledge graph can be read from (and written to) the knowledge graph DB using functions in data.py.

5.  Functions in ai.py allow for labeling clusters, generating a knowledge "brain", consisting of a set of related knowledge graphs and associated neural networks, each generated from a different permutation of the original data set.  Each data set is a different ordering of the inputs, resulting in a different hierarchy that is necessary for services when modeling one input as a function of all others.  In addition, there is an unfinished function "thought" within ai.py which is intended to return the result of a prediction from the appropriate model in the knowledge brain.

6.  # spark sqlContext should be used when writing to/from the graph DB
    # to create the data frame of edges and vertices in the following format
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

7.  Functions within config.py allow for reading (and writing) the configuration JSON file.

8.  The nn.py file contains the definition of the deep belief network (DBN) that does classification (for creation of the knowledge brain) and regression (for prediction using the data in one of the clusters defined by a particular knowledge brain).
