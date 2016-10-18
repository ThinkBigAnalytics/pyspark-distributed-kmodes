.. image:: docs/images/thinkbig.png

##########################
pyspark-distributed-kmodes
##########################

Ensemble based distributed K-modes clustering for PySpark
---------------------------------------------------------

This repository contains the source code for the `pyspark_kmodes` package to perform K-modes clustering in PySpark. The package implements the ensemble-based algorithm proposed by Visalakshi and Arunprabha (IJERD, March 2015).

K-modes clustering is performed on each partition of a Spark RDD, and the resulting clusters are collected to the driver node. Local K-modes clustering is then performed on the centroids returned from each partition to yield a final set of cluster centroids.

This package was written by `Marissa Saunders <marissa.saunders@thinkbiganalytics.com>`_ and relies on an adaptation of the KModes package by Nico de Vos `https://github.com/nicodv/kmodes <https://github.com/nicodv/kmodes>`_ for the local iterations. Using this package for clustering Clickstream data is described by Marissa Saunders in this `YouTube video <https://www.youtube.com/watch?v=1fYBTehHHIU>`_.


Installation
------------

This module has been developed and tested on Spark 1.5.2 and 1.6.1 and should work under Python 2.7 and 3.5.

The module depends on scikit-learn 0.16+ (for ``check_array``). See ``requirements.txt`` for this and other package dependencies.

Once cloned or downloaded, execute ``pip`` from the top-level directory to install:

::

    $ ls
    LICENSE			README.rst		pyspark_kmodes		setup.cfg
    MANIFEST.in		docs			requirements.txt	setup.py

    $ pip install .
    [...]


Getting Started
---------------

The ``docs`` directory includes a sample Jupyter/iPython notebook to demonstrate its use.

::

    $ cd docs

    $ jupyter notebook PySpark-Distributed-KModes-example.ipynb 


References
----------

* NK Visalakshi and K Arunprabha, 2015. *Ensemble based Distributed K-Modes Clustering*, International Journal of Engineering Research and Development, Vol. 11, No. 3, pp.79-89, `http://files.figshare.com/2011247/J1137989.pdf <http://files.figshare.com/2011247/J1137989.pdf>`_.

* Zhexue Huang, 1998. *Extensions to the k-Means Algorithm for Clustering Large Data Sets with Categorical Values*, Data Mining and Knowledge Discovery 2, pp. 283–304, `http://www.cse.ust.hk/~qyang/537/Papers/huang98extensions.pdf <http://www.cse.ust.hk/~qyang/537/Papers/huang98extensions.pdf>`_.
