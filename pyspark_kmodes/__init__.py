__version__ = (0, 1, 0)

# capture dependency on PySpark at run time:

#try:
#    from pyspark.rdd import RDD
#except ImportError:
#    raise ImportError("PySpark must be installed and available in PYTHONPATH")

from .Kmodes import KModes

from .pyspark_kmodes import (
    Cluster,
    k_modes_record,
    EnsembleKModes,
    EnsembleKModesModel,
    with_2_arguments,
    with_arguments,
    iter_k_modes,
    get_max_value_key,
    matching_dissim,
    k_modes_partitioned,
    run_local_kmodes,
    check_for_empty_cluster,
    get_cluster_rdd,
    get_cluster_record
)
