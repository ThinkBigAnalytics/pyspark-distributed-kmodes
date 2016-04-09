# Author: 'Marissa Saunders' <marissa.saunders@thinkbiganalytics.com> 
# License: MIT

from collections import defaultdict
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array
import random

from .Kmodes import KModes


"""
Ensemble based distributed K-modes clustering for PySpark

This module uses the algorithm proposed by Visalakshi and Arunprabha (IJERD, March 2015) to perform K-modes clustering in an ensemble-based way.

K-modes clustering is performed on each partition of an rdd and the resulting clusters are collected to the driver node.  Local K-modes clustering is then performed on the centroids returned from each partition to yield a final set of cluster centroids.

Relies on an adaptation of the KModes package from https://github.com/nicodv/kmodes for the local iterations

"""


def with_2_arguments(mapper):
    """ 
    Python decorator that wraps a map, filter, or other similar Spark function to enable it to take multiple parameters initiated at time of call
            
    - inputs:
        - *mapper*: the function you intend to wrap
        
    - outputs:
        - *return value*: a function wrapping the passed function

    - example:
        @with_argument
        def my_mapper(element, multiplier):
            return element * multiplier
    
        rdd = sc.parallelize([1,2,3])
        rdd.map(my_mapper(5))
        # returns: [5, 10, 15]
    
    """
    def instantiate(*args, **kwargs):
        def map_with_arguments(a,b):
            return mapper(a,b, *args, **kwargs)
        return map_with_arguments
    return instantiate

def with_arguments(mapper):
    """ 
    Python decorator that wraps a map, filter, or other similar Spark function to enable it to take multiple parameters initiated at time of call
            
    - inputs:
        - *mapper*: the function you intend to wrap
        
    - outputs:
        - *return value*: a function wrapping the passed function

    - example:
        @with_argument
        def my_mapper(element, multiplier):
            return element * multiplier
    
        rdd = sc.parallelize([1,2,3])
        rdd.map(my_mapper(5))
        # returns: [5, 10, 15]
    
    """
    def instantiate(*args, **kwargs):
        def map_with_arguments(a):
            return mapper(a, *args, **kwargs)
        return map_with_arguments
    return instantiate


@with_2_arguments
def iter_k_modes(split_Index, iterator, clusters, n_clusters):
    """ 
    Function that is used with mapPartitionsWithIndex to perform a single iteration
    of the k-modes algorithm on each partition of data.
    
        - Inputs
        
            - *clusters*: is a list of cluster objects for all partitions, 
            - *n_clusters*: is the number of clusters to use on each partition
        
        - Outputs  
            
            - *clusters*: a list of updated clusters,
            - *moved*: the number of data items that changed clusters
    """
    moved = 0
    clusters = clusters[int(n_clusters*(split_Index)):int(n_clusters*(split_Index+1))]
    for record in iterator:  
        clusters, temp_move = record.update_cluster(clusters)
        moved += temp_move
    yield (clusters, moved)
    
    


class Cluster:
    """
    This is the k-modes cluster object 
    
    - Initialization:
            - just the centroid
    - Structure:
    
            - the cluster mode (.centroid),
            - the index of each of the rdd points in the cluster(.members)
            - the frequency at which each of the values is observed for each category in each variable calculated over the cluster members (.freq)
    
    - Methods:
    
            - add_member(record): add a data point to the cluster
            - subtract_member(record): remove a data point from the cluster
            - update_mode: recalculate the centroid of the cluster based on the frequencies.
            
    """
    

    def __init__(self, centroid):
        self.centroid = centroid
        self.freq = [defaultdict(int) for _ in range(len(centroid))]
        self.members = []
    
    def add_member(self, record):
        for ind_attr, val_attr in enumerate(record.record):
            self.freq[ind_attr][val_attr] += 1
        self.members.append(record.index)
    
    def subtract_member(self, record):
        for ind_attr, val_attr in enumerate(record.record):
            self.freq[ind_attr][val_attr] -= 1
        self.members.remove(record.index)
        
    def update_mode(self):
        new_centroid = []
        for ind_attr, val_attr in enumerate(self.centroid):
            new_centroid.append(get_max_value_key(self.freq[ind_attr]))
        self.centroid = new_centroid
        
    

class k_modes_record:
    
    """ A single item in the rdd that is used for training the k-modes 
    calculation.  
    
        - Initialization:
            - A tuple containing (Index, DataPoint)
        
        - Structure:
            - the index (.index)
            - the data point (.record)
        
        - Methods:
            - update_cluster(clusters): determines which cluster centroid is closest to the data point and updates the cluster membership lists appropriately.  It also updates the frequencies appropriately.
    """
    
    def __init__(self,record ):
        self.record = record[1]
        self.index = record[0]
 
    def update_cluster(self, clusters):
        # clusters contains a list of cluster objects.  This function calculates which cluster is closest to the
        # record contained in this object and changes the cluster to contain the index of this mode.
        # It also updates the cluster frequencies.
        
        modes_arr = [cluster.centroid for cluster in clusters]
        
        diss = matching_dissim(self.record,modes_arr)
        new_cluster = np.argmin(diss)
        
        old_cluster = None
        moved = 0
        for cluster_num, cluster in enumerate(clusters):
                if self.index in cluster.members:
                    old_cluster = cluster_num
                    break
        if old_cluster is None:
            # First cycle through
            moved += 1
            old_cluster = new_cluster
            clusters[new_cluster].add_member(self)
            clusters[new_cluster].update_mode()
            
        else:
            
            if old_cluster == new_cluster:
                pass
        
            elif new_cluster != old_cluster:
                moved +=1
                clusters[old_cluster].subtract_member(self)
                clusters[old_cluster].update_mode()
                clusters[new_cluster].add_member(self)
                clusters[new_cluster].update_mode()
                
                old_cluster = new_cluster
                
        return (clusters, moved)

    def get_cost_and_cluster(self, clusters):
        modes_arr = [cluster.centroid for cluster in clusters]
        diss = matching_dissim(self.record,modes_arr)
        new_cluster = np.argmin(diss)
        cost = diss[new_cluster]
        return (self.index, new_cluster, cost)
    
def get_max_value_key(dic):
    """
    Fast method to get key for maximum value in dict.
    From https://github.com/nicodv/kmodes
    """
    v = list(dic.values())
    k = list(dic.keys())
    return k[v.index(max(v))]
    
def matching_dissim(a, b):
    """
    Simple matching dissimilarity function
    adapted from https://github.com/nicodv/kmodes
    """
    list_dissim = []
    for compare_to in b: 
        list_dissim.append(sum(elem1 != elem2 for elem1,elem2 in zip(a, compare_to)))
    return list_dissim

def k_modes_partitioned(data_rdd, n_clusters, max_iter, seed = None):
    
    """
    Perform a k-modes calculation on each partition of data.
    
        - Input:
            - *data_rdd*: in the form (index, record). Make sure that the data is partitioned appropriately: i.e. spread across partitions, and a relatively large number of data points per partition.
            - *n_clusters*: the number of clusters to use on each partition
            - *max_iter*: the maximum number of iterations
            - *seed*:  controls the sampling of the initial clusters from the data_rdd
            
        - Output:
            - *clusters*: the final clusters for each partition
            - *rdd*: rdd containing the k_modes_record objects
    """
    n_partitions = data_rdd.getNumPartitions()
    rdd = data_rdd.map(lambda x: k_modes_record(x))
    
    # Create clusters - random sampling of records
    clusters = [Cluster(centroid.record) for centroid in rdd.takeSample(False, n_partitions * n_clusters, seed=None)]
    # On each partition do an iteration of k modes analysis, passing back the final clusters. Repeat until no points move

    did_it_move = True
    iter_count = 0
    while did_it_move:
        moved = 0
        print("Iteration ", iter_count)
        iter_count +=1
        clusters = rdd.mapPartitionsWithIndex(iter_k_modes(clusters, n_clusters))
        new_clusters = []
        for cluster in clusters.collect():
            moved = moved + cluster[1]
            for m in cluster[0]:
                new_clusters.append(m)
        
        clusters = new_clusters
        # Check for empty clusters and reinitialize with random element from biggest cluster is needed
        clusters = check_for_empty_cluster(clusters, rdd)
        if moved == 0:
            did_it_move = False 
        if iter_count >= max_iter:
            break
    
    return clusters, rdd

def run_local_kmodes(clusters, n_clusters,init = 'Cao', n_init = 1, verbose = 1):
    """ 
    Perform local k-modes calculations on the clusters from all partitions
        -Input:
            - *clusters*: the list of cluster objects returned from the partitioned k-modes calculation,
            - *n_clusters*: the number of final clusters wanted (should be the same as the number of clusters per mode)
            - *init*: (optional) initialization method; defaults to 'Cao'
            - *n_init*: (optional) the number of times to run the clustering algorithm (default: 1)
            - *verbose*: (optional) verbosity of output (default: 1)
    """
    # Now do k-modes on the main machine
    km = KModes(n_clusters = n_clusters, init = init, n_init = n_init, verbose = verbose)
    new_centroids = [cluster.centroid for cluster in clusters]
    new_modes = km.fit(new_centroids, dtype = "object")
    return [list(new_modes.cluster_centroids_), new_modes.cost_]


def check_for_empty_cluster(clusters, rdd):
    """ Checks if there are any clusters with no members and re-initializes any empty clusters with
    a random member of the largest cluster on that partition
    
        - Input:
            - *clusters*: a list of cluster objects
            - *rdd*: the rdd of k-modes_record objects that the clusters were trained on.
        - Output:
            -*clusters*: the updated list of clusters
            
    """
    import random
    n_partitions = rdd.getNumPartitions()
    n_clusters = len(clusters)/n_partitions
    cluster_sizes = [len(cluster.members) for cluster in clusters]
    for index, size in enumerate(cluster_sizes):
        
        if size == 0:
            partition_index = int(np.floor(index/n_partitions))
            partition_sizes = cluster_sizes[n_clusters*(partition_index):n_clusters*(partition_index+1)]
            biggest_cluster = np.argmax(np.array(partition_sizes)) + n_clusters*(partition_index)
            random_element = random.choice(clusters[biggest_cluster].members)
            new_centroid = rdd.filter(lambda x: x.index == random_element).map(lambda x: x.record).collect()[0]
            clusters[index].centroid = new_centroid
    return clusters
            


class EnsembleKModes:
    
        """Uses the algorithm proposed by Visalakshi and Arunprabha (IJERD, March 2015) to perform K-modes clustering in an ensemble-based way.

        K-modes clustering is performed on each partition of an rdd and the resulting clusters are collected to the driver node.  Local K-modes clustering is then performed on the centroids returned from each partition to yield a final set of cluster centroids.

        >>> from pyspark_kmodes import *
        >>> sc.addPyFile("pyspark_kmodes.py")
        
        >>> from random import shuffle
        >>> ## Arguments / Variables
        >>> n_modes = 2
        >>> set_partitions = 32
        >>> max_iter = 10

        >>> # Create the data set
        >>> data = np.random.choice(["a", "b", "c"], (50000, 10))
        >>> data2 = np.random.choice(["e", "f", "g"], (50000, 10))
        >>> data = list(data) + list(data2)
        >>> shuffle(data)

        >>> # Create the rdd
        >>> rdd = sc.parallelize(data)
        >>> rdd = rdd.coalesce(set_partitions)

        >>> method = EnsembleKModes(2, 10)
        >>> model = method.fit(rdd)
        
        Init: initializing centroids
        Init: initializing clusters
        Starting iterations...
        Run 1, iteration: 1/100, moves: 1, cost: 376.0
        Run 1, iteration: 2/100, moves: 0, cost: 376.0
        Avg cost/partition: 5.875
        Final centroids:
        [['c' 'a' 'a' 'a' 'c' 'a' 'b' 'a' 'a' 'a']
         ['f' 'e' 'g' 'f' 'f' 'e' 'e' 'f' 'f' 'g']]
        
        >>> print model.clusters
        [['c' 'a' 'a' 'a' 'c' 'a' 'b' 'a' 'a' 'a']
         ['f' 'e' 'g' 'f' 'f' 'e' 'e' 'f' 'f' 'g']]    
         
        >>> print method.mean_cost
        6.64787
        
        >>> print model.clusters
        >>> predictions = method.predictions
        >>> datapoints = method.indexed_rdd
        >>> combined = datapoints.zip(predictions)
        >>> print combined.take(10)
        [['c' 'a' 'a' 'a' 'c' 'a' 'b' 'a' 'a' 'a']
         ['f' 'e' 'g' 'f' 'f' 'e' 'e' 'f' 'f' 'g']]
        [((0, array(['e', 'e', 'f', 'e', 'e', 'f', 'g', 'e', 'f', 'e'], 
              dtype='|S1')), (0, 1)), ((1, array(['b', 'c', 'a', 'b', 'a', 'c', 'a', 'a', 'c', 'b'], 
              dtype='|S1')), (1, 0)), ((2, array(['e', 'g', 'f', 'f', 'g', 'g', 'e', 'e', 'e', 'e'], 
              dtype='|S1')), (2, 1)), ((3, array(['e', 'g', 'f', 'f', 'f', 'f', 'g', 'f', 'g', 'g'], 
              dtype='|S1')), (3, 1)), ((4, array(['b', 'b', 'b', 'c', 'a', 'c', 'b', 'c', 'c', 'c'], 
              dtype='|S1')), (4, 0)), ((5, array(['g', 'e', 'e', 'f', 'f', 'e', 'e', 'e', 'g', 'g'], 
              dtype='|S1')), (5, 1)), ((6, array(['b', 'a', 'a', 'c', 'b', 'b', 'a', 'b', 'b', 'c'], 
              dtype='|S1')), (6, 0)), ((7, array(['a', 'a', 'c', 'b', 'a', 'b', 'b', 'b', 'b', 'c'], 
              dtype='|S1')), (7, 0)), ((8, array(['c', 'b', 'a', 'b', 'c', 'a', 'b', 'b', 'c', 'a'], 
              dtype='|S1')), (8, 0)), ((9, array(['a', 'c', 'a', 'a', 'b', 'c', 'b', 'b', 'a', 'a'], 
              dtype='|S1')), (9, 0))]

        >>> model.predict(rdd).take(5)
        
        [1, 0, 1, 1, 0]
        
        >>> model.predict(['e', 'e', 'f', 'e', 'e', 'f', 'g', 'e', 'f', 'e'])
        
        1
        


        """
        def __init__(self, n_clusters, max_dist_iter, dist_seed = None, local_kmodes_iter = 1, local_init_method = 'Cao', verbosity = 1):   
            
            self.n_clusters = n_clusters
            self.max_dist_iter = max_dist_iter
            self.dist_seed = dist_seed
            self.local_kmodes_iter = local_kmodes_iter
            self.init_method = local_init_method
            self.verbosity = verbosity
            
        def fit(self, rdd):
            
            
            
            """ Compute distributed k-modes clustering.
            
                - Input:
    
                - *rdd*: data rdd
                
                - Modifies:
                    -*.indexed_rdd*: the training data with an index
                    -*.partitions*: the number of partitions the data is distributed over
                    -*.mean_cost*: the mean cost for the training data
                    -*.predictions*: the predictions for the training data with an index
            """

            self.indexed_rdd = rdd.zipWithIndex().map(lambda x: (x[1], x[0]))
            self.partitions = self.indexed_rdd.getNumPartitions()
            
            # Calculate the modes for each partition and return the clusters and an indexed rdd.
            clusters, objects_rdd = k_modes_partitioned(self.indexed_rdd,
                                                        self.n_clusters,self.max_dist_iter)

            # Calculate the modes locally for the set of all modes

            local_clusters = run_local_kmodes(clusters, self.n_clusters)
            if self.verbosity:
                print("Avg cost/partition:", local_clusters[1]/len(clusters))
                print("Final centroids:")
                for c in local_clusters[0]:
                    print(c)
        
            # Calculate the cost over the entire rdd.
            
            new_clusters = []
            for c in local_clusters[0]:
                new_clusters.append(Cluster(c))
                
            clusters_costs = objects_rdd.map(lambda x: x.get_cost_and_cluster(new_clusters))
            
            self.mean_cost = clusters_costs.map(lambda x: x[2]).mean()
            
            self.predictions = clusters_costs.map(lambda x:(x[0],x[1]))
            
            model = EnsembleKModesModel(local_clusters[0])
            
            return model
        
@with_arguments
def get_cluster_rdd(record, clusters):
        diss = matching_dissim(record,clusters)
        new_cluster = np.argmin(diss)
        return (new_cluster)

def get_cluster_record(record, clusters):
        diss = matching_dissim(record,clusters)
        new_cluster = np.argmin(diss)
        return (new_cluster)
    

        
class EnsembleKModesModel:
    
    def __init__(self, clusters):
        
        self.clusters = clusters
        
    
    def predict(self, data):
        import pyspark
        if isinstance(data, pyspark.rdd.RDD):
            
            predictions = data.map(get_cluster_rdd(self.clusters))
        
        else:
            try:
                data_len = len(data)
            except:
                print("Cannot take length of", type(data))
            
            assert(data_len == len(self.clusters[0])), "data is not the same length as cluster centroids"
            predictions = get_cluster_record(data, self.clusters)
        
        return predictions

