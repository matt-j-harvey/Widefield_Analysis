import numpy as mp
import matplotlib.pyplot as plt
import os
import sys

import numpy as np


def get_cluster_centroids(clustering, save_file):

    clusters = np.unique(clustering)

    cluster_centroid_list = []

    for cluster in clusters:
        if cluster != 0:
            cluster_mask = np.where(clustering == cluster, 1, 0)
            cluster_indexes = np.nonzero(cluster_mask)
            cluster_centroid = np.mean(cluster_indexes, axis=1)
            cluster_centroid_list.append(cluster_centroid)

    cluster_centroid_list = np.array(cluster_centroid_list)
    print("Cluster centroid list", np.shape(cluster_centroid_list))
    cluster_centroid_list.T[[0, 1]] = cluster_centroid_list.T[[1, 0]]

    np.save(save_file, cluster_centroid_list)
    plt.scatter(x=cluster_centroid_list[:, 0], y=cluster_centroid_list[:, 1])
    plt.show()


# Load Consensus Clustters
consensus_clusters = np.load(r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Final_Consensus_Clusters.npy")
save_file = r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/cluster_centroids.npy"
get_cluster_centroids(consensus_clusters, save_file)