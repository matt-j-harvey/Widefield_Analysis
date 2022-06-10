import numpy as mp
import matplotlib.pyplot as plt
import os
import sys

import numpy as np
from skimage.feature import canny

def get_cluster_outlines(clustering, save_file):

    clusters = np.unique(clustering)
    edge_array = np.zeros(np.shape(clustering))

    for cluster in clusters:
        if cluster != 0:
            cluster_mask = np.where(clustering == cluster, 1, 0)
            cluster_edges = canny(cluster_mask.astype('float32'))
            edge_indexes = np.nonzero(cluster_edges)
            edge_array[edge_indexes] = 1


    np.save(save_file, edge_array)
    plt.imshow(edge_array)
    plt.show()


# Load Consensus Clustters
consensus_clusters = np.load(r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Final_Consensus_Clusters.npy")
save_file = r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/cluster_outlines.npy"
get_cluster_outlines(consensus_clusters, save_file)