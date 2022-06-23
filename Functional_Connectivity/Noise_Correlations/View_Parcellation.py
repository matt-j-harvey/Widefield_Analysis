import numpy as np
from skimage.feature import canny
import matplotlib.pyplot as plt
import os


# Load Clusters
clusters = np.load(os.path.join("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Final_Consensus_Clusters.npy"), allow_pickle=True)[()]
plt.imshow(clusters)
plt.show()
# View Edge
edges_array = np.zeros(np.shape(clusters))
unique_clusters = list(np.unique(clusters))

for cluster in unique_clusters:
    cluster_mask = np.where(clusters == cluster, 1, 0)
    edges = canny(cluster_mask.astype('float32'))
    edge_indicies = np.nonzero(edges)
    edges_array[edge_indicies] = 1

plt.imshow(edges_array, cmap="Greys")
plt.show()