import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny



def smooth_contours(pixel_assignments):

    unique_clusters = list(np.unique(pixel_assignments))
    smoothed_template = np.zeros(np.shape(pixel_assignments))

    for cluster in unique_clusters:

        cluster_mask = np.where(pixel_assignments == cluster, 1, 0)

        edges = canny(cluster_mask.astype('float32'), sigma=5)
        edge_indexes = np.nonzero(edges)
        smoothed_template[edge_indexes] = 1

    return smoothed_template




cluster_assignments = np.load("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Mirrored_Curated_Clusters.npy")

plt.imshow(cluster_assignments, cmap='flag')
plt.show()


unique_clusters = list(np.unique(cluster_assignments))
unique_clusters.sort()

new_cluster_assigments = np.zeros(np.shape(cluster_assignments))
for cluster in unique_clusters:
    cluster_index = unique_clusters.index(cluster)

    new_cluster_assigments = np.where(cluster_assignments == cluster, cluster_index, new_cluster_assigments)

cluster_assignments = new_cluster_assigments
plt.imshow(cluster_assignments)
plt.show()


thrity_indexes = np.nonzero(np.where(cluster_assignments == 30, 1, 0))
zero_indexes = np.nonzero(np.where(cluster_assignments == 0, 1, 0))



cluster_assignments[thrity_indexes] = 0
cluster_assignments[zero_indexes] = 30


print("Number of clusters", len(list(np.unique(cluster_assignments))))
print(list(np.unique(cluster_assignments)))
print("Max cluster", np.max(cluster_assignments))
plt.imshow(cluster_assignments)
plt.show()

template = smooth_contours(cluster_assignments)
plt.imshow(template)
plt.show()

np.save("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Final_Consensus_Clusters.npy", cluster_assignments)
