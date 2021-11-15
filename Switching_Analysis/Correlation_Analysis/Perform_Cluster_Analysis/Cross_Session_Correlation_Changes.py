import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import networkx as nx
import cv2
from matplotlib.pyplot import cm
from matplotlib.colors import to_rgb
import os


def draw_brain_network(base_directory, adjacency_matrix):

    # Load Cluster Centroids
    cluster_centroids = np.load(base_directory + "/Cluster_Centroids.npy")

    # Create NetworkX Graph
    graph = nx.from_numpy_matrix(adjacency_matrix)

    # Get Edge Weights
    edges = graph.edges()
    weights = [graph[u][v]['weight'] for u, v in edges]
    weights = np.divide(weights, np.max(np.abs(weights)))

    # Get Edge Colours
    colourmap = cm.get_cmap('bwr')
    colours = []
    for weight in weights:
        colour = colourmap(weight)
        colours.append(colour)

    # Load Cluster Outlines
    cluster_outlines = np.load("/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/clean_clusters_outline.npy")
    plt.imshow(cluster_outlines, cmap='binary')

    image_height = np.shape(cluster_outlines)[0]

    # Draw Graph
    # Invert Cluster Centroids
    inverted_centroids = []
    for centroid in cluster_centroids:
        y_value = centroid[1]
        x_value = centroid[0]
        inverted_y = image_height - y_value
        inverted_centroids.append([x_value, inverted_y])

    #plt.title(session_name)
    nx.draw(graph, pos=inverted_centroids, node_size=1,  width=weights, edge_color=colours)
    #plt.savefig(base_directory + "/" + session_name + "_Signficant_Correlation_Changes.png")
    #plt.close()
    plt.show()


def exclude_clusters(matrix):

    excluded_cluster_list = [2, 119, 156, 160, 217]

    for cluster in excluded_cluster_list:
        matrix[cluster] = 0
        matrix[:, cluster] =0
    return matrix




clusters_file = "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/clean_clusters.npy"

session_list = ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging/"]



mean_visual_tensors = []
mean_odour_tensors = []

for base_directory in session_list:

    # Load Correlation Tensors
    visual_context_correlation_tensor = np.load(base_directory + "/Stimulus_Evoked_Visual_Context_Correlation_Tensor.npy")
    odour_context_correlation_tensor = np.load(base_directory + "/Stimulus-Evoked_Odour_Context_Correlation_Tensor.npy")

    # Remove Nans
    visual_context_correlation_tensor = np.nan_to_num(visual_context_correlation_tensor)
    odour_context_correlation_tensor = np.nan_to_num(odour_context_correlation_tensor)

    # Get Mean Correlations
    visual_context_correlation_tensor = np.mean(visual_context_correlation_tensor, axis=0)
    odour_context_correlation_tensor = np.mean(odour_context_correlation_tensor, axis=0)

    #plt.imshow(visual_context_correlation_tensor - odour_context_correlation_tensor)
    #plt.show()

    print(np.shape(visual_context_correlation_tensor))
    print(np.shape(odour_context_correlation_tensor))

    # Add To List
    mean_visual_tensors.append(visual_context_correlation_tensor)
    mean_odour_tensors.append(odour_context_correlation_tensor)

t_stats, p_values = stats.ttest_ind(a=mean_visual_tensors, b=mean_odour_tensors, axis=0)

t_stats = np.nan_to_num(t_stats)
p_values = np.nan_to_num(p_values)

print("P value", np.min(p_values), np.max(p_values))
print("T stats", np.min(t_stats), np.max(t_stats))

#t_stats = np.ndarray.reshape(t_stats, (number_of_clusters, number_of_clusters))
#p_values = np.ndarray.reshape(p_values, (number_of_clusters, number_of_clusters))


number_of_tests = np.shape(p_values)[0]
p_threshold = 0.01 / number_of_tests
print("Corrected P Threshold", p_threshold)

# Show Significant Changes
signficance_map = np.where(p_values < p_threshold, t_stats, 0)

# Exclude Midline
signficance_map = exclude_clusters(signficance_map)

plt.imshow(signficance_map, cmap='bwr')
plt.show()

draw_brain_network(session_list[0], signficance_map)
