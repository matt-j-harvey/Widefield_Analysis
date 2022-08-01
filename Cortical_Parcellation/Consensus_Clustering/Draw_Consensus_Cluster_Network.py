import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm





def draw_brain_network_axis(cluster_centroids, cluster_outlines, adjacency_matrix, axis, session_name):

    # Create NetworkX Graph
    graph = nx.from_numpy_matrix(adjacency_matrix)

    # Get Edge Weights
    edges = graph.edges()
    weights = [graph[u][v]['weight'] for u, v in edges]
    #weights = np.divide(weights, np.max(weights))

    # Get Edge Colours
    positive_colourmap = cm.get_cmap('Reds')
    negative_colourmap = cm.get_cmap('Blues')

    colours = []
    for weight in weights:

        if weight >= 0:
            colour = positive_colourmap(weight)
        else:
            colour = negative_colourmap(np.abs(weight))

        colours.append(colour)


    image_height = np.shape(cluster_outlines)[0]

    # Draw Graph
    # Invert Cluster Centroids
    inverted_centroids = []
    for centroid in cluster_centroids:
        y_value = centroid[1]
        x_value = centroid[0]
        inverted_y = image_height - y_value
        inverted_centroids.append([x_value, inverted_y])

    axis.set_title(session_name)
    axis.imshow(cluster_outlines, cmap='binary')
    nx.draw(graph, pos=cluster_centroids, node_size=1, edge_color=colours, ax=axis) #width=weights, edge_color=colours

    #plt.savefig(base_directory + "/" + session_name + "_Signficant_Correlation_Changes.png")
    #plt.close()



def draw_brain_network(cluster_centroids, cluster_outlines, adjacency_matrix, session_name):

    # Create NetworkX Graph
    graph = nx.from_numpy_matrix(adjacency_matrix)

    # Get Edge Weights
    edges = graph.edges()
    weights = [graph[u][v]['weight'] for u, v in edges]
    #weights = np.divide(weights, np.max(weights))

    # Get Edge Colours
    positive_colourmap = cm.get_cmap('Reds')
    negative_colourmap = cm.get_cmap('Blues')

    colours = []
    for weight in weights:

        if weight >= 0:
            colour = positive_colourmap(weight)
        else:
            colour = negative_colourmap(np.abs(weight))

        colours.append(colour)


    image_height = np.shape(cluster_outlines)[0]

    # Draw Graph
    # Invert Cluster Centroids
    inverted_centroids = []
    for centroid in cluster_centroids:
        y_value = centroid[1]
        x_value = centroid[0]
        inverted_y = image_height - y_value
        inverted_centroids.append([x_value, inverted_y])

    plt.title(session_name)
    plt.imshow(cluster_outlines, cmap='binary')
    nx.draw(graph, pos=cluster_centroids, node_size=1, edge_color=colours) #width=weights, edge_color=colours
    plt.show()
    #plt.savefig(base_directory + "/" + session_name + "_Signficant_Correlation_Changes.png")
    #plt.close()


def threshold_matrix(matrix, percentile=50):
    matrix = np.nan_to_num(matrix)
    abs_matrix = np.abs(matrix)

    weight_threshold = np.percentile(abs_matrix, q=percentile)
    print("Weight threshold", weight_threshold)
    thresholded_matrix = np.where(abs_matrix >= weight_threshold, matrix, 0)
    return thresholded_matrix

def remove_noise_areas(connectivity_matirx):

    noise_areas = [26, 28, 27, 34, 37]

    for region in noise_areas:
        connectivity_matirx[region-1:] = 0
        connectivity_matirx[:, region-1] = 0

    return connectivity_matirx

# Load Adjacency Matricies
"""
pre_learning_vis_1_noise_correlations = np.load("/media/matthew/Expansion/Widefield_Analysis/Discrimination_Analysis/Noise_Correlation_Changes/Post_Learning_Changes/Pre_Learning_Vis_1.npy")
intermediate_vis_1_noise_correlations = np.load("/media/matthew/Expansion/Widefield_Analysis/Discrimination_Analysis/Noise_Correlation_Changes/Post_Learning_Changes/Intermediate_Learning_Vis_1.npy")
post_learning_vis_1_noise_correlations = np.load("/media/matthew/Expansion/Widefield_Analysis/Discrimination_Analysis/Noise_Correlation_Changes/Post_Learning_Changes/Post_Learning_Vis_1.npy")
"""


pre_learning_vis_1_noise_correlations = np.load("/media/matthew/Expansion/Widefield_Analysis/Discrimination_Analysis/Noise_Correlation_Changes/Genotype_Changes/Post_Learning_Sig.npy")
intermediate_vis_1_noise_correlations = np.load("/media/matthew/Expansion/Widefield_Analysis/Discrimination_Analysis/Noise_Correlation_Changes/Genotype_Changes/Post_Learning_Sig.npy")
post_learning_vis_1_noise_correlations = np.load("/media/matthew/Expansion/Widefield_Analysis/Discrimination_Analysis/Noise_Correlation_Changes/Genotype_Changes/Post_Learning_Sig.npy")


pre_learning_vis_1_noise_correlations = remove_noise_areas(pre_learning_vis_1_noise_correlations)
intermediate_vis_1_noise_correlations = remove_noise_areas(intermediate_vis_1_noise_correlations)
post_learning_vis_1_noise_correlations = remove_noise_areas(post_learning_vis_1_noise_correlations)

# Zero Diags
np.fill_diagonal(pre_learning_vis_1_noise_correlations, 0)
np.fill_diagonal(intermediate_vis_1_noise_correlations, 0)
np.fill_diagonal(post_learning_vis_1_noise_correlations, 0)

# Load Pixel Assignments
pixel_assignments = np.load(r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Final_Consensus_Clusters.npy")
plt.imshow(pixel_assignments)
plt.show()

# Load Cluster Centroids
cluster_centroids = np.load("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/cluster_centroids.npy")

# Load Cluster Outlines
cluster_outlines = np.load(r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/cluster_outlines.npy")

plt.imshow(post_learning_vis_1_noise_correlations)
plt.show()

learning_changes = np.subtract(post_learning_vis_1_noise_correlations, pre_learning_vis_1_noise_correlations)


percentile = 90
pre_learning_vis_1_noise_correlations = threshold_matrix(pre_learning_vis_1_noise_correlations, percentile)
intermediate_vis_1_noise_correlations = threshold_matrix(intermediate_vis_1_noise_correlations, percentile)
post_learning_vis_1_noise_correlations = threshold_matrix(post_learning_vis_1_noise_correlations, percentile)

figure_1 = plt.figure()
rows = 1
columns = 3
pre_learning_axis = figure_1.add_subplot(rows, columns, 1)
intermediate_learning_axis = figure_1.add_subplot(rows, columns, 2)
post_learning_axis = figure_1.add_subplot(rows, columns, 3)

draw_brain_network_axis(cluster_centroids, cluster_outlines, pre_learning_vis_1_noise_correlations, pre_learning_axis, session_name="Pre Learning")
draw_brain_network_axis(cluster_centroids, cluster_outlines, intermediate_vis_1_noise_correlations, intermediate_learning_axis, session_name="Intermeidate Learning")
draw_brain_network_axis(cluster_centroids, cluster_outlines, post_learning_vis_1_noise_correlations, post_learning_axis, session_name="Post Learning")
plt.show()