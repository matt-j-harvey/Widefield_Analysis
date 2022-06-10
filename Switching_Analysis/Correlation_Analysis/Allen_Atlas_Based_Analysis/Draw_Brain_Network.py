import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm



def draw_brain_network(cluster_centroids, cluster_outlines, adjacency_matrix, session_name):

    # Create NetworkX Graph
    graph = nx.from_numpy_matrix(adjacency_matrix)

    # Get Edge Weights
    edges = graph.edges()
    weights = [graph[u][v]['weight'] for u, v in edges]
    weights = np.divide(weights, np.max(weights))

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




def draw_brain_network_no_overlay(cluster_centroids, adjacency_matrix):

    # Create NetworkX Graph
    graph = nx.from_numpy_matrix(adjacency_matrix)

    # Get Edge Weights
    edges = graph.edges()
    weights = [graph[u][v]['weight'] for u, v in edges]
    weights = np.divide(weights, np.max(weights))

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
        y_value = centroid[0]
        x_value = centroid[1]
        inverted_y = image_height - y_value
        inverted_centroids.append([x_value, y_value])


    nx.draw(graph, pos=inverted_centroids, node_size=1,  width=weights, edge_color=colours)
    plt.show()
    #plt.savefig(base_directory + "/" + session_name + "_Signficant_Correlation_Changes.png")
    #plt.close()



def draw_brain_network_single_colour(cluster_centroids, cluster_outlines, adjacency_matrix, cmap, alpha=0.5):

    # Create NetworkX Graph
    graph = nx.from_numpy_matrix(adjacency_matrix)

    # Get Edge Weights
    edges = graph.edges()
    weights = [graph[u][v]['weight'] for u, v in edges]
    weights = np.divide(weights, np.max(weights))

    # Get Edge Colours
    colourmap = cm.get_cmap(cmap)

    colours = []
    for weight in weights:
        colour = colourmap(weight)
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

    plt.imshow(cluster_outlines, cmap='Purples')
    #width = weights
    nx.draw(graph, pos=inverted_centroids, node_size=0, edge_color=colours, alpha=alpha)
    plt.show()
    #plt.savefig(base_directory + "/" + session_name + "_Signficant_Correlation_Changes.png")
    #plt.close()


