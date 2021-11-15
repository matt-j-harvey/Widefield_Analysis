import numpy as np
import matplotlib.pyplot as plt
import cv2
import networkx as nx
import community as community_louvain
from sklearn.cluster import SpectralClustering, MiniBatchKMeans, DBSCAN, AgglomerativeClustering

def visualise_clusters(clusters, downsampled_indicies, downsampled_height, downsampled_width):

    #load_clusters
    number_of_clusters = len(clusters)
    print("numberf clusters", number_of_clusters)


    # View
    image = np.zeros((downsampled_height * downsampled_width))
    for cluster_index in range(number_of_clusters):
        #colour_value = float(cluster_index) / number_of_clusters
        colour_value = np.random.randint(low=0, high=10)
        cluster = clusters[cluster_index]
        for pixel in cluster:
            pixel_index = downsampled_indicies[pixel]

            #image[pixel_index] = colour_value

            image[pixel_index] = cluster_index

    image = np.ndarray.reshape(image, (downsampled_height, downsampled_width))
    plt.imshow(image, cmap='turbo') #gist_ncar
    plt.show()






def downsample_mask(base_directory):

    # Load Mask
    mask = np.load(base_directory + "/mask.npy")

    # Downsample Mask
    original_height = np.shape(mask)[0]
    original_width = np.shape(mask)[1]
    downsampled_height = int(original_height/2)
    downsampled_width = int(original_width/2)
    downsampled_mask = cv2.resize(mask, dsize=(downsampled_width, downsampled_height))

    # Binairse Mask
    downsampled_mask = np.where(downsampled_mask > 0.1, 1, 0)
    downsampled_mask = downsampled_mask.astype(int)

    flat_mask = np.ndarray.flatten(downsampled_mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, downsampled_height, downsampled_width


def get_cluster_assignment_list(base_directory):

    # Load Mask
    indicies, downsampled_height, downsampled_width = downsample_mask(base_directory)
    number_of_downsampled_pixels = np.shape(indicies)[0]
    print("Number of Pixels", number_of_downsampled_pixels)

    # Create Cluster Assignment Matrix
    cluster_assignment_matrix = np.zeros((number_of_downsampled_pixels))

    # Load Clusters
    cluster_file = base_directory + "/Clusters.npy"
    clustering = np.load(cluster_file, allow_pickle=True)

    cluster_index = 0
    for cluster in clustering:
        for pixel in cluster:
            cluster_assignment_matrix[pixel] = cluster_index

        cluster_index += 1

    return cluster_assignment_matrix


def increment_consensus_matrix(consensus_matrix, cluster_assignments, number_of_pixels, weight_increment):

    for pixel_1 in range(number_of_pixels):
        percent = np.around(float(pixel_1)/number_of_pixels, 2) * 100
        print(int(percent), "%")
        for pixel_2 in range(pixel_1, number_of_pixels):
            if cluster_assignments[pixel_1] == cluster_assignments[pixel_2]:
                consensus_matrix[pixel_1, pixel_2] += weight_increment
                consensus_matrix[pixel_2, pixel_1] += weight_increment

    return consensus_matrix



def organise_clusters_louvain(partition):

    cluster_assignments = np.array(list(partition.values()))
    number_of_clusters = np.max(cluster_assignments)
    print("Number of clusters", number_of_clusters)

    clusters = []
    for cluster in range(number_of_clusters):
        pixels = np.where(cluster_assignments==cluster)
        clusters.append(pixels)

    return clusters

def organise_clusters_other(labels):
    number_of_clusters = np.max(labels)
    print("Number of clusters", number_of_clusters)

    clusters = []
    for cluster in range(number_of_clusters):
        pixels = np.where(labels==cluster)
        clusters.append(pixels)

    return clusters



def louvain_cluster_consensus_matrix(consensus_matrix):

    # Convert To NetworkX Graph
    print("Making Graph")
    affinity_network = nx.from_numpy_matrix(consensus_matrix)
    consensus_matrix = None

    # Perform Clustering
    print("Performing Clustering")
    partition = community_louvain.best_partition(affinity_network)

    # Get Clusters
    clusters = organise_clusters_louvain(partition)

    return clusters


def perform_consensus_clustering(session_list, consensus_matrix_file, consensus_clusters_file):

    # Get Details
    number_of_sessions = len(session_list)
    weight_increment = float(1) / number_of_sessions
    indicies, downsampled_height, downsampled_width = downsample_mask(session_list[0])
    number_of_downsampled_pixels = np.shape(indicies)[0]

    # Create Consensus Matrix
    consensus_matrix = np.zeros((number_of_downsampled_pixels, number_of_downsampled_pixels))

    # Get All Cluster Assignment Lists
    for file in session_list:
        print("Adding File: ", file)
        cluster_assigments = get_cluster_assignment_list(file)
        consensus_matrix = increment_consensus_matrix(consensus_matrix, cluster_assigments, number_of_downsampled_pixels, weight_increment)

    # Save Consensus Matrix
    np.save(consensus_matrix_file, consensus_matrix)

    # Load Consensus Matrix
    print("loading consensus matrix")
    consensus_matrix = np.load(consensus_matrix_file)
    np.fill_diagonal(consensus_matrix, 1)
    print("Consensus Matrix Min", np.min(consensus_matrix))
    print("Consensus Matrix Max", np.max(consensus_matrix))

    # Distance_Matrix
    print("Creating Distance Matrix")
    consensus_matrix = 1 - consensus_matrix

    print("Performing clustering")
    model = AgglomerativeClustering(affinity='precomputed', n_clusters=None, distance_threshold=0.8, linkage='average', compute_full_tree=True)
    labels = model.fit_predict(consensus_matrix)
    clusters = organise_clusters_other(labels)

    np.save(consensus_clusters_file, clusters)

    # Visualise Clusters
    clusters = np.load(consensus_clusters_file, allow_pickle=True)
    visualise_clusters(clusters, indicies, downsampled_height, downsampled_width)
















session_list = ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging/"]

consensus_matrix_file = "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/consensus_matrix.npy"
consensus_clusters_file = "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/consensus_clusters.npy"


