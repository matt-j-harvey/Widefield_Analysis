import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.transform import resize
from skimage.feature import canny
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, AffinityPropagation, KMeans
import networkx as nx
import community as community_louvain
from scipy import stats
from skimage.morphology import opening, closing


def sort_matrix(matrix):

    # Cluster Matrix
    Z = ward(pdist(matrix))

    # Get Dendogram Leaf Order
    new_order = leaves_list(Z)

    # Sorted Matrix
    sorted_matrix = matrix[:, new_order][new_order]

    return sorted_matrix


def transform_image(image, alignment_dictionary):
    # Rotate
    angle = alignment_dictionary['rotation']
    x_shift = alignment_dictionary['x_shift']
    y_shift = alignment_dictionary['y_shift']

    transformed_image = np.copy(image)
    transformed_image = ndimage.rotate(transformed_image, angle, reshape=False, prefilter=True)
    transformed_image = np.roll(a=transformed_image, axis=0, shift=y_shift)
    transformed_image = np.roll(a=transformed_image, axis=1, shift=x_shift)

    return transformed_image


def align_session(cluster_assignments, alignment_dictionary, downsample_size):

    # Convert To Int
    cluster_assignments = np.ndarray.astype(cluster_assignments, int)

    # Get Unique Clusters
    unique_cluster_list = list(set(np.unique(cluster_assignments)))

    # Create Aligned Cluster
    aligned_clusters = np.zeros(downsample_size)

    for cluster_index in unique_cluster_list:

        # Get Cluster Mask
        cluster_mask = np.where(cluster_assignments == cluster_index, 1, 0)
        cluster_mask = np.ndarray.astype(cluster_mask, float)

        # Transform
        cluster_mask = transform_image(cluster_mask, alignment_dictionary)
        #plt.imshow(cluster_mask)
        #plt.show()

        # Resize
        cluster_mask = resize(cluster_mask, downsample_size, preserve_range=True)
        #plt.imshow(cluster_mask)
        #plt.show()

        # Rebinarise
        aligned_clusters = np.where(cluster_mask > 0.5, cluster_index, aligned_clusters)


    return aligned_clusters


def get_cluster_centroids(aligned_clusters_list):

    number_of_sessions, number_of_pixels = np.shape(aligned_clusters_list)

    assignment_variance = np.zeros(number_of_pixels)


    for pixel_1 in range(number_of_pixels):
        print("Pixel ", pixel_1, " of ", number_of_pixels)
        pixel_1_vector = aligned_clusters_list[:, pixel_1]
        pixel_variance = 0

        for pixel_2 in range(pixel_1, number_of_pixels):
            pixel_variance += np.var(np.where(pixel_1_vector == aligned_clusters_list[:, pixel_2], 1, 0))
        assignment_variance[pixel_1] = pixel_variance

    assignment_variance = np.reshape(assignment_variance, (75, 76))
    plt.imshow(assignment_variance)
    plt.show()


def create_connectivity_matrix(aligned_clusters_list):

    print("Aligned Clusters List", np.shape(aligned_clusters_list))

    number_of_sessions, number_of_pixels = np.shape(aligned_clusters_list)
    print("Number Of Sessions", number_of_sessions)
    print("Number of Pixels", number_of_pixels)

    connectivity_increment = float(1) / number_of_sessions
    print("Connectivity Increment", connectivity_increment)

    """
    for session in aligned_clusters_list:
        session = np.ndarray.reshape(session, (75, 76))
        plt.imshow(session)
        plt.show()
    """
    upper_connectivity_matrix = np.zeros((number_of_pixels, number_of_pixels))

    for pixel_1 in range(number_of_pixels):
        print("Pixel ", pixel_1, " of ", number_of_pixels)
        pixel_1_vector = aligned_clusters_list[:, pixel_1]

        for pixel_2 in range(pixel_1, number_of_pixels):
            upper_connectivity_matrix[pixel_1, pixel_2] = np.sum(np.where(pixel_1_vector == aligned_clusters_list[:, pixel_2], connectivity_increment, 0))
            """
            equality_vector = np.where(pixel_1_vector == aligned_clusters_list[:, pixel_2], 1, 0)
            connectivity_increment = np.sum(equality_vector)
            upper_connectivity_matrix[pixel_1, pixel_2] = connectivity_increment
            """
            #connectivity_matrix[pixel_2, pixel_1] = connection_strength

            #print("Pixel 1 Vector", pixel_1_vector)
            #print("Pixel 2 vector", pixel_2_vector)
            #print("Equality Vector", equality_vector)

            """
            for session_index in range(number_of_sessions):
                if aligned_clusters_list[session_index][pixel_1] == aligned_clusters_list[session_index][pixel_2]:
                    connectivity_matrix[pixel_1, pixel_2] += connectivity_increment
                    connectivity_matrix[pixel_2, pixel_1] += connectivity_increment
    """

    lower_connectivity_matrix = np.transpose(upper_connectivity_matrix)
    connectivity_matrix = np.add(upper_connectivity_matrix, lower_connectivity_matrix)
    return connectivity_matrix


def align_clusters(session_list, downsample_size):

    aligned_cluster_assignments = []

    session_count = 0
    for session in session_list:
        print("Aliging Session: ", session_count, " of ", len(session_list))

        # Load Cluster Assignments
        cluster_assignments = np.load(os.path.join(session, "Pixel_Assignments.npy"))
        cluster_assignments = np.ndarray.astype(cluster_assignments, int)

        # Load Aligment Dictionary
        alignment_dictionary = np.load(os.path.join(session, "Cluster_Alignment_Dictionary.npy"), allow_pickle=True)[()]

        # Align Clusters
        cluster_assignments = align_session(cluster_assignments, alignment_dictionary, downsample_size)

        # Flatten Clusters and Append To List
        cluster_assignments = np.ndarray.flatten(cluster_assignments)
        aligned_cluster_assignments.append(cluster_assignments)

        session_count += 1

    aligned_cluster_assignments = np.array(aligned_cluster_assignments)
    return aligned_cluster_assignments


def view_louvain(partition, downsample_size):

    print("partition: ", partition)

    pixel_assigments = []
    for pixel in partition.keys():
        pixel_assigments.append(partition[pixel])

    pixel_assigments = np.array(pixel_assigments)
    pixel_assigments = np.reshape(pixel_assigments, downsample_size)
    pixel_assigments = resize(pixel_assigments, (600, 608), preserve_range=True)
    pixel_assigments = np.ndarray.astype(pixel_assigments, 'float32')
    edges = canny(pixel_assigments, sigma=1)
    plt.imshow(edges)
    plt.show()
    plt.imshow(pixel_assigments, cmap='flag')
    plt.show()

    return pixel_assigments



session_list = [
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_01_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_03_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_05_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_07_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_09_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_11_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_13_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_15_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_17_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_19_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_22_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_24_Discrimination_Imaging",

    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging",

    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_04_29_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_01_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_03_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_05_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_07_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_09_Discrimination_Imaging",

    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_09_25_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_09_29_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A//2021_10_01_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_10_03_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_10_05_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_10_07_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging",

    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_15_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_16_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_17_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_19_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_23_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_25_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_14_Discrimination_Imaging",

    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_16_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_17_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_19_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_21_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_24_Discrimination_Imaging",
]

# Get Consensus Clusters
downsample_size = (150, 152)

# Align Clusters
aligned_cluster_assignments = align_clusters(session_list, downsample_size)

# Create Connectiity Matrix
connectivity_matrix = create_connectivity_matrix(aligned_cluster_assignments)
np.save("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Connectivity_Matrix.npy", connectivity_matrix)

# Threshold Connetivity Matrix
connectivity_matrix = np.load("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Connectivity_Matrix.npy")
connectivity_matrix = np.where(connectivity_matrix > 0.90, connectivity_matrix, 0)

# Create and Partition Graph
network_graph = nx.from_numpy_array(connectivity_matrix)
parition = community_louvain.best_partition(network_graph)
np.save("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Graph Partition.npy", parition)

# Save pixel Assignments
partition = np.load("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Graph Partition.npy", allow_pickle=True)[()]
pixel_assignments = view_louvain(partition, downsample_size)
np.save("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Pixel_Assignments.npy", pixel_assignments)
