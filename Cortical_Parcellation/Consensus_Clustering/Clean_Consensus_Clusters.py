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


def remove_all_ocurrences_of_value_from_list(myList, valueToBeRemoved):

    try:
        while True:
            myList.remove(valueToBeRemoved)
    except ValueError:
        pass

    return myList


def reassign_pixel(pixel_x, pixel_y, image_height, image_width, pixel_assignments):

    neighbourhood = []

    new_assignment = -1
    # Check We're Not On The Edge
    if pixel_x + 1 < image_width:
        if pixel_x - 1 >= 0:
            if pixel_y - 1 >=0:
                if pixel_y + 1 < image_height:

                    neighbourhood.append(pixel_assignments[pixel_y, pixel_x + 1])
                    neighbourhood.append(pixel_assignments[pixel_y, pixel_x - 1])

                    neighbourhood.append(pixel_assignments[pixel_y + 1, pixel_x])
                    neighbourhood.append(pixel_assignments[pixel_y - 1, pixel_x])

                    neighbourhood.append(pixel_assignments[pixel_y - 1, pixel_x - 1])
                    neighbourhood.append(pixel_assignments[pixel_y + 1, pixel_x + 1])

                    neighbourhood.append(pixel_assignments[pixel_y - 1, pixel_x + 1])
                    neighbourhood.append(pixel_assignments[pixel_y + 1, pixel_x - 1])

                    remove_all_ocurrences_of_value_from_list(neighbourhood, -1)
                    if len(neighbourhood) > 0:
                        new_assignment = stats.mode(neighbourhood)[0]


    return new_assignment


def perform_morphological_opening(pixel_assignments):

    clusters = list(np.unique(pixel_assignments))

    complete = False
    count = 0
    while complete == False:
        print(count)
        new_pixel_assignments = np.ones(np.shape(pixel_assignments))
        new_pixel_assignments = np.multiply(new_pixel_assignments, -1)

        for cluster in clusters:
            cluster_mask = np.where(pixel_assignments == cluster, 1, 0)
            cluster_mask = opening(cluster_mask)
            cluster_indexes = np.nonzero(cluster_mask)
            new_pixel_assignments[cluster_indexes] = cluster

        if np.array_equal(new_pixel_assignments, pixel_assignments):
            complete = True
            return new_pixel_assignments
        else:
            pixel_assignments = new_pixel_assignments
            count += 1


def size_threshold_consensus_clusters(pixel_assignments):

    # Remove Clusters Smaller Than A Certain Size Threshold
    size_threshold = 50
    unique_clusters = list(np.unique(pixel_assignments))
    for cluster in unique_clusters:
        cluster_mask = np.where(pixel_assignments == cluster, 1, 0)
        if np.sum(cluster_mask) < size_threshold:
            pixel_assignments = np.where(pixel_assignments==cluster, -1, pixel_assignments)

    return pixel_assignments


def regrow_clusters(pixel_assignments):

    complete = False
    current_number_of_unassigned_pixels = 0
    while complete != True:
        unassigned_mask = np.where(pixel_assignments == -1, 1, 0)
        unassigned_pixels = np.nonzero(unassigned_mask)

        new_number_of_unassigned_pixels = np.shape(unassigned_pixels)[1]

        if new_number_of_unassigned_pixels == current_number_of_unassigned_pixels:
            return pixel_assignments
        else:
            current_number_of_unassigned_pixels = new_number_of_unassigned_pixels

            print(np.shape(unassigned_pixels))
            unassigned_pixels = np.vstack(unassigned_pixels)
            unassigned_pixels = np.transpose(unassigned_pixels)

            new_pixel_assignments = np.copy(pixel_assignments)

            for pixel in unassigned_pixels:
                new_pixel = reassign_pixel(pixel[1], pixel[0], 600, 608, pixel_assignments)
                new_pixel_assignments[pixel[0], pixel[1]] = new_pixel


        pixel_assignments = new_pixel_assignments


def clean_consensus_clusters():

    # View Raw Assignments
    pixel_assignments = np.load("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Pixel_Assignments.npy")
    plt.title("Raw Clusters")
    plt.imshow(pixel_assignments)
    plt.show()
    
    # Size Threshold
    pixel_assignments = size_threshold_consensus_clusters(pixel_assignments)
    np.save("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/thresholded_pixel_assignments.npy", pixel_assignments)
    plt.title("Size Thresholded")
    plt.imshow(pixel_assignments)
    plt.show()

    # Morohological Opening
    pixel_assignments = np.load("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/thresholded_pixel_assignments.npy")
    pixel_assignments = perform_morphological_opening(pixel_assignments)
    np.save("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/thresholded_pixel_assignments.npy", pixel_assignments)
    plt.title("Morphological Opening")
    plt.imshow(pixel_assignments)
    plt.show()

    # Regrow Clusters
    pixel_assignments = np.load("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/thresholded_pixel_assignments.npy")
    pixel_assignments = regrow_clusters(pixel_assignments)
    np.save("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/regrown_pixel_assignments.npy", pixel_assignments)
    plt.title("Regrown Clusters")
    plt.imshow(pixel_assignments)
    plt.show()


clean_consensus_clusters()