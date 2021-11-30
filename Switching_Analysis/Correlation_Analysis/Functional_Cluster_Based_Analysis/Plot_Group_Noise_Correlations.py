import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import networkx as nx
import cv2
from matplotlib.pyplot import cm
from matplotlib.colors import to_rgb
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist
import os
from skimage.transform import rescale, resize, downscale_local_mean

import sys

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Analysis/Switching_Analysis/Correlation_Analysis/Allen_Atlas_Based_Analysis")

import Draw_Brain_Network
import Allen_Atlas_Drawing_Functions

def sort_matrix(matrix):

    # Cluster Matrix
    Z = ward(pdist(matrix))

    # Get Dendogram Leaf Order
    new_order = leaves_list(Z)

    # Sorted Matrix
    sorted_matrix = matrix[:, new_order][new_order]

    return sorted_matrix


def jointly_sort_matricies(key_matrix, other_matricies):

    # Cluster Matrix
    Z = ward(pdist(key_matrix))

    # Get Dendogram Leaf Order
    new_order = leaves_list(Z)

    # Sorted Key Matrix
    sorted_key_matrix = key_matrix[:, new_order][new_order]

    # Sort Other Matricies
    sorted_matrix_list = []
    for matrix in other_matricies:
        sorted_matrix = matrix[:, new_order][new_order]
        sorted_matrix_list.append(sorted_matrix)

    return sorted_key_matrix, sorted_matrix_list


def get_average_correlation_modulation(session_list):

    visual_matrix_list = []
    odour_matrix_list = []
    difference_list = []

    for base_directory in session_list:
        visual_context_correlations = np.load(os.path.join(base_directory, "Noise_Correlations", "Visual_Context_Noise_Correlations.npy"))
        odour_context_correlations = np.load(os.path.join(base_directory, "Noise_Correlations", "Odour_Context_Noise_Correlations.npy"))
        difference_matrix = np.diff([visual_context_correlations, odour_context_correlations], axis=0)[0]

        visual_matrix_list.append(visual_context_correlations)
        odour_matrix_list.append(odour_context_correlations)
        difference_list.append(difference_matrix)

    visual_matrix_list = np.array(visual_matrix_list)
    odour_matrix_list = np.array(odour_matrix_list)

    group_visual_mean = np.mean(visual_matrix_list, axis=0)
    group_odour_mean = np.mean(odour_matrix_list, axis=0)
    difference_matrix = np.diff([group_visual_mean, group_odour_mean], axis=0)[0]
    #difference_matrix = np.mean(difference_list, axis=0)
    print(np.shape(group_odour_mean))
    #number_of_clusters = np.sqrt(np.shape(group_odour_mean)[0])
    #print("Number of clusters", number_of_clusters)

    #group_odour_mean = np.ndarray.reshape(group_odour_mean, (number_of_clusters, number_of_clusters))
    #group_visual_mean = np.ndarray.reshape(group_visual_mean, (number_of_clusters, number_of_clusters))
    #difference_matrix = np.ndarray.reshape(difference_matrix, (number_of_clusters, number_of_clusters))

    return group_visual_mean, group_odour_mean, difference_matrix, visual_matrix_list, odour_matrix_list


def draw_matricies(control_visual, control_odour, control_diff, mutant_visual, mutant_odour, mutant_diff):

    # Sort Matricies
    matrix_list = [control_visual,
                   control_odour,
                   control_diff,
                   mutant_visual,
                   mutant_odour,
                   mutant_diff]

    sorted_control_difference_matrix, sorted_matrix_list = jointly_sort_matricies(control_diff, matrix_list)

    figure_1 = plt.figure()
    control_visual_axis = figure_1.add_subplot(2, 3, 1)
    control_odour_axis = figure_1.add_subplot(2, 3, 2)
    control_difference_axis = figure_1.add_subplot(2, 3, 3)
    mutant_visual_axis = figure_1.add_subplot(2, 3, 4)
    mutant_odour_axis = figure_1.add_subplot(2, 3, 5)
    mutant_difference_axis = figure_1.add_subplot(2, 3, 6)


    control_visual_axis.imshow(sorted_matrix_list[0], cmap='bwr', vmin=-1, vmax=1)
    control_odour_axis.imshow(sorted_matrix_list[1], cmap='bwr', vmin=-1, vmax=1)
    control_difference_axis.imshow(sorted_matrix_list[2], cmap='bwr', vmin=-0.5, vmax=0.5)
    mutant_visual_axis.imshow(sorted_matrix_list[3], cmap='bwr', vmin=-1, vmax=1)
    mutant_odour_axis.imshow(sorted_matrix_list[4], cmap='bwr', vmin=-1, vmax=1)
    mutant_difference_axis.imshow(sorted_matrix_list[5], cmap='bwr', vmin=-0.5, vmax=0.5)
    plt.show()


    # Plot As Brain Regions
    cluster_centroids = np.load("/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/Cluster_Centroids.npy")
    base_directory =  "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/"
    masked_atlas, boundary_indicies = Allen_Atlas_Drawing_Functions.add_region_boundaries(base_directory)
    mask_outline = Allen_Atlas_Drawing_Functions.add_region_outline(base_directory)

    masked_atlas = np.add(mask_outline, masked_atlas)

    cluster_centroids = np.multiply(cluster_centroids, 2)
    plt.imshow(masked_atlas)
    plt.show()



    Draw_Brain_Network.draw_brain_network(cluster_centroids, masked_atlas, control_diff, None)
    Draw_Brain_Network.draw_brain_network(cluster_centroids, masked_atlas, mutant_diff, None)



def draw_matricies_threshold(visual_matrix_list, odour_matrix_list, difference_matrix):

    # Perform Signficance Testing
    t_stats, p_values = stats.ttest_rel(visual_matrix_list, odour_matrix_list, axis=0)

    # Threshold Difference Matrix
    thresholded_difference_matrix = np.where(p_values < 0.05, difference_matrix, 0)

    positive_difference_matrix = np.where(thresholded_difference_matrix > 0, thresholded_difference_matrix, 0)
    negative_difference_matrix = np.where(thresholded_difference_matrix < 0, thresholded_difference_matrix, 0)

    # Plot As Brain Regions
    cluster_centroids = np.load("/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/Cluster_Centroids.npy")
    base_directory = "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/"
    masked_atlas, boundary_indicies = Allen_Atlas_Drawing_Functions.add_region_boundaries(base_directory)
    mask_outline = Allen_Atlas_Drawing_Functions.add_region_outline(base_directory)

    masked_atlas = np.add(mask_outline, masked_atlas)
    masked_atlas = np.where(masked_atlas >0, 1, 0)
    cluster_centroids = np.multiply(cluster_centroids, 2)

    np.fill_diagonal(positive_difference_matrix, 0)
    np.fill_diagonal(negative_difference_matrix, 0)

    Draw_Brain_Network.draw_brain_network_single_colour(cluster_centroids, masked_atlas, positive_difference_matrix, "Reds", alpha=0.5)
    Draw_Brain_Network.draw_brain_network_single_colour(cluster_centroids, masked_atlas, negative_difference_matrix, "Blues", alpha=0.1)


controls = [
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging/",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging/",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN78.1A/2020_12_09_Switching_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN78.1D/2020_11_29_Switching_Imaging"]

mutants = [
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging/",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK12.1F/2021_09_22_Transition_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN71.2A/2020_12_17_Switching_Imaging"]


control_visual, control_odour, control_diff, control_visual_list, control_odour_list = get_average_correlation_modulation(controls)
mutant_visual, mutant_odour, mutant_diff, mutant_visual_list, mutant_odour_list = get_average_correlation_modulation(mutants)

draw_matricies(control_visual, control_odour, control_diff, mutant_visual, mutant_odour, mutant_diff)

draw_matricies_threshold(control_visual_list, control_odour_list, control_diff)
draw_matricies_threshold(mutant_visual_list, mutant_odour_list, mutant_diff)