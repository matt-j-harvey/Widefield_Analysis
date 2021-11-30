import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import networkx as nx
import cv2
from matplotlib.pyplot import cm
from matplotlib.colors import to_rgb
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist
import os
import tables
import sys
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions
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




def concatenate_and_subtract_mean(tensor):

    # Get Tensor Structure
    print("Tensor shape", np.shape(tensor))
    number_of_trials = np.shape(tensor)[0]
    number_of_timepoints = np.shape(tensor)[1]
    number_of_clusters = np.shape(tensor)[2]

    # Get Mean Trace
    mean_trace = np.mean(tensor, axis=0)

    # Subtract Mean Trace
    subtracted_tensor = np.subtract(tensor, mean_trace)

    # Concatenate Trials
    concatenated_subtracted_tensor = np.reshape(subtracted_tensor, (number_of_trials * number_of_timepoints, number_of_clusters))
    concatenated_subtracted_tensor = np.transpose(concatenated_subtracted_tensor)

    return concatenated_subtracted_tensor


def convert_block_boundaries_to_trial_type(visual_onsets, odour_onsets, visual_blocks, odour_blocks):

    # Get Combined Onsets
    combined_onsets = np.concatenate([visual_onsets, odour_onsets])
    combined_onsets = list(combined_onsets)
    combined_onsets.sort()

    translated_odour_blocks = []
    translated_visual_blocks = []

    visual_onsets = list(visual_onsets)
    odour_onsets = list(odour_onsets)

    for visual_block in visual_blocks:
        start_combined_index = visual_block[0]
        stop_combined_index = visual_block[1]

        start_onset = combined_onsets[start_combined_index]
        stop_onset = combined_onsets[stop_combined_index]

        start_visual_index = visual_onsets.index(start_onset)
        stop_visual_index = visual_onsets.index(stop_onset)

        translated_visual_blocks.append([start_visual_index, stop_visual_index])


    for odour_block in odour_blocks:
        start_combined_index = odour_block[0]
        stop_combined_index  = odour_block[1]

        start_onset = combined_onsets[start_combined_index]
        stop_onset = combined_onsets[stop_combined_index]

        start_odour_index = odour_onsets.index(start_onset)
        stop_odour_index = odour_onsets.index(stop_onset)

        translated_odour_blocks.append([start_odour_index, stop_odour_index])


    return translated_visual_blocks, translated_odour_blocks



def create_correlation_tensors(base_directory, visual_tensor_file, odour_tensor_file):

    # Load Tensors
    visual_context_tensor = np.load(os.path.join(base_directory, visual_tensor_file))
    odour_context_tensor = np.load(os.path.join(base_directory, odour_tensor_file))
    print("Visual Context Tensor Shape", np.shape(visual_context_tensor))
    print("Odour Context Tensor Shape", np.shape(odour_context_tensor))

    # Concatenate and Subtract Means
    visual_context_tensor = concatenate_and_subtract_mean(visual_context_tensor)
    odour_context_tensor =  concatenate_and_subtract_mean(odour_context_tensor)

    # Get Correlation Matrix
    visual_correlation_matrix = np.corrcoef(visual_context_tensor)
    odour_correlation_matrix = np.corrcoef(odour_context_tensor)

    return visual_correlation_matrix, odour_correlation_matrix


def draw_modulated_correlations(base_directory, visual_matrix_list, odour_matrix_list):

    # Get Mean Matricies
    mean_visual_matrix = np.mean(visual_matrix_list, axis=0)
    mean_odour_matrix = np.mean(odour_matrix_list, axis=0)

    # Get Difference
    difference_matrix = np.diff([mean_visual_matrix, mean_odour_matrix], axis=0)[0]

    # Perform Signficance Testing
    t_stats, p_values = stats.ttest_rel(visual_matrix_list, odour_matrix_list, axis=0)

    # Threshold Difference Matrix
    #thresholded_difference_matrix = np.where(p_values < 0.05, difference_matrix, 0)
    #thresholded_difference_matrix = difference_matrix

    positive_difference_matrix = np.where(difference_matrix > 0, difference_matrix, 0)
    negative_difference_matrix = np.where(difference_matrix < 0, np.abs(difference_matrix), 0)

    masked_atlas, boundary_indicies = Allen_Atlas_Drawing_Functions.add_region_boundaries(base_directory)
    mask_outline = Allen_Atlas_Drawing_Functions.add_region_outline(base_directory)

    masked_atlas = np.add(mask_outline, masked_atlas)

    cluster_centroids = np.load("/home/matthew/Documents/Allen_Atlas_Templates/Allen_Region_Centroids.npy")
    Draw_Brain_Network.draw_brain_network_single_colour(cluster_centroids, masked_atlas, positive_difference_matrix, "Reds")
    Draw_Brain_Network.draw_brain_network_single_colour(cluster_centroids, masked_atlas, negative_difference_matrix, "Blues")


controls = ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN78.1A/2020_12_09_Switching_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN78.1D/2020_11_29_Switching_Imaging/"]

mutants = [ "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN71.2A/2020_12_17_Switching_Imaging/"]
            #"/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK12.1F/2021_09_22_Transition_Imaging/"



visual_tensor_file = "Allen_Activity_Tensor_Vis_1_Visual.npy"
odour_tensor_file = "Allen_Activity_Tensor_Vis_1_Odour.npy"
trial_start = -10
trial_stop = 40

# Get Noise Correlation Maps Controls
control_visual_correlation_matrix_list = []
control_odour_correlation_matrix_list = []

for base_directory in controls:
    visual_correlation_matrix, odour_correlation_matrix = create_correlation_tensors(base_directory, visual_tensor_file, odour_tensor_file)
    control_visual_correlation_matrix_list.append(visual_correlation_matrix)
    control_odour_correlation_matrix_list.append(odour_correlation_matrix)


# Get Noise Correlation Maps Mutants
mutant_visual_correlation_matrix_list = []
mutant_odour_correlation_matrix_list = []

for base_directory in mutants:
    visual_correlation_matrix, odour_correlation_matrix = create_correlation_tensors(base_directory, visual_tensor_file, odour_tensor_file)
    mutant_visual_correlation_matrix_list.append(visual_correlation_matrix)
    mutant_odour_correlation_matrix_list.append(odour_correlation_matrix)


# View Mean Correlation Maps
control_visual_correlation_matrix_list = np.array(control_visual_correlation_matrix_list)
control_odour_correlation_matrix_list  = np.array(control_odour_correlation_matrix_list)
mutant_visual_correlation_matrix_list  = np.array(mutant_visual_correlation_matrix_list)
mutant_odour_correlation_matrix_list   = np.array(mutant_odour_correlation_matrix_list)

control_mean_visual_correlation_matrix = np.mean(control_visual_correlation_matrix_list, axis=0)
control_mean_odour_correlation_matrix = np.mean(control_odour_correlation_matrix_list, axis=0)
mutant_mean_visual_correlation_matrix = np.mean(mutant_visual_correlation_matrix_list, axis=0)
mutant_mean_odour_correlation_matrix = np.mean(mutant_odour_correlation_matrix_list, axis=0)

control_difference_matrix = np.diff([control_mean_visual_correlation_matrix, control_mean_odour_correlation_matrix], axis=0)[0]
mutant_difference_matrix = np.diff([mutant_mean_visual_correlation_matrix, mutant_mean_odour_correlation_matrix], axis=0)[0]

#difference_matrix, [mean_visual_matrix, mean_odour_matrix] = jointly_sort_matricies(difference_matrix, [mean_visual_matrix, mean_odour_matrix])


matrix_list = [ control_mean_visual_correlation_matrix,
                control_mean_odour_correlation_matrix,
                control_difference_matrix,
                mutant_mean_visual_correlation_matrix,
                mutant_mean_odour_correlation_matrix,
                mutant_difference_matrix]

sorted_control_difference_matrix, sorted_matrix_list = jointly_sort_matricies(control_difference_matrix, matrix_list)

figure_1 = plt.figure()
control_visual_axis     = figure_1.add_subplot(2, 3, 1)
control_odour_axis      = figure_1.add_subplot(2, 3, 2)
control_difference_axis = figure_1.add_subplot(2, 3, 3)
mutant_visual_axis      = figure_1.add_subplot(2, 3, 4)
mutant_odour_axis       = figure_1.add_subplot(2, 3, 5)
mutant_difference_axis  = figure_1.add_subplot(2, 3, 6)

control_visual_axis.imshow(     sorted_matrix_list[0], cmap='bwr', vmin=-1, vmax=1)
control_odour_axis.imshow(      sorted_matrix_list[1], cmap='bwr', vmin=-1, vmax=1)
control_difference_axis.imshow( sorted_matrix_list[2], cmap='bwr', vmin=-0.5, vmax=0.5)
mutant_visual_axis.imshow(      sorted_matrix_list[3], cmap='bwr', vmin=-1, vmax=1)
mutant_odour_axis.imshow(       sorted_matrix_list[4], cmap='bwr', vmin=-1, vmax=1)
mutant_difference_axis.imshow(  sorted_matrix_list[5], cmap='bwr', vmin=-0.5, vmax=0.5)

plt.show()

draw_modulated_correlations(controls[0], control_visual_correlation_matrix_list, control_odour_correlation_matrix_list)

draw_modulated_correlations(mutants[0], mutant_visual_correlation_matrix_list, mutant_odour_correlation_matrix_list)


# Signficance Test
t_stats, p_values = stats.ttest_rel(control_visual_correlation_matrix_list, control_odour_correlation_matrix_list, axis=0)
print("p value shape", np.shape(p_values))


thresholded_t_stats = np.where(p_values < 0.05, np.abs(t_stats), 0)
thresholded_t_stats = sort_matrix(thresholded_t_stats)

p_values = np.ndarray.flatten(p_values)
p_values = list(p_values)
p_values.sort()
print("P values", p_values)

plt.imshow(thresholded_t_stats)
plt.show()



# Signficance Test
t_stats, p_values = stats.ttest_rel(mutant_visual_correlation_matrix_list, mutant_odour_correlation_matrix_list, axis=0)
print("p value shape", np.shape(p_values))


thresholded_t_stats = np.where(p_values < 0.05, np.abs(t_stats), 0)
thresholded_t_stats = sort_matrix(thresholded_t_stats)

p_values = np.ndarray.flatten(p_values)
p_values = list(p_values)
p_values.sort()
print("P values", p_values)

plt.imshow(thresholded_t_stats)
plt.show()