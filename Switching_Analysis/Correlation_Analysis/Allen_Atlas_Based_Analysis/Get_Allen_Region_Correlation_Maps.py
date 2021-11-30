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




def get_block_boundaries(visual_context_onsets, odour_context_onsets):

    # Get Combined Onsets
    combined_onsets = np.concatenate([visual_context_onsets, odour_context_onsets])
    combined_onsets = list(combined_onsets)
    combined_onsets.sort()

    visual_blocks = []
    odour_blocks = []

    current_block_start = 0
    current_block_end = None

    # Get Initial Onset
    if combined_onsets[0] in visual_context_onsets:
        current_block_type = 0
    elif combined_onsets[0] in odour_context_onsets:
        current_block_type = 1
    else:
        print("Error! onsets not in either vidual or oflactory onsets")

    # Iterate Through All Subsequent Onsets
    number_of_onsets = len(combined_onsets)
    for onset_index in range(1, number_of_onsets):

        # Get Onset
        onset = combined_onsets[onset_index]

        # If we are currently in an Visual Block
        if current_block_type == 0:

            # If The Next Onset is An Odour Block - Block Finish, add Block To Boundaries
            if onset in odour_context_onsets:
                current_block_end = onset_index-1
                visual_blocks.append([current_block_start, current_block_end])
                current_block_type = 1
                current_block_start = onset_index

        # If we Are currently in an Odour BLock
        if current_block_type == 1:

            # If The NExt Onset Is a Visual Trial - BLock Finish Add Block To Block Boundaires
            if onset in visual_context_onsets:
                current_block_end = onset_index - 1
                odour_blocks.append([current_block_start, current_block_end])
                current_block_type = 0
                current_block_start = onset_index

    return visual_blocks, odour_blocks


def concatenate_and_subtract_mean(tensor):

    # Get Tensor Structure
    number_of_trials = np.shape(tensor)[0]
    number_of_timepoints = np.shape(tensor)[1]
    number_of_clusters = np.shape(tensor)[2]

    # Get Mean Trace
    mean_trace = np.mean(tensor, axis=0)

    # Subtract Mean Trace
    subtracted_tensor = np.subtract(tensor, mean_trace)
    #subtracted_tensor = tensor

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



def create_correlation_tensors(base_directory, visual_onsets_file, odour_onsets_file):


    # Load Onsets
    visual_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", visual_onsets_file))
    odour_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", odour_onsets_file))

    visual_blocks, odour_blocks = get_block_boundaries(visual_onsets, odour_onsets)
    visual_blocks, odour_blocks = convert_block_boundaries_to_trial_type(visual_onsets, odour_onsets, visual_blocks, odour_blocks)
    print("Visual Blocks", visual_blocks)
    print("Odour Blocks", odour_blocks)

    # Load Tensors
    visual_context_tensor = np.load(os.path.join(base_directory, "Allen_Activity_Tensor_Pre_Visual.npy"))
    odour_context_tensor = np.load(os.path.join(base_directory, "Allen_Activity_Tensor_Pre_Odour.npy"))
    print("Visual Context Tensor Shape", np.shape(visual_context_tensor))

    visual_block_maps = []
    odour_block_maps = []

    # Create Correlation Maps
    for visual_block in visual_blocks:

        # Extract Block Trials
        block_start = visual_block[0]
        block_stop = visual_block[1]
        block_tensor = visual_context_tensor[block_start:block_stop]

        # Concatenate and Subtract Mean
        block_tensor = concatenate_and_subtract_mean(block_tensor)

        # Get Correlation Matrix
        correlation_matrix = np.corrcoef(block_tensor)
        np.fill_diagonal(correlation_matrix, 0)

        # Append To List
        visual_block_maps.append(correlation_matrix)


    for odour_block in odour_blocks:

        # Extract Block Trials
        block_start = odour_block[0]
        block_stop = odour_block[1]
        block_tensor = odour_context_tensor[block_start:block_stop]

        # Concatenate and Subtract Mean
        block_tensor = concatenate_and_subtract_mean(block_tensor)

        # Get Correlation Matrix
        correlation_matrix = np.corrcoef(block_tensor)
        np.fill_diagonal(correlation_matrix, 0)

        # Append To List
        odour_block_maps.append(correlation_matrix)


    return visual_block_maps, odour_block_maps



def perform_decoding(X, y, base_directory):

    clf = RidgeClassifier()
    #clf = LogisticRegression(penalty='l1', solver='saga')
    score_list = []
    coefficient_list = []

    for n in range(5):

        strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=n)

        for train_index, test_index in strat_k_fold.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train Classifier
            clf.fit(X_train, y_train)

            # Predict Test Data
            y_pred = clf.predict(X_test)

            # Score Prediction
            score = accuracy_score(y_test, y_pred, normalize=True)
            score_list.append(score)

            # Get Coefficients
            coefficients = clf.coef_
            coefficient_list.append(coefficients)

    mean_score = np.mean(score_list)
    print("Scores: ", score_list)
    print("Mean Score: ", mean_score)




    # View Mean Coefficients
    coefficient_list = np.array(coefficient_list)
    mean_coefficients = np.mean(coefficient_list, axis=0)

    number_of_connections = np.shape(mean_coefficients)[1]
    print("Number of connections", number_of_connections)
    number_of_regions = int(np.sqrt(number_of_connections))
    print("Number of regions", number_of_regions)

    mean_coefficients = np.ndarray.reshape(mean_coefficients, (number_of_regions, number_of_regions))
    sorted_mean_coefficients = sort_matrix(mean_coefficients)
    max_coef = np.max(np.abs(coefficients))
    plt.imshow(sorted_mean_coefficients, cmap='bwr', vmin=-1*max_coef, vmax=max_coef)
    plt.show()

    # Load Mask
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    masked_atlas, boundary_indicies = Allen_Atlas_Drawing_Functions.add_region_boundaries(base_directory)
    cluster_centroids = np.load("/home/matthew/Documents/Allen_Atlas_Templates/Allen_Region_Centroids.npy")
    plt.scatter(cluster_centroids[:, 1], cluster_centroids[:, 0])
    plt.imshow(masked_atlas)
    plt.show()

    Draw_Brain_Network.draw_brain_network(cluster_centroids, masked_atlas, mean_coefficients, None)

def attempt_to_perform_decoding(visual_maps_list, odour_maps_list, session_list):

    number_of_visual_maps = np.shape(visual_maps_list)[0]
    number_of_odour_maps = np.shape(odour_maps_list)[0]
    number_of_regions = np.shape(visual_maps_list)[1]

    visual_maps_list = np.ndarray.reshape(visual_maps_list, (number_of_visual_maps, number_of_regions * number_of_regions))
    odour_maps_list = np.ndarray.reshape(odour_maps_list, (number_of_odour_maps, number_of_regions * number_of_regions))

    combined_maps = np.vstack([visual_maps_list, odour_maps_list])

    visual_labels = np.zeros(np.shape(visual_maps_list)[0])
    odour_labels = np.ones(np.shape(odour_maps_list)[0])
    combined_labels = np.concatenate([visual_labels, odour_labels])

    combined_maps = np.nan_to_num(combined_maps)
    print("Combined Maps Shape", np.shape(combined_maps))
    print("Combined Labels Shape", np.shape(combined_labels))

    """
    classifier = LogisticRegression(solver='saga')
    print("Combined Maps Shape", np.shape(combined_maps))
    scores = cross_val_score(classifier, combined_maps, combined_labels, scoring='accuracy', cv=10)
    print("Scores: ", scores)
    print("Mean Score: ", np.mean(scores))
    """

    perform_decoding(combined_maps, combined_labels, session_list[0])

    clf = RidgeClassifier()
    score_list = []
    for n in range(5):
        strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=n)
        scores = cross_val_score(clf, combined_maps, combined_labels, cv=strat_k_fold)
        for score in scores:
            score_list.append(score)

    print("Scores ", score_list)
    print("Mean Score", np.mean(score_list))

    """
    classifier.fit(combined_maps, combined_labels)

    coefficients = classifier.coef_
    print("Coefficients SHape", np.shape(coefficients))

    coefficients = np.ndarray.reshape(coefficients, (number_of_regions, number_of_regions))
    coefficients = sort_matrix(coefficients)

    plt.imshow(np.abs(coefficients))
    plt.show()
    """










controls = ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN78.1A/2020_12_09_Switching_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN78.1D/2020_11_29_Switching_Imaging/"]

mutants = [ "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK12.1F/2021_09_22_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN71.2A/2020_12_17_Switching_Imaging/"]


combined_list = controls + mutants
visual_onsets_file = "visual_context_stable_vis_2_frame_onsets.npy"
odour_onsets_file = "odour_context_stable_vis_2_frame_onsets.npy"
trial_start = -70
trial_stop = -0

visual_correlation_maps = []
odour_correlation_maps = []

for base_directory in controls:
    visual_block_maps, odour_block_maps = create_correlation_tensors(base_directory, visual_onsets_file, odour_onsets_file)

    for map in visual_block_maps:
        visual_correlation_maps.append(map)

    for map in odour_block_maps:
        odour_correlation_maps.append(map)


visual_correlation_maps = np.array(visual_correlation_maps)
odour_correlation_maps = np.array(odour_correlation_maps)

mean_visual_matrix = np.mean(visual_correlation_maps, axis=0)
mean_odour_matrix = np.mean(odour_correlation_maps, axis=0)
difference_matrix = np.diff([mean_visual_matrix, mean_odour_matrix], axis=0)[0]


#difference_matrix, [mean_visual_matrix, mean_odour_matrix] = jointly_sort_matricies(difference_matrix, [mean_visual_matrix, mean_odour_matrix])

figure_1 = plt.figure()
visual_axis = figure_1.add_subplot(1, 3, 1)
odour_axis = figure_1.add_subplot(1, 3, 2)
difference_axis = figure_1.add_subplot(1, 3, 3)

visual_axis.imshow(mean_visual_matrix, cmap='bwr', vmin=-1, vmax=1)
odour_axis.imshow(mean_odour_matrix, cmap='bwr', vmin=-1, vmax=1)
difference_axis.imshow(difference_matrix, cmap='bwr', vmin=-0.5, vmax=0.5)

plt.show()


attempt_to_perform_decoding(visual_correlation_maps, odour_correlation_maps, combined_list)