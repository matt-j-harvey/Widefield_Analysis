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
from sklearn.linear_model import LogisticRegression

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions


def get_activity_tensor(activity_matrix, onsets, start_window, stop_window, base_directory):

    number_of_pixels = np.shape(activity_matrix)[1]
    number_of_trials = np.shape(onsets)[0]
    number_of_timepoints = stop_window - start_window

    # Create Empty Tensor To Hold Data
    activity_tensor = np.zeros((number_of_trials, number_of_timepoints, number_of_pixels))

    # Get Correlation Matrix For Each Trial
    for trial_index in range(number_of_trials):

        # Get Trial Activity
        trial_start = onsets[trial_index] + start_window
        trial_stop = onsets[trial_index] + stop_window
        trial_activity = activity_matrix[trial_start:trial_stop]

        # Remove Nans
        trial_activity = np.nan_to_num(trial_activity)

        # Smooth
        trial_activity = smooth_maps(trial_activity, base_directory)

        # Add To Tensor
        activity_tensor[trial_index] = trial_activity

    return activity_tensor


def concatenate_and_subtract_mean(tensor):
    # Get Tensor Structure
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




def get_selected_pixels(selected_regions, pixel_assignments):

    selected_pixels = []
    for region in selected_regions:
        region_mask = np.where(pixel_assignments == region, 1, 0)
        region_indicies = np.nonzero(region_mask)[0]
        for index in region_indicies:
            selected_pixels.append(index)
    selected_pixels.sort()

    return selected_pixels


def get_region_correlation_vector(region_pixels, activity_tensor):

    number_of_pixels = np.shape(activity_tensor)[0]
    number_of_pixels_in_region = len(region_pixels)
    correlation_map = np.zeros((number_of_pixels))

    # Get Mean Region Trace
    region_trace = activity_tensor[region_pixels]
    region_trace = np.mean(region_trace, axis=0)

    for pixel in range(number_of_pixels):
        correlation = np.corrcoef(region_trace, activity_tensor[pixel])[0][1]
        correlation_map[pixel] = correlation

    return correlation_map



def get_block_boundaries(combined_onsets, visual_context_onsets, odour_context_onsets):

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


def split_onsets_into_blocks(visual_onsets, odour_onsets):

    # Get Combined Onsets
    combined_onsets = np.concatenate([visual_onsets, odour_onsets])
    combined_onsets = list(combined_onsets)
    combined_onsets.sort()

    visual_blocks = []
    odour_blocks = []

    current_block = []
    current_block.append(combined_onsets[0])

    # Get Initial Onset
    if combined_onsets[0] in visual_onsets:
        current_block_type = 0
    elif combined_onsets[0] in odour_onsets:
        current_block_type = 1
    else:
        print("Error! onsets not in either visual or oflactory onsets")


    # Iterate Through All Subsequent Onsets
    number_of_onsets = len(combined_onsets)
    for onset_index in range(1, number_of_onsets):

        # Get Onset
        onset = combined_onsets[onset_index]

        # If we are currently in an Visual Block
        if current_block_type == 0:

            # If The Next Onset is An Odour Block - Block Finish, add Block To Boundaries
            if onset in odour_onsets:
                visual_blocks.append(current_block)
                current_block_type = 1
                current_block = []
                current_block.append(onset)
            else:
                current_block.append(onset)

        # If we Are currently in an Odour BLock
        elif current_block_type == 1:

            # If The Next Onset Is a Visual Trial - BLock Finish Add Block To Block Boundaires
            if onset in visual_onsets:
                odour_blocks.append(current_block)
                current_block_type = 0
                current_block = []
                current_block.append(onset)
            else:
                current_block.append(onset)

    return visual_blocks, odour_blocks


def get_maps_for_block_list(block_list, selected_pixels, delta_f_matrix, trial_start, trial_stop, base_directory):

    block_maps = []

    for block in block_list:

        # Get Activity Tensor
        activity_tensor = get_activity_tensor(delta_f_matrix, block, trial_start, trial_stop, base_directory)

        # Concatenate Activity Tensor And Subtract Mean
        activity_tensor = concatenate_and_subtract_mean(activity_tensor)

        # Get Region Correlation Map
        correlation_map = get_region_correlation_vector(selected_pixels, activity_tensor)

        block_maps.append(correlation_map)

    return block_maps


def view_coefficients(base_directory, vector):

    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)
    image = Widefield_General_Functions.create_image_from_data(vector, indicies, image_height, image_width)
    plt.imshow(np.abs(image), cmap='inferno')
    plt.show()
    return image_matrix




def get_session_correlation_maps(session_list, onsets_file_list, trial_start, trial_stop):

    # Open Atlas Labels
    atlas_labels = np.recfromcsv(r"/home/matthew/Documents/Allen_Atlas_Templates/Atlas_Labels.csv")
    print(atlas_labels)
    atlas_image = np.load(r"/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/Pixel_Assignmnets_Image.npy")
    plt.imshow(atlas_image)
    plt.show()

    for base_directory in session_list:
        print(base_directory)

        # Load Delta F Matrix
        delta_f_matrix_filepath = os.path.join(base_directory, "Delta_F.h5")
        delta_f_matrix_container = tables.open_file(delta_f_matrix_filepath, mode='r')
        delta_f_matrix = delta_f_matrix_container.root['Data']

        # Load Region Assigments
        pixel_assignments = np.load(os.path.join(base_directory, "Pixel_Assignmnets.npy"))

        # Get Selected Pixels
        selected_regions = [45, 46]
        selected_pixels = get_selected_pixels(selected_regions, pixel_assignments)

        # Get Correlation Map
        condition_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", onsets_file_list[0]))
        condition_2_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", onsets_file_list[1]))

        # Split Onsets Into Blocks
        visual_blocks, odour_blocks = split_onsets_into_blocks(condition_1_onsets, condition_2_onsets)

        # Get Maps For These Blocks
        visual_context_maps = get_maps_for_block_list(visual_blocks, selected_pixels, delta_f_matrix, trial_start, trial_stop, base_directory)
        odour_context_maps = get_maps_for_block_list(odour_blocks, selected_pixels, delta_f_matrix, trial_start, trial_stop, base_directory)

        # View Average Maps
        mean_visual_context_map = np.mean(visual_context_maps, axis=0)
        mean_odour_context_map = np.mean(odour_context_maps, axis=0)

        compare_correlation_vectors(mean_visual_context_map, mean_odour_context_map, base_directory)

        # Save Map Lists
        save_directory = os.path.join(base_directory, "Block_Pre_Stim_Correlation_Maps")
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)

        np.save(os.path.join(save_directory, "Visual_Context_Maps.npy"), visual_context_maps)
        np.save(os.path.join(save_directory, "Odour_Context_Maps.npy"), odour_context_maps)



def compare_correlation_vectors(condition_1_vector, condition_2_vector, base_directory):

    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    condition_1_image = Widefield_General_Functions.create_image_from_data(condition_1_vector, indicies, image_height, image_width)
    condition_2_image = Widefield_General_Functions.create_image_from_data(condition_2_vector, indicies, image_height, image_width)
    difference_image = np.diff([condition_1_image, condition_2_image], axis=0)[0]

    condition_1_image_magnitude = np.max(np.abs(condition_1_image))
    condition_2_image_magnitude = np.max(np.abs(condition_2_image))
    image_magnitude = np.max([condition_1_image_magnitude, condition_2_image_magnitude])
    difference_magnitude = np.max(np.abs(difference_image))

    figure_1 = plt.figure()
    condition_1_axis = figure_1.add_subplot(1, 3, 1)
    condition_2_axis = figure_1.add_subplot(1, 3, 2)
    difference_axis = figure_1.add_subplot(1, 3, 3)

    condition_1_axis.imshow(condition_1_image, cmap='bwr', vmin=-1 * image_magnitude, vmax=image_magnitude)
    condition_2_axis.imshow(condition_2_image, cmap='bwr', vmin=-1 * image_magnitude, vmax=image_magnitude)
    difference_axis.imshow(difference_image,   cmap='bwr', vmin=-1 * difference_magnitude, vmax=1 * difference_magnitude)

    plt.show()


def smooth_maps(maps, base_directory):

    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)
    maps = np.nan_to_num(maps)

    smoothed_maps = []
    for map in maps:
        image = Widefield_General_Functions.create_image_from_data(map, indicies, image_height, image_width)
        image = ndimage.gaussian_filter(image, sigma=2)
        image = np.ndarray.flatten(image)
        smoothed_image = image[indicies]
        smoothed_maps.append(smoothed_image)

    smoothed_maps = np.array(smoothed_maps)
    print("Smoothed maps shape", np.shape(smoothed_maps))
    return smoothed_maps


def attempt_to_perform_decoding(session_list):
    visual_maps_list = []
    odour_maps_list = []

    for base_directory in session_list:
        map_directory = os.path.join(base_directory, "Block_Pre_Stim_Correlation_Maps")
        visual_maps = np.load(os.path.join(map_directory, "Visual_Context_Maps.npy"))
        odour_maps = np.load(os.path.join(map_directory, "Odour_Context_Maps.npy"))

        for visual_map in visual_maps:
            visual_maps_list.append(visual_map)

        for odour_map in odour_maps:
            odour_maps_list.append(odour_map)

    visual_maps_list = np.array(visual_maps_list)
    odour_maps_list = np.array(odour_maps_list)

    # Smooth Maps
    visual_maps_list = smooth_maps(visual_maps_list, session_list[0])
    odour_maps_list = smooth_maps(odour_maps_list, session_list[0])

    combined_maps = np.vstack([visual_maps_list, odour_maps_list])

    mean_visual_map = np.mean(visual_maps_list, axis=0)
    mean_odour_map = np.mean(odour_maps_list, axis=0)

    compare_correlation_vectors(mean_visual_map, mean_odour_map, session_list[0])


    visual_labels = np.zeros(np.shape(visual_maps_list)[0])
    odour_labels = np.ones(np.shape(odour_maps_list)[0])
    combined_labels = np.concatenate([visual_labels, odour_labels])

    combined_maps = np.nan_to_num(combined_maps)
    print("Combined Maps Shape", np.shape(combined_maps))
    print("Combined Labels Shape", np.shape(combined_labels))

    classifier = LogisticRegression(penalty='l2', solver='saga')
    classifier.fit(combined_maps, combined_labels)

    coefficients = classifier.coef_
    view_coefficients(session_list[0], coefficients)


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


trial_start = -70
trial_stop = -0

visual_onsets_file = "visual_context_stable_vis_2_frame_onsets.npy"
odour_onsets_file = "odour_context_stable_vis_2_frame_onsets.npy"

onsets_file_list = [visual_onsets_file, odour_onsets_file]
get_session_correlation_maps(controls + mutants, onsets_file_list, trial_start, trial_stop)



attempt_to_perform_decoding(mutants)

