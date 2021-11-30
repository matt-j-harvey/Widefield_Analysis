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


def get_activity_tensor(activity_matrix, onsets, start_window, stop_window):

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

        # Add To Tensor
        activity_tensor[trial_index] = trial_activity

    return activity_tensor


def get_selected_pixels(selected_regions, pixel_assignments):

    selected_pixels = []
    for region in selected_regions:
        region_mask = np.where(pixel_assignments == region, 1, 0)
        region_indicies = np.nonzero(region_mask)[0]
        for index in region_indicies:
            selected_pixels.append(index)
    selected_pixels.sort()

    return selected_pixels


def get_allen_atlas_region_tensor(base_directory, onsets_file, trial_start, trial_stop, save_directory):

    # Load Delta F Matrix
    delta_f_matrix_filepath = os.path.join(base_directory, "Delta_F.h5")
    delta_f_matrix_container = tables.open_file(delta_f_matrix_filepath, mode='r')
    delta_f_matrix = delta_f_matrix_container.root['Data']

    # Load Onsets
    onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", onsets_file))

    # Get Activity Tensor
    activity_tensor = get_activity_tensor(delta_f_matrix, onsets, trial_start, trial_stop)
    print("Activity tensor shape", np.shape(activity_tensor))

    # Load Region Assigments
    pixel_assignments = np.load(os.path.join(base_directory, "Pixel_Assignmnets.npy"))
    pixel_assignments = np.ndarray.astype(pixel_assignments, np.int)

    #pixel_assignments_image = np.load(os.path.join(base_directory, "Pixel_Assignmnets_Image.npy"))
    #plt.imshow(pixel_assignments_image, cmap='jet')
    #plt.show()

    # Get Number Of Regions
    region_list = list(pixel_assignments)
    region_list = set(region_list)
    region_list = list(region_list)
    number_of_regions = len(region_list)
    number_of_timepoints = trial_stop - trial_start
    number_of_trials = len(onsets)

    print("Number of trials", number_of_trials)
    print("Number of timepoints", number_of_timepoints)
    print("Number of regions", number_of_regions)

    print("Region List", region_list)
    allen_region_tensor = []

    excluded_list = [0, 1, 3, 4, 5, 10, 17, 18]
    for region in region_list:
        if region not in excluded_list:
            print("Region: ", region)
            selected_pixels = get_selected_pixels([region], pixel_assignments)
            region_activity = activity_tensor[:, :, selected_pixels]
            print("Region Activity Shape", np.shape(region_activity))
            region_activity = np.mean(region_activity, axis=2)
            allen_region_tensor.append(region_activity)

    allen_region_tensor = np.array(allen_region_tensor)
    allen_region_tensor = np.moveaxis(allen_region_tensor, source=[0,1,2], destination=[2,0,1])
    print("Full Tensor Shape", np.shape(allen_region_tensor))

    # Save Region Tensor
    full_save_directory = os.path.join(base_directory, save_directory)
    np.save(full_save_directory, allen_region_tensor)



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

visual_onsets_file = "visual_context_stable_vis_2_frame_onsets_Matched.npy"
odour_onsets_file = "odour_context_stable_vis_2_frame_onsets_Matched.npy"

trial_start = -10
trial_stop = 40


# Get Region Tensors
for base_directory in combined_list:
    get_allen_atlas_region_tensor(base_directory, visual_onsets_file, trial_start, trial_stop, "Allen_Activity_Tensor_Vis_2_Visual.npy")
    get_allen_atlas_region_tensor(base_directory, odour_onsets_file, trial_start, trial_stop, "Allen_Activity_Tensor_Vis_2_Odour.npy")