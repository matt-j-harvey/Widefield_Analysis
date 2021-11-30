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
from scipy.spatial.distance import cdist
from datetime import datetime

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions
import Draw_Brain_Network
import Allen_Atlas_Drawing_Functions






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


def correlate_one_with_many(one, many):
    c = 1 - cdist(one, many, metric='correlation')[0]
    return c


def create_seed_correlation_map(region_mean_trace, concatenated_tensor):

    number_of_pixels = np.shape(concatenated_tensor)[1]
    correlation_map = correlate_one_with_many(region_mean_trace, concatenated_tensor)

    """
    for pixel in range(number_of_pixels):
        pixel_trace = concatenated_tensor[pixel]
        correlation = np.corrcoef(region_mean_trace, pixel_trace)[0][1]
        correlation_map[pixel] = correlation
        print(pixel, "Correlation", correlation)
    """
    return correlation_map


def create_correlation_tensors(base_directory, onsets_file, trial_start, trial_stop, selected_regions):

    # Load Delta F Matrix
    delta_f_matrix_filepath = os.path.join(base_directory, "Delta_F.h5")
    delta_f_matrix_container = tables.open_file(delta_f_matrix_filepath, mode='r')
    delta_f_matrix = delta_f_matrix_container.root['Data']

    # Load Region Assigments
    pixel_assignments = np.load(os.path.join(base_directory, "Pixel_Assignmnets.npy"))

    # Get Selected Pixels
    # Visual Cortex
    selected_pixels = []
    for region in selected_regions:
        region_mask = np.where(pixel_assignments == region, 1, 0)
        region_indicies = np.nonzero(region_mask)[0]
        for index in region_indicies:
            selected_pixels.append(index)
    selected_pixels.sort()


    # Load Onsets
    onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", onsets_file))
    #print("Trials: ", len(onsets))

    # Create Trial Tensor
    activity_tensor = get_activity_tensor(delta_f_matrix, onsets, trial_start, trial_stop)
    #print("Raw Activity Tensor Shape", np.shape(activity_tensor))

    # Concatenate_and_subtract_Mean
    activity_tensor = concatenate_and_subtract_mean(activity_tensor)
    #print("Concatenated Shape", np.shape(activity_tensor))

    # Get Mean Response For Region For Each Trial
    region_trace = activity_tensor[selected_pixels]
    region_mean = np.mean(region_trace, axis=0)
    number_of_timepoints = np.shape(region_trace)[1]
    region_mean = np.ndarray.reshape(region_mean, (1, number_of_timepoints))
    #print(np.shape(region_trace))
    #print(np.shape(region_mean))

    # Create Correlation Map
    process_start_time = datetime.now()
    correlation_map = create_seed_correlation_map(region_mean, activity_tensor)
    process_stop_time = datetime.now()
    print("Trials: ", len(onsets), "Started: ", process_start_time, "Stopped:", process_stop_time)

    return correlation_map


def get_seed_correlation_maps(base_directory, visual_onsets_file, odour_onsets_file, trial_start, trial_stop):

    v1 = [45, 46]
    pmv = [47, 48]
    amv = [39, 40]
    rsc = [32, 28]
    s1 = [21, 24]
    m2 = [8, 9]

    print(base_directory)

    # Get Visual Maps
    v1_visual_correlation_map = create_correlation_tensors(base_directory,  visual_onsets_file, trial_start, trial_stop, v1)
    pmv_visual_correlation_map = create_correlation_tensors(base_directory, visual_onsets_file, trial_start, trial_stop, pmv)
    amv_visual_correlation_map = create_correlation_tensors(base_directory, visual_onsets_file, trial_start, trial_stop, amv)
    rsc_visual_correlation_map = create_correlation_tensors(base_directory, visual_onsets_file, trial_start, trial_stop, rsc)
    s1_visual_correlation_map = create_correlation_tensors(base_directory,  visual_onsets_file, trial_start, trial_stop, s1)
    m2_visual_correlation_map = create_correlation_tensors(base_directory,  visual_onsets_file, trial_start, trial_stop, m2)

    # Get Odour Maps
    v1_odour_correlation_map = create_correlation_tensors(base_directory,  odour_onsets_file, trial_start, trial_stop, v1)
    pmv_odour_correlation_map = create_correlation_tensors(base_directory, odour_onsets_file, trial_start, trial_stop, pmv)
    amv_odour_correlation_map = create_correlation_tensors(base_directory, odour_onsets_file, trial_start, trial_stop, amv)
    rsc_odour_correlation_map = create_correlation_tensors(base_directory, odour_onsets_file, trial_start, trial_stop, rsc)
    s1_odour_correlation_map = create_correlation_tensors(base_directory,  odour_onsets_file, trial_start, trial_stop, s1)
    m2_odour_correlation_map = create_correlation_tensors(base_directory, odour_onsets_file, trial_start, trial_stop, m2)

    # Save Visual Maps
    np.save(base_directory + "/V1_Visual_Correlation_Map.npy", v1_visual_correlation_map)
    np.save(base_directory + "/PMV_Visual_Correlation_Map.npy", pmv_visual_correlation_map)
    np.save(base_directory + "/AMV_Visual_Correlation_Map.npy", amv_visual_correlation_map)
    np.save(base_directory + "/RSC_Visual_Correlation_Map.npy", rsc_visual_correlation_map)
    np.save(base_directory + "/S1_Visual_Correlation_Map.npy", s1_visual_correlation_map)
    np.save(base_directory + "/M2_Visual_Correlation_Map.npy", m2_visual_correlation_map)

    # Save Odour Maps
    np.save(base_directory + "/V1_Odour_Correlation_Map.npy", v1_odour_correlation_map)
    np.save(base_directory + "/PMV_Odour_Correlation_Map.npy", pmv_odour_correlation_map)
    np.save(base_directory + "/AMV_Odour_Correlation_Map.npy", amv_odour_correlation_map)
    np.save(base_directory + "/RSC_Odour_Correlation_Map.npy", rsc_odour_correlation_map)
    np.save(base_directory + "/S1_Odour_Correlation_Map.npy", s1_odour_correlation_map)
    np.save(base_directory + "/M2_Odour_Correlation_Map.npy", m2_odour_correlation_map)



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


visual_onsets_file = "visual_context_stable_vis_2_frame_onsets.npy"
odour_onsets_file = "odour_context_stable_vis_2_frame_onsets.npy"
trial_start = 0
trial_stop = 20

for base_directory in controls:
    get_seed_correlation_maps(base_directory, visual_onsets_file, odour_onsets_file, trial_start, trial_stop)

for base_directory in mutants:
    get_seed_correlation_maps(base_directory, visual_onsets_file, odour_onsets_file, trial_start, trial_stop)

