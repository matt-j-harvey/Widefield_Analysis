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
import h5py
import sys
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
from datetime import datetime

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions



def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


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

    # Subtract mean
    trial_mean = np.mean(activity_tensor, axis=0)
    activity_tensor = np.subtract(activity_tensor, trial_mean)

    activity_tensor = np.reshape(activity_tensor, (number_of_trials * number_of_timepoints, number_of_pixels))
    return activity_tensor



def create_correlation_tensors(base_directory, onsets_file, trial_start, trial_stop, selected_regions):

    # Load Delta F Matrix
    delta_f_matrix_filepath = os.path.join(base_directory, "Delta_F_Registered.hdf5")
    delta_f_matrix_container = h5py.File(delta_f_matrix_filepath, 'r')
    delta_f_matrix = delta_f_matrix_container['Data']

    # Load Region Assigments
    pixel_assignments = np.load("/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/Pixel_Assignmnets.npy")
    #print("Pixel assignments", set(np.unique(pixel_assignments)))


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
    activity_tensor = np.nan_to_num(activity_tensor)

    # Get Region Trace
    region_trace = activity_tensor[:, selected_pixels]

    print("Activity tensor shape", np.shape(activity_tensor))
    region_trace = np.mean(region_trace, axis=1)


    # Perform Z Scoreing
    region_trace = stats.zscore(region_trace)
    activity_tensor = stats.zscore(activity_tensor, axis=0)

    region_trace = np.reshape(region_trace, (np.shape(region_trace)[0], 1))

    plt.plot(region_trace)
    plt.show()

    print("REgiong trace shape", np.shape(region_trace))
    print("activity tensor shape", np.shape(activity_tensor))

    # Get Cofluctuations
    activity_tensor = np.multiply(region_trace, activity_tensor)


    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    axis_1.imshow(activity_tensor, cmap='bwr')
    forceAspect(axis_1)
    plt.show()

    figure_1 = plt.figure()
    plt.ion()
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)
    for timepoint in activity_tensor:
        axis_1 = figure_1.add_subplot(1,1,1)
        data_image = Widefield_General_Functions.create_image_from_data(timepoint, indicies, image_height, image_width)
        axis_1.imshow(data_image, cmap='bwr', vmin=-0.5, vmax=0.5)
        plt.draw()
        plt.pause(0.1)
        plt.clf()


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
    v1_visual_correlation_map = create_correlation_tensors(base_directory,  visual_onsets_file, trial_start, trial_stop, rsc)
    v1_odour_correlation_map = create_correlation_tensors(base_directory,  odour_onsets_file, trial_start, trial_stop, rsc)



    # Load Mask
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    v1_visual_correlation_map = Widefield_General_Functions.create_image_from_data(v1_visual_correlation_map, indicies, image_height, image_width)
    plt.imshow(v1_visual_correlation_map)
    plt.show()
    
    # Get Odour Maps
    # Save Visual Maps
    #np.save(base_directory + "/V1_Visual_Correlation_Map.npy",  v1_visual_correlation_map)
    #np.save(base_directory + "/V1_Odour_Correlation_Map.npy",  v1_odour_correlation_map)


controls = [
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_02_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_08_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_23_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_31_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_15_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_10_29_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_11_05_Transition_Imaging"
            ]


mutants = [
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_10_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK24.1C/2021_11_10_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive1/Transition_2/NXAK20.1B/2021_11_22_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Transition_2/NXAK20.1B/2021_11_24_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Transition_2/NXAK20.1B/2021_11_26_Transition_Imaging",
]


visual_onsets_file = "visual_context_stable_vis_2_onsets.npy"
odour_onsets_file = "odour_context_stable_vis_2_onsets.npy"
trial_start = -10
trial_stop = 40

for base_directory in controls:
    get_seed_correlation_maps(base_directory, visual_onsets_file, odour_onsets_file, trial_start, trial_stop)

for base_directory in mutants:
    get_seed_correlation_maps(base_directory, visual_onsets_file, odour_onsets_file, trial_start, trial_stop)

