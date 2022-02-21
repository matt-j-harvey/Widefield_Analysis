import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
import os
import math
import scipy
import tables
from bisect import bisect_left
import cv2
from sklearn.decomposition import TruncatedSVD
from pathlib import Path
import joblib
from scipy import signal, ndimage, stats
from skimage.transform import resize
from scipy.interpolate import interp1d
import sys
from sklearn.linear_model import LinearRegression
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")
sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Analysis/Movement_Controls/Residual_Analysis")

import Widefield_General_Functions
import Get_Running_Linear_Regression_Coefficients


def spatially_smooth_activity_tensor(base_directory, activity_tensor, sigma):

    # Get Tensor Shape
    number_of_trials = np.shape(activity_tensor)[0]
    number_of_timepoints = np.shape(activity_tensor)[1]


    # Load Mask
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    for trial_index in range(number_of_trials):
        for time_index in range(number_of_timepoints):

            timepoint_data = activity_tensor[trial_index, time_index]
            timepoint_image = Widefield_General_Functions.create_image_from_data(timepoint_data, indicies, image_height, image_width)
            timepoint_image = ndimage.gaussian_filter(timepoint_image, sigma=sigma)
            timepoint_image = np.ndarray.reshape(timepoint_image, (image_height * image_width))
            timepoint_data = timepoint_image[indicies]
            activity_tensor[trial_index, time_index] = timepoint_data

    return activity_tensor


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


def get_onsets(base_directory, onsets_file_list):

    onsets = []
    for onsets_file in onsets_file_list:
        onsets_file_contents = np.load(os.path.join(base_directory, "Stimuli_Onsets", onsets_file))
        for onset in onsets_file_contents:
            onsets.append(onset)

    return onsets


"""
def correct_running_activity_tensor(base_directory, onsets, trial_start, trial_stop, activity_tensor, running_coefficients, display=False):

    # Load Running Information
    downsampled_running_trace = np.load(os.path.join(base_directory, "Movement_Controls", "Downsampled_Running_Trace.npy"))

    running_coefficients = np.expand_dims(running_coefficients, axis=0)

    predicited_tensor = []
    corrected_tensor = []

    number_of_trials = len(onsets)
    for trial_index in range(number_of_trials):

        # Get Trial Details
        onset = onsets[trial_index]
        start = onset + trial_start
        stop = onset + trial_stop

        # Get Trial Running Trace
        trial_running_trace = downsampled_running_trace[start:stop]
        trial_running_trace = np.expand_dims(trial_running_trace, axis=1)

        # Predict Activity
        predicited_activity = np.matmul(trial_running_trace, running_coefficients)
        print("Predicited Activity Shape", np.shape(predicited_activity))

        # Get Actual Activity
        actual_activity = activity_tensor[trial_index]
        print("Actual Activity Shape", np.shape(actual_activity))

        # Get Corrected Activity
        corrected_activity = np.subtract(actual_activity, predicited_activity)
        print("Corrected Activity Shape", np.shape(corrected_activity))

        # Add These To Tensors
        predicited_tensor.append(predicited_activity)
        corrected_tensor.append(corrected_activity)

        if display == True:
            figure_1 = plt.figure()
            real_axis       = figure_1.add_subplot(1, 3 ,1)
            predicited_axis = figure_1.add_subplot(1, 3, 2)
            corrected_axis  = figure_1.add_subplot(1, 3, 3)

            real_axis.imshow(np.transpose(actual_activity),             cmap='jet', vmin=0, vmax=1)
            predicited_axis.imshow(np.transpose(predicited_activity),   cmap='jet', vmin=0, vmax=1)
            corrected_axis.imshow(np.transpose(corrected_activity),     cmap='jet', vmin=0, vmax=1)

            real_axis.set_aspect('auto')
            predicited_axis.set_aspect('auto')
            corrected_axis.set_aspect('auto')

            print("Predicted Activity Shape", np.shape(predicited_activity))
            plt.show()


    predicited_tensor = np.array(predicited_tensor)
    corrected_tensor = np.array(corrected_tensor)

    return predicited_tensor, corrected_tensor




def create_activity_tensor_tables(base_directory, onsets_file_list, trial_start, trial_stop, tensor_name, running_correction=False, spatial_smoothing=False, smoothing_sd=2):
    print(base_directory)


    # Load Delta F Matrix
    delta_f_matrix_filepath = os.path.join(base_directory, "Delta_F.h5")
    delta_f_matrix_container = tables.open_file(delta_f_matrix_filepath, mode='r')
    delta_f_matrix = delta_f_matrix_container.root['Data']

    # Load Onsets
    onsets = []
    for onsets_file in onsets_file_list:
        print(onsets_file_list)
        print(onsets_file)
        onsets_file_contents = np.load(os.path.join(base_directory, "Stimuli_Onsets", onsets_file))
        for onset in onsets_file_contents:
            onsets.append(onset)
    print("Number_of_trails: ", len(onsets))

    # Create Trial Tensor
    activity_tensor = get_activity_tensor(delta_f_matrix, onsets, trial_start, trial_stop)

    # Smooth if required
    if spatial_smoothing == True:
        activity_tensor = spatially_smooth_activity_tensor(base_directory, activity_tensor, sigma=smoothing_sd)


    # Save Tensors
    save_directory = os.path.join(base_directory, "Activity_Tensors")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    if running_correction == True:

        # Get Running Regression coefficients
        coefficient_save_directory = os.path.join(save_directory, "Movement_Controls", "Running_Regression_Coefficients")
        running_coefficients = get_running_regression_coefficients(base_directory, activity_tensor, onsets, trial_start, trial_stop, coefficient_save_directory, tensor_name)

        predicted_tensor, corrected_tensor = correct_running_activity_tensor(base_directory, onsets, trial_start, trial_stop, activity_tensor, running_coefficients)
        np.save(os.path.join(save_directory, tensor_name + "_Activity_Tensor.npy"), activity_tensor)
        np.save(os.path.join(save_directory, tensor_name + "_Predicted_Tensor.npy"), predicted_tensor)
        np.save(os.path.join(save_directory, tensor_name + "_Corrected_Tensor.npy"), corrected_tensor)
    else:
        np.save(os.path.join(save_directory, tensor_name + "_Activity_Tensor.npy"), activity_tensor)

"""





def get_region_activity(tensor, region_map, selected_regions):

    # Create Binary Map
    binary_map = np.isin(region_map, selected_regions)

    # Get Region Indicies
    region_indicies = np.argwhere(binary_map)

    # Get Region Traces
    region_tensor = tensor[:, :, region_indicies]

    # Get Mean Trace
    region_mean = np.mean(region_tensor, axis=2)

    return region_mean





def create_allen_atlas_activity_tensor(base_directory, onsets_file_list, trial_start, trial_stop):
    print(base_directory)

    # Load Delta F Matrix
    delta_f_matrix = np.load(os.path.join(base_directory, "Allen_Region_Delta_F.npy"))

    # Load Onsets
    onsets = []
    print("Onsets file List", onsets_file_list)
    for onsets_file in onsets_file_list:
        print("Onsets File", onsets_file)
        onsets_file_contents = np.load(os.path.join(base_directory, "Stimuli_Onsets", onsets_file))
        for onset in onsets_file_contents:
            onsets.append(onset)
    print("Number_of_trails: ", len(onsets))

    # Create Trial Tensor
    activity_tensor = get_activity_tensor(delta_f_matrix, onsets, trial_start, trial_stop)


    return activity_tensor



def create_activity_tensor(base_directory, onsets_file_list, trial_start, trial_stop, tensor_name, spatial_smoothing=False, smoothing_sd=2, save_tensor=True):
    print(base_directory)


    # Load Delta F Matrix
    delta_f_matrix_filepath = os.path.join(base_directory, "Delta_F_Registered.hdf5")
    delta_f_matrix_container = h5py.File(delta_f_matrix_filepath, 'r')
    delta_f_matrix = delta_f_matrix_container['Data']

    # Load Onsets
    onsets = []
    print("Onsets file List", onsets_file_list)
    for onsets_file in onsets_file_list:
        print("Onsets File", onsets_file)
        onsets_file_contents = np.load(os.path.join(base_directory, "Stimuli_Onsets", onsets_file))
        for onset in onsets_file_contents:
            onsets.append(onset)
    print("Number_of_trails: ", len(onsets))

    # Create Trial Tensor
    activity_tensor = get_activity_tensor(delta_f_matrix, onsets, trial_start, trial_stop)

    # Smooth if required
    if spatial_smoothing == True:
        activity_tensor = spatially_smooth_activity_tensor(base_directory, activity_tensor, sigma=smoothing_sd)

    # Save Tensors
    if save_tensor == True:
        save_directory = os.path.join(base_directory, "Activity_Tensors")
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)

        np.save(os.path.join(save_directory, tensor_name + "_Activity_Tensor.npy"), activity_tensor)

    return activity_tensor
