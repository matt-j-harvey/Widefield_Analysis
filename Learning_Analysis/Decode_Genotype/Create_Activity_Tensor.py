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
from tqdm import tqdm

import Decoding_Utils


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


def get_activity_tensor(activity_matrix, onsets, start_window, stop_window, start_cutoff=3000):

    number_of_pixels = np.shape(activity_matrix)[1]
    number_of_trials = np.shape(onsets)[0]
    number_of_timepoints = np.shape(activity_matrix)[0]

    # Create Empty Tensor To Hold Data
    activity_tensor = []

    # Get Correlation Matrix For Each Trial
    for trial_index in range(number_of_trials):

        # Get Trial Activity
        trial_start = onsets[trial_index] + start_window
        trial_stop = onsets[trial_index] + stop_window

        if trial_start > start_cutoff and trial_stop < number_of_timepoints:
            trial_activity = activity_matrix[trial_start:trial_stop]
            activity_tensor.append(trial_activity)

    activity_tensor = np.array(activity_tensor)
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


def correct_baseline(activity_tensor, trial_start):

    corrected_tensor = []
    for trial in activity_tensor:
        trial_baseline = trial[0: -1 * trial_start]
        trial_baseline = np.mean(trial_baseline, axis=0)
        trial = np.subtract(trial, trial_baseline)
        corrected_tensor.append(trial)
    corrected_tensor = np.array(corrected_tensor)

    return corrected_tensor


def create_activity_tensor(base_directory, onsets_file, trial_start, trial_stop, tensor_name, spatial_smoothing=False, smoothing_sd=2, save_tensor=True, baseline_correction=False):

    # Load Delta F Matrix
    delta_f_matrix_filepath = os.path.join(base_directory, "Delta_F.h5")
    delta_f_matrix_container = tables.open_file(delta_f_matrix_filepath, mode='r')
    delta_f_matrix = delta_f_matrix_container.root.Data

    # Load Onsets
    onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", onsets_file))

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

        print("Save Dirctory", save_directory)
        print("Tensor Name", tensor_name)
        np.save(os.path.join(save_directory, tensor_name + "_Activity_Tensor.npy"), activity_tensor)

    return activity_tensor


def reconstruct_activity_tensor(activity_tensor, indicies, image_height, image_width, alignment_dictionary):

    reconstructed_tensor = []

    for trial in activity_tensor:
        reconstructed_trial = []

        for frame in trial:

            # Reconstruct Image
            frame = Decoding_Utils.create_image_from_data(frame, indicies, image_height, image_width)

            # Align Image
            frame = Decoding_Utils.transform_image(frame, alignment_dictionary)

            reconstructed_trial.append(frame)
        reconstructed_tensor.append(reconstructed_trial)

    reconstructed_tensor = np.array(reconstructed_tensor)
    return reconstructed_tensor


def apply_shared_tight_mask(activity_tensor):

    # Load Tight Mask
    indicies, image_height, image_width = Decoding_Utils.load_tight_mask()

    transformed_tensor = []
    for trial in activity_tensor:
        transformed_trial = []

        for frame in trial:
            frame = np.ndarray.flatten(frame)
            frame = frame[indicies]
            transformed_trial.append(frame)
        transformed_tensor.append(transformed_trial)

    transformed_tensor = np.array(transformed_tensor)
    return transformed_tensor


def create_standard_alignment_tensor(base_directory, onsets_file, start_window, stop_window):

    # Load Mask
    indicies, image_height, image_width = Decoding_Utils.load_generous_mask(base_directory)

    # Load Alignment Dictionary
    alignment_dictionary = np.load(os.path.join(base_directory, "Cluster_Alignment_Dictionary.npy"), allow_pickle=True)[()]

    # Load Onsets
    onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", onsets_file))

    # Load Activity Matrix
    delta_f_file = os.path.join(base_directory, "Delta_F.h5")
    delta_f_container = tables.open_file(delta_f_file, "r")
    activity_matrix = delta_f_container.root.Data

    # Get Activity Tensor
    activity_tensor = get_activity_tensor(activity_matrix, onsets, start_window, stop_window)

    # Reconstruct Into Local Brain Space
    activity_tensor = reconstruct_activity_tensor(activity_tensor, indicies, image_height, image_width, alignment_dictionary)

    # Apply Shared Tight Mask
    activity_tensor = apply_shared_tight_mask(activity_tensor)

    # Close Delta F File
    delta_f_container.close()

    return activity_tensor



def get_array_name(base_directory):
    split_base_directory = os.path.normpath(base_directory)
    split_base_directory = split_base_directory.split(os.sep)
    mouse_name = split_base_directory[-2]
    session_name = split_base_directory[-1]
    array_name = mouse_name + "_" + session_name
    return array_name


def create_decoding_tensor(control_session_list, mutant_session_list, save_file, onsets_file, start_window, stop_window, baseline_correction=False):

    # Create Tables File
    file_container = tables.open_file(save_file, mode='w')
    control_group = file_container.create_group(where="/", name="Controls")
    mutant_group = file_container.create_group(where="/", name="Mutants")

    # Add Control Data
    for base_directory in tqdm(control_session_list):
        activity_tensor = create_standard_alignment_tensor(base_directory, onsets_file, start_window, stop_window)

        if baseline_correction == True:
            activity_tensor = correct_baseline(activity_tensor, start_window)

        array_name = get_array_name(base_directory)
        tensor_storage = file_container.create_carray(control_group, array_name, tables.UInt16Atom(), shape=(np.shape(activity_tensor)))
        tensor_storage[:] = activity_tensor
        tensor_storage.flush()

    # Add Mutant Data
    for base_directory in tqdm(mutant_session_list):
        activity_tensor = create_standard_alignment_tensor(base_directory, onsets_file, start_window, stop_window)

        if baseline_correction == True:
            activity_tensor = correct_baseline(activity_tensor, start_window)

        array_name = get_array_name(base_directory)
        tensor_storage = file_container.create_carray(mutant_group, array_name, tables.UInt16Atom(), shape=(np.shape(activity_tensor)))
        tensor_storage[:] = activity_tensor
        tensor_storage.flush()

    file_container.close()


"""
control_session_list = [r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_24_Discrimination_Imaging",
                        r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_25_Discrimination_Imaging",
                        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_22_Discrimination_Imaging",
                        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_24_Discrimination_Imaging",
                        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_09_Discrimination_Imaging",
                        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging"]

mutant_session_list = [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_09_Discrimination_Imaging",
                        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging",
                        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_14_Discrimination_Imaging",
                        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_15_Discrimination_Imaging",
                        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging",
                        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_08_Discrimination_Imaging"]

# Load Analysis Details
analysis_name = "Correct_Rejections_Response"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Decoding_Utils.load_analysis_container(analysis_name)

# Create Activity Tensors
save_file = os.path.join(r"/media/matthew/Expansion/Widefield_Analysis/Decoding_Analysis", analysis_name + "_Combined_Activity_Tensor_Baseline_Corrected.h5")
create_decoding_tensor(control_session_list, mutant_session_list, save_file, onset_files[0], start_window, stop_window, baseline_correction=True)
"""
