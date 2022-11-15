import numpy as np
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

import Trial_Aligned_Utils


def get_trial_stop(trial_start, stop_trace, trace_threshold, max_length=140):

    number_of_timepoints = len(stop_trace)
    trial_ongoing = True
    trial_length = 0
    while trial_ongoing:

        if trial_start + trial_length >= number_of_timepoints:
            trial_ongoing = False
            return None
        else:
            if stop_trace[trial_start + trial_length] > trace_threshold or trial_length > max_length:
                trial_ongoing = False
                return trial_start + trial_length
            else:
                trial_length += 1


def get_ragged_activity_tensor(activity_matrix, onsets, start_window, stop_trace, trace_threshold=0.5, start_cutoff=3000):

    number_of_pixels = np.shape(activity_matrix)[1]
    number_of_trials = np.shape(onsets)[0]
    number_of_timepoints = np.shape(activity_matrix)[0]

    # Create Empty Tensor To Hold Data
    activity_tensor = []

    # Get Correlation Matrix For Each Trial
    for trial_index in range(number_of_trials):

        # Get Trial Activity
        trial_start = onsets[trial_index] + start_window
        trial_stop = get_trial_stop(onsets[trial_index], stop_trace, trace_threshold)
        if trial_stop != None:

            if trial_start > start_cutoff and trial_stop < number_of_timepoints:
                trial_activity = activity_matrix[trial_start:trial_stop]
                activity_tensor.append(trial_activity)

    activity_tensor = np.array(activity_tensor)
    return activity_tensor



def pad_ragged_tensor_with_nans(ragged_tensor):

    # Get Longest Trial
    length_list = []
    for trial in ragged_tensor:
        trial_length, number_of_pixels = np.shape(trial)
        length_list.append(trial_length)

    max_length = np.max(length_list)

    # Create Padded Tensor
    number_of_trials = len(length_list)
    padded_tensor = np.empty((number_of_trials, max_length, number_of_pixels))
    padded_tensor[:] = np.nan

    # Fill Padded Tensor
    for trial_index in range(number_of_trials):
        trial_data = ragged_tensor[trial_index]
        trial_length = np.shape(trial_data)[0]
        padded_tensor[trial_index, 0:trial_length] = trial_data

    return padded_tensor


def reconstruct_activity_tensor(activity_tensor, indicies, image_height, image_width, alignment_dictionary):

    reconstructed_tensor = []

    for trial in activity_tensor:
        reconstructed_trial = []

        for frame in trial:

            # Reconstruct Image
            frame = Trial_Aligned_Utils.create_image_from_data(frame, indicies, image_height, image_width)

            # Cheeky Smooth
            frame = ndimage.gaussian_filter(frame, sigma=1)

            # Align Image
            frame = Trial_Aligned_Utils.transform_image(frame, alignment_dictionary)

            reconstructed_trial.append(frame)
        reconstructed_tensor.append(reconstructed_trial)

    reconstructed_tensor = np.array(reconstructed_tensor)
    return reconstructed_tensor


def apply_shared_tight_mask(activity_tensor):

    # Load Tight Mask
    indicies, image_height, image_width = Trial_Aligned_Utils.load_tight_mask()

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


def create_combined_stimulus_trace(behaviour_matrix, trace_name_list):

    # Create AI Channel Dict
    ai_channel_dict = Trial_Aligned_Utils.create_stimuli_dictionary()

    if len(trace_name_list) > 1:
        trace_list = []
        for trace_name in trace_name_list:
            trace = behaviour_matrix[ai_channel_dict[trace_name]]
            trace_list.append(trace)

        combined_stimuli_trace = np.vstack(trace_list)
        combined_stimuli_trace = np.max(combined_stimuli_trace, axis=0)
    else:
        combined_stimuli_trace = behaviour_matrix[ai_channel_dict[trace_name]]

    return combined_stimuli_trace



def regression_motor_correction(trial_activity, trial_start, trial_stop, behaviour_matrix, face_motion_components, regression_dictionary):

    # Get Lick and Running Traces
    stimuli_dictionary = Trial_Aligned_Utils.create_stimuli_dictionary()
    running_trace = behaviour_matrix[stimuli_dictionary["Running"]]
    lick_trace = behaviour_matrix[stimuli_dictionary["Lick"]]

    # Get Regression Coefs
    regression_coefs = regression_dictionary["Coefs"]
    regression_coefs = np.transpose(regression_coefs)
    regression_intercepts = regression_dictionary["Intercepts"]

    # Create Design Matrix
    design_matrix = np.hstack([
        np.expand_dims(lick_trace[trial_start:trial_stop], 1),
        np.expand_dims(running_trace[trial_start:trial_stop], 1),
        face_motion_components[trial_start:trial_stop]
    ])

    # Get Predicted Activity
    predicted_activity = np.dot(design_matrix, regression_coefs)
    predicted_activity = np.add(predicted_activity, regression_intercepts)

    residual_activity = np.subtract(trial_activity, predicted_activity)

    # Visualise Predicted Activity
    """
    indicies, image_height, image_width = Transition_Utils.load_generous_mask(base_directory)
    colourmap = Transition_Utils.get_mussall_cmap()
    residual_magnitude = np.max(np.abs(residual_activity))

    plt.ion()
    for frame in residual_activity:
        frame = Transition_Utils.create_image_from_data(frame, indicies, image_height, image_width)
        plt.imshow(frame, cmap=colourmap, vmax=residual_magnitude, vmin=-1*residual_magnitude)
        plt.draw()
        plt.pause(0.1)
        plt.clf()
    """

    return residual_activity



def get_ragged_tensors(activity_matrix, full_behaviour_matrix, selected_ai_channels, trial_start_stop_list, movement_correction, face_motion_components, regression_dict=None):

    # Get Selected Traces
    channel_index_dictionary = Trial_Aligned_Utils.create_stimuli_dictionary()
    selected_channels_list = []
    for behaviour_trace in selected_ai_channels:
        selected_channels_list.append(channel_index_dictionary[behaviour_trace])

    # Get Data Structure
    behaviour_matrix = np.transpose(full_behaviour_matrix)
    behaviour_matrix = behaviour_matrix[:, selected_channels_list]
    number_of_trials = len(trial_start_stop_list)

    # Create Empty Tensor To Hold Data
    activity_tensor = []
    behaviour_tensor = []

    # Get Correlation Matrix For Each Trial
    for trial_index in range(number_of_trials):

        # Get Trial Start and Stop
        trial_start = trial_start_stop_list[trial_index][0]
        trial_stop = trial_start_stop_list[trial_index][1]

        trial_behaviour = behaviour_matrix[trial_start:trial_stop]
        trial_activity = activity_matrix[trial_start:trial_stop]

        # If Motion Correction Is True - Correct For This
        if movement_correction == True:
            trial_activity = regression_motor_correction(trial_activity, trial_start, trial_stop, full_behaviour_matrix, face_motion_components, regression_dict)

        activity_tensor.append(trial_activity)
        behaviour_tensor.append(trial_behaviour)

    behaviour_tensor = np.array(behaviour_tensor)
    activity_tensor = np.array(activity_tensor)

    return activity_tensor, behaviour_tensor


def get_trial_stats_and_stops(onsets_list, number_of_timepoints, start_window, stop_trace, trace_threshold, start_cutoff):

    number_of_trials = np.shape(onsets_list)[0]
    trial_start_stop_tuple_list = []

    for trial_index in range(number_of_trials):

        trial_start = onsets_list[trial_index] + start_window
        trial_stop = get_trial_stop(onsets_list[trial_index], stop_trace, trace_threshold)

        if trial_stop != None:
            if trial_start > start_cutoff and trial_stop < number_of_timepoints:
                trial_start_stop_tuple_list.append([trial_start, trial_stop])

    return trial_start_stop_tuple_list

def visualise_reconstructed_tensor(tensor):

    for frame in tensor[0]:
        plt.imshow(frame)
        plt.show()

def visualise_tight_mask_tensor(tensor):

    # Load Tight Mask
    indicies, image_height, image_width = Trial_Aligned_Utils.load_tight_mask()

    for frame in tensor[0]:
        template = np.zeros(image_height * image_width)
        template[indicies] = frame
        template = np.reshape(template, (image_height, image_width))
        plt.imshow(template)
        plt.show()


def create_extended_standard_alignment_tensor(base_directory, tensor_save_directory, onsets_file, start_window, stop_stimuli, selected_ai_channels, trace_threshold=0.5, start_cutoff=3000, min_trial_number=1, max_trial_num=50, movement_correction=True):

    # Load Mask
    indicies, image_height, image_width = Trial_Aligned_Utils.load_downsampled_mask(base_directory)

    # Load Alignment Dictionary
    alignment_dictionary = np.load(os.path.join(base_directory, "Brain_Alignment_Dictionary.npy"), allow_pickle=True)[()]

    # Load Onsets
    onset_file_path = os.path.join(base_directory, "Stimuli_Onsets", onsets_file)
    onsets_list = np.load(onset_file_path)

    if len(onsets_list) < min_trial_number:
        print("Minimum Trial Number Not Met For Condition: ", onsets_file, "In Session: ", base_directory)
    else:
        if len(onsets_list) > max_trial_num:
            onsets_list = onsets_list[0:max_trial_num]

        # Load Activity Matrix
        delta_f_file = os.path.join(base_directory, "Downsampled_Delta_F.h5")
        delta_f_container = tables.open_file(delta_f_file, "r")
        activity_matrix = delta_f_container.root.Data
        number_of_timepoints = np.shape(activity_matrix)[0]

        # Load Downsampled Behaviour Matrix
        downsampled_ai_matrix = np.load(os.path.join(base_directory, "Downsampled_AI_Matrix_Framewise.npy"))

        # Create Combined Stimulus Trace
        combined_stimulus_trace = create_combined_stimulus_trace(downsampled_ai_matrix, stop_stimuli)

        # Get Trial Start and Stops
        trial_start_stop_list = get_trial_stats_and_stops(onsets_list, number_of_timepoints, start_window, combined_stimulus_trace, trace_threshold, start_cutoff)

        # If Movement Correction True, Load Regression Dictionary
        if movement_correction == True:
            regression_dictionary = np.load(os.path.join(base_directory, "Regression_Coefs",  "Regression_Dicionary_Face.npy"), allow_pickle=True)[()]
        else:
            regression_dictionary = None

        #Load Face Motion Components
        face_motion_components = np.load(os.path.join(base_directory, "Mousecam_Analysis", "Matched_Transformed_Mousecam_Face_Data.npy"))

        # Get Ragged Tensors
        activity_tensor, behaviour_tensor = get_ragged_tensors(activity_matrix, downsampled_ai_matrix, selected_ai_channels, trial_start_stop_list, movement_correction, face_motion_components, regression_dict=regression_dictionary)

        # Reconstruct Into Local Brain Space
        activity_tensor = reconstruct_activity_tensor(activity_tensor, indicies, image_height, image_width, alignment_dictionary)
        #visualise_reconstructed_tensor(activity_tensor)

        # Apply Shared Tight Mask
        activity_tensor = apply_shared_tight_mask(activity_tensor)
        #visualise_tight_mask_tensor(activity_tensor)

        # Save Tensors
        session_tensor_directory = Trial_Aligned_Utils.check_save_directory(base_directory, tensor_save_directory)
        tensor_name = onsets_file.replace("_onsets.npy", "")
        activity_tensor_name = tensor_name + "_Extended_Activity_Tensor.npy"
        behaviour_tensor_name = tensor_name + "_Extended_Behaviour_Tensor.npy"

        session_activity_tensor_file = os.path.join(session_tensor_directory, activity_tensor_name)
        session_behaviour_tensor_file = os.path.join(session_tensor_directory, behaviour_tensor_name)

        np.save(session_activity_tensor_file, activity_tensor)
        np.save(session_behaviour_tensor_file, behaviour_tensor)

        # Close Delta F File
        delta_f_container.close()

