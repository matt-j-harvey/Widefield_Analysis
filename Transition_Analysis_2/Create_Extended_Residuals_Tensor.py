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
from sklearn.linear_model import LinearRegression, Ridge
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
from tqdm import tqdm

import Transition_Utils


def unpack_tensors(activity_tensor, behaviour_tensor):

    # Get Tensor Shapes
    number_of_trials, trial_length, number_of_pixels = np.shape(activity_tensor)
    number_of_behaviour_traces = np.shape(behaviour_tensor)[2]

    # Reshape Tensors
    activity_tensor = np.reshape(activity_tensor, (number_of_trials * trial_length, number_of_pixels))
    behaviour_tensor = np.reshape(behaviour_tensor, (number_of_trials * trial_length, number_of_behaviour_traces))
    print("Activity Tensor", np.shape(activity_tensor))
    print("Behaviour Tensor", np.shape(behaviour_tensor))

    # Remove NaNs
    timepoints_with_actual_data = []
    number_of_timepoints = np.shape(activity_tensor)[0]
    for timepoint_index in range(number_of_timepoints):

        nan_in_activity = np.isnan(activity_tensor[timepoint_index]).any()
        nan_in_behaviour = np.isnan(behaviour_tensor[timepoint_index]).any()
        print("Activity", nan_in_activity)
        print("Behaviour",  nan_in_behaviour)

        # If Ifs Not A Padded NaN
        if not nan_in_activity and not nan_in_behaviour:
            timepoints_with_actual_data.append(timepoint_index)

        elif nan_in_behaviour and nan_in_activity:
            print("Removed A NaN")

        elif bool(nan_in_activity) != bool(nan_in_behaviour):
            print("Error nan One Tesnor But Not The Other")

    activity_tensor = activity_tensor[timepoints_with_actual_data]
    behaviour_tensor = behaviour_tensor[timepoints_with_actual_data]

    return activity_tensor, behaviour_tensor, timepoints_with_actual_data, number_of_trials, trial_length


def get_trial_structure(tensor):

    trial_length_list = []
    number_of_trials = np.shape(tensor)[0]

    for trial in tensor:
        trial_length = np.shape(trial)[0]
        trial_length_list.append(trial_length)
    return number_of_trials, trial_length_list


def reshape_condition_data(condition_data, trial_length_list):

    reshaped_tensor = []
    trial_start = 0
    for trial_length in trial_length_list:
        trial_stop = trial_start + trial_length
        trial_data = condition_data[trial_start:trial_stop]
        reshaped_tensor.append(trial_data)
        trial_start += trial_length

    reshaped_tensor = np.array(reshaped_tensor)
    return reshaped_tensor


def pad_ragged_tensor_with_nans(ragged_tensor):

    # Get Longest Trial
    length_list = []
    for trial in ragged_tensor:
        trial_length, number_of_pixels = np.shape(trial)
        length_list.append(trial_length)

    print("Length list", length_list)
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


def get_residual_tensors(base_directory, onsets_file_list, extended_tensor_directory, model_name):
    print("")
    # Get Tensor Names
    activity_tensor_list = []
    behaviour_tensor_list = []

    condition_size_list = []
    trial_number_list = []
    trial_length_list = []

    for condition in onsets_file_list:

        # Get Tensor Filepaths
        mouse_tensor_root_directory = Transition_Utils.check_save_directory(base_directory, extended_tensor_directory)
        tensor_name = condition.replace("_onsets.npy", "")
        behaviour_tensor_name = tensor_name + "_Extended_Behaviour_Tensor.npy"
        activity_tensor_name = tensor_name + "_Extended_Activity_Tensor.npy"
        behaviour_tensor_file = os.path.join(mouse_tensor_root_directory, behaviour_tensor_name)
        activity_tensor_file = os.path.join(mouse_tensor_root_directory, activity_tensor_name)

        # Load Tensors
        behaviour_tensor = np.load(behaviour_tensor_file, allow_pickle=True)
        activity_tensor = np.load(activity_tensor_file, allow_pickle=True)
        print("Activity Tensor Shape", np.shape(activity_tensor))
        print("Behaviour Tensor Shape", np.shape(behaviour_tensor))

        # Get Tensor Structure
        number_of_trials, trial_lengths = get_trial_structure(behaviour_tensor)
        trial_number_list.append(number_of_trials)
        trial_length_list.append(trial_lengths)
        condition_size_list.append(np.sum(trial_lengths))

        # Concatenate Trials
        behaviour_tensor = np.vstack(behaviour_tensor)
        activity_tensor = np.vstack(activity_tensor)

        # Add To Lists
        activity_tensor_list.append(activity_tensor)
        behaviour_tensor_list.append(behaviour_tensor)

    # Create Combined Tensors
    activity_tensor_list = np.vstack(activity_tensor_list)
    behaviour_tensor_list = np.vstack(behaviour_tensor_list)
    print("Concatenated Activity Tensors Across Conditions", np.shape(activity_tensor_list))
    print(np.shape(behaviour_tensor_list))

    # Create Linear Model
    model = Ridge()

    # Fit Linear Model and Get Predictions
    model.fit(X=behaviour_tensor, y=activity_tensor)
    prediction = model.predict(X=behaviour_tensor_list)
    residuals = np.subtract(activity_tensor_list, prediction)


    # Reshape Residuals
    residual_tensor_list = []
    number_of_conditions = len(condition_size_list)
    condition_start = 0
    for condition_index in range(number_of_conditions):
        condition_size = condition_size_list[condition_index]
        condition_stop = condition_start + condition_size
        condition_data = residuals[condition_start:condition_stop]

        condition_data = reshape_condition_data(condition_data, trial_length_list[condition_index])
        print("Condition Data Shape", np.shape(condition_data))
        condition_data = pad_ragged_tensor_with_nans(condition_data)
        print("Condition Data Shape", np.shape(condition_data))

        condition_start += condition_size
        residual_tensor_list.append(condition_data)

    # Save Activity Tensor
    session_tensor_directory = Transition_Utils.check_save_directory(base_directory, extended_tensor_directory)
    for condition_index in range(number_of_conditions):
        tensor_name = onsets_file_list[condition_index].replace("_onsets.npy", "")
        session_tensor_file = os.path.join(session_tensor_directory, model_name + "_" + tensor_name + "_Extended_Residual_Tensor.npy")
        np.save(session_tensor_file, residual_tensor_list[condition_index])



# Load Session List
mouse_list = ["NXAK14.1A", "NRXN78.1D", "NXAK4.1B", "NXAK7.1B", "NXAK22.1A"]
session_type = "Transition"
session_list = []
for mouse_name in mouse_list:
    session_list = session_list + Transition_Utils.load_mouse_sessions(mouse_name, session_type)

# Load Analysis Details
"""
analysis_name = "Absence Of Expected Odour"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Transition_Utils.load_analysis_container(analysis_name)
"""

analysis_name = "Perfect_v_Imperfect_Switches"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Transition_Utils.load_analysis_container(analysis_name)

# Create Behaviour Tensors
extended_tensor_root_directory = "/media/matthew/Expansion/Widefield_Analysis/Extended_Tensors"
for base_directory in tqdm(session_list):
    get_residual_tensors(base_directory, onset_files, extended_tensor_root_directory, analysis_name)
