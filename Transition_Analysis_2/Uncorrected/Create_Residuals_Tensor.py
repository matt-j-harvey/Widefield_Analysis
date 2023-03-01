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


def get_residual_tensors(base_directory, onsets_file_list, selected_ai_races, behaviour_tensor_root_directory, activity_tensor_root_directory, residual_root_directory, model_name):

    # Get Tensor Names
    activity_tensor_list = []
    behaviour_tensor_list = []

    condition_size_list = []
    trial_number_list = []
    trial_length_list = []

    for condition in onsets_file_list:
        tensor_name = condition.replace("_onsets.npy", "")
        behaviour_tensor_name = tensor_name + "_Behaviour_Tensor.npy"
        activity_tensor_name = tensor_name + "_Activity_Tensor.npy"

        condition_behaviour_tensor_root_directory = Transition_Utils.check_save_directory(base_directory, behaviour_tensor_root_directory)
        condition_activity_tensor_root_directory = Transition_Utils.check_save_directory(base_directory, activity_tensor_root_directory)
        behaviour_tensor_file = os.path.join(condition_behaviour_tensor_root_directory, behaviour_tensor_name)
        activity_tensor_file = os.path.join(condition_activity_tensor_root_directory, activity_tensor_name)

        behaviour_tensor = np.load(behaviour_tensor_file)
        activity_tensor = np.load(activity_tensor_file)

        # Reshape Tensors
        number_of_trials, trial_length, number_of_pixels = np.shape(activity_tensor)
        number_of_behaviour_traces = np.shape(behaviour_tensor)[2]
        condition_size_list.append(number_of_trials * trial_length)
        trial_number_list.append(number_of_trials)
        trial_length_list.append(trial_length)

        activity_tensor = np.reshape(activity_tensor, (number_of_trials * trial_length, number_of_pixels))
        behaviour_tensor = np.reshape(behaviour_tensor, (number_of_trials * trial_length, number_of_behaviour_traces))
        print("Behaviour tensor", np.shape(behaviour_tensor))
        print("Activity Tensor", np.shape(activity_tensor))

        # Add To Lists
        activity_tensor_list.append(activity_tensor)
        behaviour_tensor_list.append(behaviour_tensor)

    # Create Combined Tensors
    activity_tensor_list = np.vstack(activity_tensor_list)
    behaviour_tensor_list = np.vstack(behaviour_tensor_list)

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

        condition_data = np.reshape(condition_data, (trial_number_list[condition_index], trial_length_list[condition_index], number_of_pixels))

        condition_start += condition_size
        residual_tensor_list.append(condition_data)

    # Save Activity Tensor
    session_tensor_directory = Transition_Utils.check_save_directory(base_directory, residual_root_directory)
    for condition_index in range(number_of_conditions):
        tensor_name = onsets_file_list[condition_index].replace("_onsets.npy", "")
        session_tensor_file = os.path.join(session_tensor_directory, model_name + "_" + tensor_name + "_Residual_Tensor.npy")
        np.save(session_tensor_file, residual_tensor_list[condition_index])




# Load Session List
mouse_list = ["NRXN78.1D", "NXAK4.1B", "NXAK7.1B", "NXAK14.1A", "NXAK22.1A"]
session_type = "Transition"
session_list = []
for mouse_name in mouse_list:
    session_list = session_list + Transition_Utils.load_mouse_sessions(mouse_name, session_type)

# Load Analysis Details
analysis_name = "Absence Of Expected Odour"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Transition_Utils.load_analysis_container(analysis_name)

# Create Behaviour Tensors
selected_ai_traces = ["Running", "Lick"]
behaviour_tensor_root_directory = "/media/matthew/Expansion/Widefield_Analysis/Behaviour_Tensors"
activity_tensor_root_directory = "/media/matthew/Expansion/Widefield_Analysis/Activity_Tensors"
residual_root_directory = "/media/matthew/Expansion/Widefield_Analysis/Model_Residuals"
for base_directory in tqdm(session_list):
    get_residual_tensors(base_directory, onset_files, selected_ai_traces, behaviour_tensor_root_directory, activity_tensor_root_directory, residual_root_directory, analysis_name)
