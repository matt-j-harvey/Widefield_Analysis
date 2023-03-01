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



def get_behaviour_tensor(behaviour_matrix, onsets, start_window, stop_window, selected_ai_channels, start_cutoff=3000):

    # Get Selected Traces
    channel_index_dictionary = Trial_Aligned_Utils.create_stimuli_dictionary()
    selected_channels_list = []
    for behaviour_trace in selected_ai_channels:
        selected_channels_list.append(channel_index_dictionary[behaviour_trace])

    # Get Data Structure
    behaviour_matrix = np.transpose(behaviour_matrix)
    behaviour_matrix = behaviour_matrix[:, selected_channels_list]
    number_of_timepoints, number_of_behaviour_traces = np.shape(behaviour_matrix)
    number_of_trials = np.shape(onsets)[0]

    # Create Empty Tensor To Hold Data
    behaviour_tensor = []

    # Get Correlation Matrix For Each Trial
    for trial_index in range(number_of_trials):

        # Get Trial Activity
        trial_start = onsets[trial_index] + start_window
        trial_stop = onsets[trial_index] + stop_window

        if trial_start > start_cutoff and trial_stop < number_of_timepoints:
            trial_behaviour = behaviour_matrix[trial_start:trial_stop]
            behaviour_tensor.append(trial_behaviour)

    behaviour_tensor = np.array(behaviour_tensor)
    print("Behaviour Tensor", np.shape(behaviour_tensor))

    return behaviour_tensor


def create_behaviour_tensor(base_directory, onsets_file, start_window, stop_window, selected_traces):

    # Load Onsets
    onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", onsets_file))

    # Load Behaviour Matrix
    downsampled_ai_matrix = np.load(os.path.join(base_directory, "Downsampled_AI_Matrix_Framewise.npy"))
    print("Downsampled AI Matrix Shape", np.shape(downsampled_ai_matrix))

    # Create Behaviour Tensor
    behaviour_tensor = get_behaviour_tensor(downsampled_ai_matrix, onsets, start_window, stop_window, selected_traces)

    """"
    # Save Behaviour Tensor
    tensor_name = onsets_file.replace("_onsets.npy", "")
    tensor_name = tensor_name + "_Behaviour_Tensor.npy"
    session_tensor_directory = Trial_Aligned_Utils.check_save_directory(base_directory, tensor_save_directory)
    session_tensor_file = os.path.join(session_tensor_directory, tensor_name)
    np.save(session_tensor_file, behaviour_tensor)
    """
    return behaviour_tensor
