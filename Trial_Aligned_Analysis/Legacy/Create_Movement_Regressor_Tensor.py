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

from Files import Session_List
from Widefield_Utils import widefield_utils



def get_movement_tensor(activity_matrix, onsets, start_window, stop_window, start_cutoff=3000):

    number_of_trials = np.shape(onsets)[0]
    number_of_timepoints = np.shape(activity_matrix)[0]

    # Create Empty Tensor To Hold Data
    movement_tensor = []

    # Get Correlation Matrix For Each Trial
    for trial_index in range(number_of_trials):

        # Get Trial Activity
        trial_onset = onsets[trial_index]
        trial_start = trial_onset + start_window
        trial_stop = trial_onset + stop_window

        if trial_start > start_cutoff and trial_stop < number_of_timepoints:
            trial_activity = activity_matrix[trial_start:trial_stop]
            trial_activity = np.nan_to_num(trial_activity)

            movement_tensor.append(trial_activity)

    movement_tensor = np.array(movement_tensor)
    return movement_tensor




def create_movement_tensor(base_directory, onsets_file, start_window, stop_window, tensor_save_directory, start_cutoff=3000):

    # Load Onsets
    onset_file_path = os.path.join(base_directory, "Stimuli_Onsets", onsets_file)
    onsets_list = np.load(onset_file_path)

    # Load Design Matrix
    design_matrix_file = os.path.join(base_directory, "Ride_Regression", "Design_Matrix.npy")
    design_matrix = np.load(design_matrix_file)
    print("Design Matrix Shape", np.shape(design_matrix))

    # Get Activity Tensors
    movement_tensor = get_movement_tensor(design_matrix, onsets_list, start_window, stop_window, start_cutoff=start_cutoff)

    # Save Tensor
    session_tensor_directory = widefield_utils.check_save_directory(base_directory, tensor_save_directory)
    tensor_name = onsets_file.replace("_onsets.npy", "")
    tensor_name = tensor_name.replace("_onset_frames.npy", "")
    activity_tensor_name = tensor_name + "_Movement_Tensor.npy"

    session_activity_tensor_file = os.path.join(session_tensor_directory, activity_tensor_name)
    np.save(session_activity_tensor_file, movement_tensor)








