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
from pathlib import Path

import Learning_Utils
import Session_List



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





def correct_baseline(activity_tensor, trial_start):

    corrected_tensor = []
    for trial in activity_tensor:
        trial_baseline = trial[0: -1 * trial_start]
        trial_baseline = np.mean(trial_baseline, axis=0)
        trial = np.subtract(trial, trial_baseline)
        corrected_tensor.append(trial)
    corrected_tensor = np.array(corrected_tensor)

    return corrected_tensor



def reconstruct_activity_tensor(activity_tensor, indicies, image_height, image_width, alignment_dictionary):

    reconstructed_tensor = []

    for trial in activity_tensor:
        reconstructed_trial = []

        for frame in trial:

            # Reconstruct Image
            frame = RT_Strat_Utils.create_image_from_data(frame, indicies, image_height, image_width)

            # Align Image
            frame = RT_Strat_Utils.transform_image(frame, alignment_dictionary)

            reconstructed_trial.append(frame)
        reconstructed_tensor.append(reconstructed_trial)

    reconstructed_tensor = np.array(reconstructed_tensor)
    return reconstructed_tensor


def apply_shared_tight_mask(activity_tensor):

    # Load Tight Mask
    indicies, image_height, image_width = RT_Strat_Utils.load_tight_mask()

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
    indicies, image_height, image_width = RT_Strat_Utils.load_generous_mask(base_directory)

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


# Load Session Tuples
control_tuples = Session_List.control_session_tuples
mutant_tuples = Session_List.mutant_session_tuples

# Load Analysis Details
analysis_name = "Hits_Pre_Post_Learning"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = RT_Strat_Utils.load_analysis_container(analysis_name)

# Create Activity Tensors
save_file = os.path.join(r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits")

for tuple in tqdm(session_tuples):

        # Get Mouse Name
        mouse_name = Path(tuple[0]).parts[-2]
        print("Mouse: ", mouse_name)

        # Check Mouse Save Directory Exists
        mouse_save_directory = os.path.join(save_file, mouse_name)
        if not os.path.exists(mouse_save_directory):
            os.mkdir(mouse_save_directory)

        for base_directory in tuple:
            session_name = Path(base_directory).parts[-1]
            tensor_save_directory = os.path.join(mouse_save_directory, session_name)
            if not os.path.exists(tensor_save_directory):
                os.mkdir(tensor_save_directory)

            activity_tensor = create_standard_alignment_tensor(base_directory, onset_files[0], start_window, stop_window)
            np.save(os.path.join(tensor_save_directory, "Hits_RT_Matched_Activity_Tensor.npy"), activity_tensor)

