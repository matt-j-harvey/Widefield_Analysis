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
from scipy.stats import zscore
import sys
from sklearn.linear_model import LinearRegression
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
from tqdm import tqdm
import pickle


# Custom Modules
from Widefield_Utils import widefield_utils
from Files import Session_List
#import Get_Extended_Tensor

def visualise_tensor(base_directory, activity_tensor):

    # View Tensors
    indicies, image_height, image_width = Trial_Aligned_Utils.load_downsampled_mask(base_directory)

    blue_black_cmap = Trial_Aligned_Utils.get_musall_cmap()
    mean_tensor = np.mean(activity_tensor, axis=0)
    plt.ion()
    for frame in mean_tensor:
        frame = Trial_Aligned_Utils.create_image_from_data(frame, indicies, image_height, image_width)
        plt.imshow(frame, cmap=blue_black_cmap, vmin=-0.05, vmax=0.05)
        plt.draw()
        plt.pause(0.1)
        plt.clf()



def apply_shared_tight_mask(activity_tensor):

    print("Activity Tensor Shape", np.shape(activity_tensor))

    # Load Tight Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()

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



def gaussian_filter_tensor(activity_tensor, indicies, image_height, image_width):

    filtered_tensor = []
    for trial in activity_tensor:
        reconstructed_trial = []

        for frame in trial:

            # Reconstruct Image
            frame = widefield_utils.create_image_from_data(frame, indicies, image_height, image_width)

            # Smooth
            frame = ndimage.gaussian_filter(frame, sigma=1)
            frame = np.reshape(frame, (image_height * image_width))
            frame = frame[indicies]

            reconstructed_trial.append(frame)
        filtered_tensor.append(reconstructed_trial)

    return filtered_tensor



def get_data_tensor(data_matrix, onsets, start_window, stop_window):

    # Create Empty Tensor To Hold Data
    data_tensor = []

    # Get Correlation Matrix For Each Trial
    number_of_trials = len(onsets)
    for trial_index in range(number_of_trials):

        # Get Trial Activity
        trial_onset = onsets[trial_index]
        trial_start = int(trial_onset + start_window)
        trial_stop = int(trial_onset + stop_window)

        trial_activity = data_matrix[trial_start:trial_stop]
        trial_activity = np.nan_to_num(trial_activity)
        data_tensor.append(trial_activity)

    return data_tensor


def baseline_correct_tensor(actvity_tensor, start_window):

    corrected_tensor = []
    for trial in actvity_tensor:
        trial = np.nan_to_num(trial)
        trial_baseline = trial[0:abs(start_window)]
        trial_baseline = np.mean(trial_baseline, axis=0)
        corrected_trial = np.subtract(trial, trial_baseline)
        corrected_tensor.append(corrected_trial)

    return corrected_tensor




def align_activity_tensor_within_mouse(activity_tensor, within_mouse_alignment_dictionary, indicies, image_height, image_width):
    aligned_tensor = []
    for trial in activity_tensor:
        aligned_trial = []
        for frame in trial:
            frame = widefield_utils.create_image_from_data(frame, indicies, image_height, image_width)
            frame = widefield_utils.transform_image(frame, within_mouse_alignment_dictionary)
            aligned_trial.append(frame)
        aligned_tensor.append(aligned_trial)

    return aligned_tensor


def align_activity_tensor_across_mice(activity_tensor, across_mouse_alignment_dictionary):
    aligned_tensor = []
    for trial in activity_tensor:
        aligned_trial = []
        for frame in trial:
            frame = widefield_utils.transform_image(frame, across_mouse_alignment_dictionary)
            aligned_trial.append(frame)
        aligned_tensor.append(aligned_trial)

    return aligned_tensor


def perform_ridge_regression_correction(activity_tensor, design_tensor, regression_dict):

    # Load Regression Coefs
    regression_coefs = regression_dict["Coefs"]
    regression_intercepts = regression_dict["Intercepts"]

    number_of_trials = np.shape(activity_tensor)[0]
    corrected_tensor = []
    for trial_index in range(number_of_trials):

        trial_raw_activity = activity_tensor[trial_index]
        trial_design_matrix = design_tensor[trial_index]

        trial_prediction = np.dot(trial_design_matrix, np.transpose(regression_coefs))
        trial_prediction = np.add(trial_prediction, regression_intercepts)

        trial_residual = np.subtract(trial_raw_activity, trial_prediction)
        corrected_tensor.append(trial_residual)

    return corrected_tensor



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




def load_onsets(base_directory, onsets_file, start_window, stop_window, number_of_timepoints, start_cutoff):

    onset_file_path = os.path.join(base_directory, "Stimuli_Onsets", onsets_file)
    raw_onsets_list = np.load(onset_file_path)

    checked_onset_list = []
    for trial_onset in raw_onsets_list:
        trial_start = trial_onset + start_window
        trial_stop = trial_onset + stop_window
        if trial_start > start_cutoff and trial_stop < number_of_timepoints:
            checked_onset_list.append(trial_onset)

    return checked_onset_list


def create_trial_tensor(base_directory, onsets_file, start_window, stop_window, tensor_save_directory,
                        start_cutoff=3000,
                        ridge_regression_correct=True,
                        align_within_mice=False,
                        align_across_mice=False,
                        baseline_correct=False,
                        gaussian_filter=False,
                        extended_tensor=False,
                        stop_stimuli=None,
                        mean_only=False,
                        use_100_df=False,
                        z_score=False,):

    """
    This Function Creates A Trial Tensor

    Steps
    1 Create Activity Tensor
    2 Regress Out Movement
    3 Gaussian Filter
    4 Baseline Correct
    5 Align Within Mouse
    6 Align Across Mice
    7 Get Behaviour Tensor
    """

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_downsampled_mask(base_directory)


    # Load Activity Matrix
    if use_100_df == False:
        delta_f_matrix_filepath = os.path.join(base_directory, "Downsampled_Delta_F.h5")
        delta_f_matrix_container = tables.open_file(delta_f_matrix_filepath, mode='r')
        activity_matrix = delta_f_matrix_container.root['Data']

    else:
        activity_matrix = np.load(os.path.join(base_directory, "Delta_F_Matrix_100_by_100_SVD.npy"))
        activity_matrix = np.nan_to_num(activity_matrix)
        indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

    number_of_timepoints, number_of_pixels = np.shape(activity_matrix)
    print("Activity Matrix Shape", np.shape(activity_matrix))

    # Load Onsets
    onsets_list = load_onsets(base_directory, onsets_file, start_window, stop_window, number_of_timepoints, start_cutoff)


    # Get Activity Tensors
    if extended_tensor == False:
        activity_tensor = get_data_tensor(activity_matrix, onsets_list, start_window, stop_window)
    else:
        activity_tensor = Get_Extended_Tensor.get_extended_tensor(base_directory, activity_matrix, onsets_list, start_window, stop_stimuli)


    # Movement Correct If Selected
    if ridge_regression_correct == True:

        if extended_tensor == True:
            print("Regression Not Currently Suppoted On Extended Tensors ")

        else:
            regression_dict = np.load(os.path.join(base_directory, "Behaviour_Regression_Trials", "Regression_Dictionary_Simple.npy"), allow_pickle=True)[()]
            design_matrix = np.load(os.path.join(base_directory, "Behaviour_Ridge_Regression", "Behaviour_Design_Matrix.npy"))
            print("Deisng Matrix Shape", np.shape(design_matrix))
            design_tensor = get_data_tensor(design_matrix, onsets_list, start_window, stop_window)
            activity_tensor = perform_ridge_regression_correction(activity_tensor, design_tensor, regression_dict)



    # Baseline Correct If Selected
    if baseline_correct == True:
        activity_tensor = baseline_correct_tensor(activity_tensor, start_window)

    # If Mean Only - Take Mean Now
    if mean_only == True:
        if extended_tensor == True:
            activity_tensor = pad_ragged_tensor_with_nans(activity_tensor)

        activity_tensor = np.nanmean(activity_tensor, axis=0)
        activity_tensor = np.array(activity_tensor)
        activity_tensor = np.expand_dims(a=activity_tensor, axis=0)


    # Gaussian Filter If Selected
    if gaussian_filter == True:
        activity_tensor = gaussian_filter_tensor(activity_tensor, indicies, image_height, image_width)

    # Align Within Mouse
    if align_within_mice == True:
        alignment_dictionary = np.load(os.path.join(base_directory, "Within_Mouse_Alignment_Dictionary.npy"), allow_pickle=True)[()]
        activity_tensor = align_activity_tensor_within_mouse(activity_tensor, alignment_dictionary, indicies, image_height, image_width)

    # Align Across Mice
    if align_across_mice == True:
        across_mouse_alignment_dictionary = widefield_utils.load_across_mice_alignment_dictionary(base_directory)
        activity_tensor = align_activity_tensor_across_mice(activity_tensor, across_mouse_alignment_dictionary)
        activity_tensor = apply_shared_tight_mask(activity_tensor)

    # Convert Tensor To Array
    if extended_tensor == False:
        activity_tensor = np.array(activity_tensor)
    else:
        activity_tensor = pad_ragged_tensor_with_nans(activity_tensor)


    # Create Activity Tensor Dict
    trial_tensor_dictionary = {
        "activity_tensor":activity_tensor,
        "regression_correction":ridge_regression_correct,
        "start_cutoff": start_cutoff,
        "align_within_mice": align_within_mice,
        "align_across_mice": align_across_mice,
        "baseline_correct": baseline_correct,
        "gaussian_filter": gaussian_filter,
        #"behaviour_tensor":behaviour_tensor,
    }

    # Save Trial Tensor
    print("Base directory", base_directory)
    print("Tensor Save Directory", tensor_save_directory)
    session_tensor_directory = widefield_utils.check_save_directory(base_directory, tensor_save_directory)
    tensor_name = onsets_file.replace("_onsets.npy", "")
    tensor_name = tensor_name.replace("_onset_frames.npy", "")
    tensor_file = os.path.join(session_tensor_directory, tensor_name)
    print("Tensor file", tensor_file)

    with open(tensor_file + ".pickle", 'wb') as handle:
        pickle.dump(trial_tensor_dictionary, handle, protocol=4)

    # Close Delta F File
    if use_100_df == False:
        delta_f_matrix_container.close()

