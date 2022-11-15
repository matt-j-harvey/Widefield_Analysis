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
import sys
from sklearn.linear_model import LinearRegression
import h5py

import Single_Session_Analysis_Utils
import Create_Ridge_Design_Matrix


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def get_design_tensor(design_matrix, onsets, start_window, stop_window, start_cutoff=3000):

    number_of_timepoints = np.shape(design_matrix)[0]

    # Create Empty Tensor To Hold Data
    design_tensor = []

    number_of_trials = len(onsets)
    for trial_index in range(number_of_trials):

        # Get Trial Activity
        trial_onset = onsets[trial_index]
        trial_start = trial_onset + start_window
        trial_stop = trial_onset + stop_window

        if trial_start > start_cutoff and trial_stop < number_of_timepoints:
            trial_activity = design_matrix[trial_start:trial_stop]
            trial_activity = np.nan_to_num(trial_activity)

            design_tensor.append(trial_activity)

    design_tensor = np.array(design_tensor)
    return design_tensor


def baseline_correct_activity_tensor(activity_tensor, start_window):

    number_of_trials = np.shape(activity_tensor)[0]

    # Create Empty Tensor To Hold Data
    corrected_tensor = []

    # Get Correlation Matrix For Each Trial
    for trial_index in range(number_of_trials):
        trial_activity = activity_tensor[trial_index]
        trial_baseline = trial_activity[0:-start_window]
        trial_baseline = np.mean(trial_baseline, axis=0)
        trial_activity = np.subtract(trial_activity, trial_baseline)
        corrected_tensor.append(trial_activity)

    corrected_tensor = np.array(corrected_tensor)
    return corrected_tensor


def get_activity_tensor(activity_matrix, onsets, start_window, stop_window, baseline_correct, start_cutoff=3000):

    number_of_pixels = np.shape(activity_matrix)[1]
    number_of_trials = np.shape(onsets)[0]
    number_of_timepoints = np.shape(activity_matrix)[0]

    # Create Empty Tensor To Hold Data
    activity_tensor = []

    # Get Correlation Matrix For Each Trial
    for trial_index in range(number_of_trials):

        # Get Trial Activity
        trial_onset = onsets[trial_index]
        trial_start = trial_onset + start_window
        trial_stop = trial_onset + stop_window

        if trial_start > start_cutoff and trial_stop < number_of_timepoints:
            trial_activity = activity_matrix[trial_start:trial_stop]
            trial_activity = np.nan_to_num(trial_activity)

            # Baseline Correct
            if baseline_correct == True:
                trial_baseline = trial_activity[0:-start_window]
                trial_baseline = np.mean(trial_baseline, axis=0)
                trial_activity = np.subtract(trial_activity, trial_baseline)

            activity_tensor.append(trial_activity)

    activity_tensor = np.array(activity_tensor)
    return activity_tensor


def spatial_smooth_tensors(base_directory, activity_tensor):

    # Load Mask Details
    indicies, image_height, image_width = Single_Session_Analysis_Utils.load_downsampled_mask(base_directory)

    smoothed_tensor = []
    for trial in activity_tensor:
        smoothed_trial = []
        for frame in trial:
            frame = Single_Session_Analysis_Utils.create_image_from_data(frame, indicies, image_height, image_width)
            frame = ndimage.gaussian_filter(frame, sigma=1)
            frame = np.reshape(frame, (image_height * image_width))
            frame = frame[indicies]

            smoothed_trial.append(frame)
        smoothed_tensor.append(smoothed_trial)
    smoothed_tensor = np.array(smoothed_tensor)
    return smoothed_tensor



def reconstruct_activity_tensor(activity_tensor, indicies, image_height, image_width, within_mouse_alignment_dictionary, across_mouse_alignment_dictionary):

    reconstructed_tensor = []

    for trial in activity_tensor:
        reconstructed_trial = []

        for frame in trial:

            # Reconstruct Image
            frame = Single_Session_Analysis_Utils.create_image_from_data(frame, indicies, image_height, image_width)

            # Align Image Within Mouse
            frame = Single_Session_Analysis_Utils.transform_image(frame, within_mouse_alignment_dictionary)

            # Align Image Across Mice
            frame = Single_Session_Analysis_Utils.transform_image(frame, across_mouse_alignment_dictionary)


            reconstructed_trial.append(frame)
        reconstructed_tensor.append(reconstructed_trial)

    reconstructed_tensor = np.array(reconstructed_tensor)
    return reconstructed_tensor


def apply_shared_tight_mask(activity_tensor):

    # Load Tight Mask
    indicies, image_height, image_width = Single_Session_Analysis_Utils.load_tight_mask()

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


def get_predicited_activity(base_directory, design_tensor):

    # Load Regression Dict
    regression_dictionary = np.load(os.path.join(base_directory, "Regression_Coefs", "Regression_Dictionary_Simple.npy"), allow_pickle=True)[()]
    regression_intercepts = regression_dictionary["Intercepts"]

    regression_coefs = regression_dictionary["Coefs"]
    regression_coefs = np.transpose(regression_coefs)

    # Get Model Prediction
    predicited_tensor = []
    for trial in design_tensor:
        model_prediction = np.dot(trial, regression_coefs)
        model_prediction = np.add(model_prediction, regression_intercepts)
        predicited_tensor.append(model_prediction)

    predicited_tensor = np.array(predicited_tensor)
    return predicited_tensor


def get_corrected_tensor(activity_tensor, predicited_tensor):
    print("Activity Tensor Shape", np.shape(activity_tensor))
    print("Predicited Tensor Shape", np.shape(predicited_tensor))

    number_of_trials, number_of_timepoints, number_of_pixels = np.shape(activity_tensor)

    activity_tensor = np.reshape(activity_tensor, (number_of_trials * number_of_timepoints, number_of_pixels))
    predicited_tensor = np.reshape(predicited_tensor, (number_of_trials * number_of_timepoints, number_of_pixels))

    residual_tensor = np.subtract(activity_tensor, predicited_tensor)

    residual_tensor = np.reshape(residual_tensor, (number_of_trials, number_of_timepoints, number_of_pixels))

    return residual_tensor



def create_activity_tensor(base_directory, onsets_file, start_window, stop_window, tensor_save_directory, start_cutoff=3000, ridge_correction=True):

    # Load Mask
    indicies, image_height, image_width = Single_Session_Analysis_Utils.load_downsampled_mask(base_directory)

    # Load Within Mouse Alignment Dictionary
    within_mouse_alignment_dictionary = np.load(os.path.join(base_directory, "Within_Mouse_Alignment_Dictionary.npy"), allow_pickle=True)[()]

    # Load Across Mouse Alignmnet Dictionary
    root_directory = Single_Session_Analysis_Utils.get_root_directory(base_directory)
    across_mouse_alignment_dictionary = np.load(os.path.join(root_directory, "Across_Mouse_Alignment_Dictionary.npy"), allow_pickle=True)[()]

    # Load Onsets
    onset_file_path = os.path.join(base_directory, "Stimuli_Onsets", onsets_file)
    onsets_list = np.load(onset_file_path)

    # Load Activity Matrix
    delta_f_file = os.path.join(base_directory, "Downsampled_Delta_F.h5")
    delta_f_container = tables.open_file(delta_f_file, "r")
    activity_matrix = delta_f_container.root.Data

    # Get Activity Tensors
    activity_tensor = get_activity_tensor(activity_matrix, onsets_list, start_window, stop_window, start_cutoff=start_cutoff, baseline_correct=False)

    # Reconstruct Into Local Brain Space
    activity_tensor = reconstruct_activity_tensor(activity_tensor, indicies, image_height, image_width, within_mouse_alignment_dictionary, across_mouse_alignment_dictionary)

    # Apply Shared Tight Mask
    activity_tensor = apply_shared_tight_mask(activity_tensor)


    if ridge_correction == True:

        # Create Design Matrix
        design_matrix = Create_Ridge_Design_Matrix.create_ridge_design_matrix(base_directory)

        # Get Design Tensor
        design_tensor = get_design_tensor(design_matrix, onsets_list, start_window, stop_window, start_cutoff=start_cutoff)

        # Get Predicted Activity
        predicited_tensor = get_predicited_activity(base_directory, design_tensor)

        # Reconstruct Into Local Brain Space
        predicited_tensor = reconstruct_activity_tensor(predicited_tensor, indicies, image_height, image_width, within_mouse_alignment_dictionary, across_mouse_alignment_dictionary)

        # Apply Shared Tight Mask
        predicited_tensor = apply_shared_tight_mask(predicited_tensor)

        # Get Corrected Tensor
        residual_tensor = get_corrected_tensor(activity_tensor, predicited_tensor)

    # Baseline Correct Tensor
    activity_tensor = baseline_correct_activity_tensor(activity_tensor, start_window)

    # Save Tensor
    session_tensor_directory = Single_Session_Analysis_Utils.check_save_directory(base_directory, tensor_save_directory)
    tensor_name = onsets_file.replace("_onsets", "")
    tensor_name = tensor_name.replace(".npy", "")
    activity_tensor_name = tensor_name + "_Activity_Tensor.npy"
    session_activity_tensor_file = os.path.join(session_tensor_directory, activity_tensor_name)
    np.save(session_activity_tensor_file, activity_tensor)

    if ridge_correction == True:
        predicited_tensor_name = tensor_name + "_Predicited_Tensor.npy"
        residual_tensor_name = tensor_name + "_Residual_Tensor.npy"

        np.save(os.path.join(session_tensor_directory, predicited_tensor_name), predicited_tensor)
        np.save(os.path.join(session_tensor_directory, residual_tensor_name), residual_tensor)

    # Close Delta F File
    delta_f_container.close()

