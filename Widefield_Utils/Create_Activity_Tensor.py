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

from Widefield_Utils import widefield_utils


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
    print("Indicies", indicies, "Image height", image_height, "image width", image_width)

    transformed_tensor = []
    for trial in activity_tensor:
        transformed_trial = []

        for frame in trial:
            frame = np.ndarray.flatten(frame)
            frame = frame[indicies]
            transformed_trial.append(frame)

            """
            template = np.zeros(image_height * image_width)
            template[indicies] = frame
            template = np.reshape(template, (image_height, image_width))
            plt.imshow(template)
            plt.show()
            """
        transformed_tensor.append(transformed_trial)

    transformed_tensor = np.array(transformed_tensor)
    return transformed_tensor



def reconstruct_activity_tensor(activity_tensor, indicies, image_height, image_width, gaussian_filter, align=False, alignment_dictionary=None):

    reconstructed_tensor = []

    for trial in activity_tensor:
        reconstructed_trial = []

        for frame in trial:

            # Reconstruct Image
            frame = widefield_utils.create_image_from_data(frame, indicies, image_height, image_width)

            # Smooth If True
            if gaussian_filter == True:
                frame = ndimage.gaussian_filter(frame, sigma=1)

            # Align Image
            if align == True:
                frame = widefield_utils.transform_image(frame, alignment_dictionary)

            reconstructed_trial.append(frame)
        reconstructed_tensor.append(reconstructed_trial)

    reconstructed_tensor = np.array(reconstructed_tensor)
    return reconstructed_tensor



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


def align_activity_tensor_across_mice(activity_tensor, base_directory):

    # Load Across Mouse Alignment Dictionary
    across_mouse_alignment_dictionary = widefield_utils.load_across_mice_alignment_dictionary(base_directory)

    aligned_tensor = []
    for trial in activity_tensor:
        aligned_trial = []
        for frame in trial:
            frame = widefield_utils.transform_image(frame, across_mouse_alignment_dictionary)

            """
            plt.title("Aligned Across Mice")
            plt.imshow(frame)
            plt.show()
            """

            aligned_trial.append(frame)
        aligned_tensor.append(aligned_trial)

    aligned_tensor = np.array(aligned_tensor)
    return aligned_tensor




def create_activity_tensor(base_directory, onsets_file, start_window, stop_window, tensor_save_directory,
                           start_cutoff=3000,
                           align_within_mice=False,
                           align_across_mice=False,
                           baseline_correct=True,
                           gaussian_filter=False):

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_downsampled_mask(base_directory)

    # Load Alignment Dictionary
    if align_within_mice == True:
        alignment_dictionary = np.load(os.path.join(base_directory, "Within_Mouse_Alignment_Dictionary.npy"), allow_pickle=True)[()]

    # Load Onsets
    onset_file_path = os.path.join(base_directory, "Stimuli_Onsets", onsets_file)
    onsets_list = np.load(onset_file_path)

    # Load Activity Matrix
    delta_f_file = os.path.join(base_directory, "Downsampled_Delta_F.h5")
    delta_f_container = tables.open_file(delta_f_file, "r")
    activity_matrix = delta_f_container.root.Data

    # Get Activity Tensors
    activity_tensor = get_activity_tensor(activity_matrix, onsets_list, start_window, stop_window, start_cutoff=start_cutoff, baseline_correct=baseline_correct)

    # Reconstruct Into Local Brain Space
    if align_within_mice == True:
        activity_tensor = reconstruct_activity_tensor(activity_tensor, indicies, image_height, image_width, gaussian_filter, align=True, alignment_dictionary=alignment_dictionary)

    if align_across_mice == True:
        activity_tensor = align_activity_tensor_across_mice(activity_tensor, base_directory)
        activity_tensor = apply_shared_tight_mask(activity_tensor)

    # Save Tensor
    session_tensor_directory = widefield_utils.check_save_directory(base_directory, tensor_save_directory)
    tensor_name = onsets_file.replace("_onsets.npy", "")
    tensor_name = tensor_name.replace("_onset_frames.npy", "")

    if align_across_mice == True:
        activity_tensor_name = tensor_name + "_Activity_Tensor_Aligned_Across_Mice.npy"

    elif align_within_mice == True:
        activity_tensor_name = tensor_name + "_Activity_Tensor_Aligned_Within_Mouse.npy"

    else:
        activity_tensor_name = tensor_name + "_Activity_Tensor_Unaligned.npy"

    session_activity_tensor_file = os.path.join(session_tensor_directory, activity_tensor_name)
    np.save(session_activity_tensor_file, activity_tensor)

    # Close Delta F File
    delta_f_container.close()

