import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import tables
from tqdm import tqdm
import joblib
from datetime import datetime
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import explained_variance_score



def create_lagged_matrix(matrix, n_lags=14):
    """
    :param matrix: Matrix of shape (n_samples, n_dimensionns)
    :param n_lags: Number Of steps to include lagged versions of the matrix
    :return: Matrix with duplicated shifted version of origional matrix with shape (n_samples, n_dimensions * n_lags)
    """

    lagged_combined_matrix = []
    for lag_index in range(n_lags):
        lagged_matrix = np.copy(matrix)
        lagged_matrix = np.roll(a=lagged_matrix, axis=1, shift=lag_index)
        lagged_matrix[0:lag_index] = 0
        lagged_combined_matrix.append(lagged_matrix)

    lagged_combined_matrix = np.hstack(lagged_combined_matrix)

    return lagged_combined_matrix


def create_event_kernel_from_event_list(event_list, number_of_widefield_frames, preceeding_window=-14, following_window=28):
    kernel_size = following_window - preceeding_window
    design_matrix = np.zeros((number_of_widefield_frames, kernel_size))

    for timepoint_index in range(number_of_widefield_frames):

        if event_list[timepoint_index] == 1:

            # Get Start and Stop Times Of Kernel
            start_time = timepoint_index + preceeding_window
            stop_time = timepoint_index + following_window

            # Ensure Start and Stop Times Dont Fall Below Zero Or Above Number Of Frames
            start_time = np.max([0, start_time])
            stop_time = np.min([number_of_widefield_frames - 1, stop_time])

            # Fill In Design Matrix
            number_of_regressor_timepoints = stop_time - start_time
            for regressor_index in range(number_of_regressor_timepoints):
                design_matrix[start_time + regressor_index, regressor_index] = 1

    return design_matrix

def create_stimuli_dictionary():

    channel_index_dictionary = {
        "Photodiode"        :0,
        "Reward"            :1,
        "Lick"              :2,
        "Visual 1"          :3,
        "Visual 2"          :4,
        "Odour 1"           :5,
        "Odour 2"           :6,
        "Irrelevance"       :7,
        "Running"           :8,
        "Trial End"         :9,
        "Camera Trigger"    :10,
        "Camera Frames"     :11,
        "LED 1"             :12,
        "LED 2"             :13,
        "Mousecam"          :14,
        "Optogenetics"      :15,
        }

    return channel_index_dictionary



def get_common_length(design_matrix):

    timepoints_list = []

    # Add Number Of Timepoints For Each Regressor
    for regresor in design_matrix:
        number_of_timepoints = np.shape(regresor)[0]
        timepoints_list.append(number_of_timepoints)

    # Get Smallest Number Of Timepoints
    common_length = np.min(timepoints_list)

    # Trim Design Matrix To This Length
    trimmed_design_matrix = []
    for regressor in design_matrix:
        trimmed_design_matrix.append(regressor[:common_length])

    return trimmed_design_matrix

def create_ridge_design_matrix(base_directory):

    # Load Downsampled AI
    downsampled_ai_file = os.path.join(base_directory, "Downsampled_AI_Matrix_Framewise.npy")
    downsampled_ai_matrix = np.load(downsampled_ai_file)
    number_of_widefield_frames = np.shape(downsampled_ai_matrix)[1]

    # Create Stimuli Dictionary
    stimuli_dictionary = create_stimuli_dictionary()

    # Extract Lick and Running Traces
    lick_trace = downsampled_ai_matrix[stimuli_dictionary["Lick"]]
    running_trace = downsampled_ai_matrix[stimuli_dictionary["Running"]]

    # Subtract Traces So When Mouse Not Running Or licking They Equal 0
    running_baseline = np.load(os.path.join(base_directory, "Running_Baseline.npy"))
    running_trace = np.subtract(running_trace, running_baseline)
    running_trace = np.clip(running_trace, a_min=0, a_max=None)
    running_trace = np.expand_dims(running_trace, 1)

    lick_baseline = np.load(os.path.join(base_directory, "Lick_Baseline.npy"))
    lick_trace = np.subtract(lick_trace, lick_baseline)
    lick_trace = np.clip(lick_trace, a_min=0, a_max=None)
    lick_trace = np.expand_dims(lick_trace, 1)

    # Get Lagged Lick and Running Traces
    lick_trace = create_lagged_matrix(lick_trace)
    running_trace = create_lagged_matrix(running_trace)

    # Create Lick Kernel
    lick_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Lick_Events.npy"))
    lick_event_kernel = create_event_kernel_from_event_list(lick_onsets, number_of_widefield_frames, preceeding_window=-5, following_window=14)
    lick_regressors = np.hstack([lick_trace, lick_event_kernel])

    # Create Running Kernel
    running_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Running_Events.npy"))
    running_event_kernel = create_event_kernel_from_event_list(running_onsets, number_of_widefield_frames, preceeding_window=-14, following_window=28)
    running_regressors = np.hstack([running_trace, running_event_kernel])

    # Load Limb Movements
    limb_movements = np.load(os.path.join(base_directory, "Mousecam_Analysis", "Matched_Limb_Movements_Simple.npy"))

    # Load Whisker Pad Motion
    whisker_pad_motion_components = np.load(os.path.join(base_directory, "Mousecam_Analysis", "matched_whisker_data.npy"))

    # Load Face Motion Data
    face_motion_components = np.load(os.path.join(base_directory, "Mousecam_Analysis", "matched_face_data.npy"))

    # Get Lagged Versions
    whisker_pad_motion_components = create_lagged_matrix(whisker_pad_motion_components)
    limb_movements = create_lagged_matrix(limb_movements)
    face_motion_components = create_lagged_matrix(face_motion_components)

    design_matrix = [
        lick_regressors,
        running_regressors,
        face_motion_components,
        # eye_movement_event_kernel,
        # blink_event_kernel,
        whisker_pad_motion_components,
        limb_movements
    ]

    design_matrix = get_common_length(design_matrix)
    design_matrix = np.hstack(design_matrix)
    design_matrix = np.nan_to_num(design_matrix)

    return design_matrix
