import numpy as np
import os

from Widefield_Utils import widefield_utils


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


def flatten_tensor(tensor):
    n_trial, trial_length, n_var = np.shape(tensor)
    tensor = np.reshape(tensor, (n_trial * trial_length, n_var))
    return tensor


def convert_tensor_list_to_matrix(tensor_list):
    flat_tensor_list = []
    for tensor in tensor_list:
        flat_tensor = flatten_tensor(tensor)
        flat_tensor_list.append(flat_tensor)

    data_matrix = np.vstack(flat_tensor_list)
    return data_matrix


def create_behaviour_regression_tensors(analysis_name, base_directory, early_cutoff):

    # Get Analysis Details
    [start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(base_directory, "Behaviour_Ridge_Regression", "Behaviour_Design_Matrix.npy"))

    # Load Delta F Matrix
    delta_f_matrix = np.load(os.path.join(base_directory, "Delta_F_Matrix_100_by_100_SVD.npy"))
    number_of_timepoints = np.shape(delta_f_matrix)[0]

    # Create Design Matricies
    behaviour_tensor_list = []
    delta_f_tensor_list = []

   # Iterate Through Conditions
    for condition in onset_files:

        # Load Onsets
        onsets = load_onsets(base_directory, condition, start_window, stop_window, number_of_timepoints, early_cutoff)

        # Get Behaviour Tensor
        behaviour_tensor = get_data_tensor(behaviour_matrix, onsets, start_window, stop_window)
        behaviour_tensor_list.append(behaviour_tensor)

        # Get Delta F Tensor
        delta_f_tensor = get_data_tensor(delta_f_matrix, onsets, start_window, stop_window)
        delta_f_tensor_list.append(delta_f_tensor)


    # Flatten Lists
    behaviour_design_matrix = convert_tensor_list_to_matrix(behaviour_tensor_list)
    delta_f_prediction_matrix = convert_tensor_list_to_matrix(delta_f_tensor_list)

    return behaviour_design_matrix, delta_f_prediction_matrix