import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm

from Widefield_Utils import widefield_utils
from Trial_Aligned_Analysis import Create_Trial_Tensors
import Create_Behaviour_Design_Matrix

def get_length_of_longest_stim(binary_list):
    max_size = 0
    prev = None

    for i in binary_list:

        if i == 1 and i == prev:
            size += 1
            if size > max_size:
                max_size = size
        else:
            size = 0

        prev = i

    return max_size



def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)



def create_odour_design_matricies_for_all_sessions(selected_session_list, data_root_directory):

    # Create Stimuli Dict
    stimuli_dictionary = widefield_utils.create_stimuli_dictionary()

    for mouse in tqdm(selected_session_list, leave=True, position=0, desc="Mouse"):
        for base_directory in tqdm(mouse, leave=True, position=1, desc="Session"):
            full_base_directory = os.path.join(data_root_directory, base_directory)

            # Load Downsampled AI Matrix
            downsampled_ai_matrix = np.load(os.path.join(full_base_directory, "Downsampled_AI_Matrix_Framewise.npy"))

            # Extract Odour Traces
            odour_1_trace = downsampled_ai_matrix[stimuli_dictionary["Odour 1"]]
            odour_2_trace = downsampled_ai_matrix[stimuli_dictionary["Odour 2"]]

            # Create Regressor Matricies
            odour_1_regressor = create_stimuli_regressor(odour_1_trace)
            odour_2_regressor = create_stimuli_regressor(odour_2_trace)
            combined_odour_regressor = np.hstack([odour_1_regressor, odour_2_regressor])

            # Save This
            np.save(os.path.join(full_base_directory, "Odour_Design_Matrix.npy"), combined_odour_regressor)




def create_regressor_matrix(number_of_timepoints, longest_stimuli, stimuli_trace):

    regressor_matrix = np.zeros((number_of_timepoints, longest_stimuli+1))

    current_depth = 0
    for timepoint_index in range(number_of_timepoints):

        if stimuli_trace[timepoint_index] == 1:
            regressor_matrix[timepoint_index, current_depth] = 1
            current_depth += 1

        else:
            current_depth = 0

    return regressor_matrix



def create_stimuli_regressor(regressor_trace, threshold=3):

    # Binarise
    regressor_trace = np.where(regressor_trace > threshold, 1, 0)

    # Get Longest Continuous Stimuli
    longest_stimuli = get_length_of_longest_stim(regressor_trace)

    # Populate Full Regressor
    number_of_timepoints = len(regressor_trace)
    regressor_matrix = create_regressor_matrix(number_of_timepoints, longest_stimuli, regressor_trace)

    return regressor_matrix


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
        trial_start = trial_onset + start_window
        trial_stop = trial_onset + stop_window

        trial_activity = data_matrix[trial_start:trial_stop]
        trial_activity = np.nan_to_num(trial_activity)
        data_tensor.append(trial_activity)

    return data_tensor




def create_behaviour_tensors(base_directory, data_root_directory, tensor_root_directory, onset_file_list, start_window, stop_window, start_cutoff):


    full_base_directory = os.path.join(data_root_directory, base_directory)

    # Load Design Matrix
    design_matrix = np.load(os.path.join(full_base_directory, "Behaviour_Ridge_Regression", "Behaviour_Design_Matrix.npy"))
    number_of_timepoints = np.shape(design_matrix)[0]

    for onsets_file in onset_file_list:

        # Load Onsets
        onset_list = load_onsets(full_base_directory, onsets_file, start_window, stop_window, number_of_timepoints, start_cutoff)

        # Get Design tensor
        design_tensor = get_data_tensor(design_matrix, onset_list, start_window, stop_window)

        # Get Tensor Name
        tensor_name = onsets_file.replace("_onsets.npy", "")
        tensor_name = tensor_name.replace("_onset_frames.npy", "")
        tensor_file = os.path.join(tensor_root_directory, base_directory, tensor_name)

        # Save Tensor
        np.save(os.path.join(tensor_file + "_behaviour_design_tensor.npy"), design_tensor)


def check_tensor_directory(tensor_root_directory, base_directory):
    full_filepath = os.path.join(tensor_root_directory, base_directory)
    if not os.path.exists(full_filepath):
        os.makedirs(full_filepath)



def create_condition_regressor(number_of_trials, trial_length):
    condition_regressor = []
    for trial_index in range(number_of_trials):
        trial_regressor = np.eye(trial_length)
        condition_regressor.append(trial_regressor)
    condition_regressor = np.vstack(condition_regressor)
    return condition_regressor


def flatten_tensor(tensor):
    n_trial, trial_length, n_var = np.shape(tensor)
    tensor = np.reshape(tensor, (n_trial * trial_length, n_var))
    return tensor


def combine_condition_tenors(condition_regressor_list):
    n_conditions = len(condition_regressor_list)
    condition_length = np.shape(condition_regressor_list[0])[1]

    # Get Full Size
    regressor_size_list = []
    for condition_regressor in condition_regressor_list:
        condition_timepoints = np.shape(condition_regressor)[0]
        regressor_size_list.append(condition_timepoints)

    # Create Empty Matrix
    total_timepoints = np.sum(regressor_size_list)
    combined_matrix = np.zeros((total_timepoints, n_conditions * condition_length))

    # Populate Matrix
    current_position = 0
    for condition_index in range(n_conditions):
        condition_start = current_position
        condition_stop = condition_start + regressor_size_list[condition_index]

        y_start = condition_index*condition_length
        y_stop = (condition_index+1) * condition_length

        combined_matrix[condition_start:condition_stop, y_start:y_stop] = condition_regressor_list[condition_index]

        current_position = condition_stop

    return combined_matrix


def create_stimuli_design_dictionary(onset_names, start_window, stop_window):

    # Create Deisgn Matrix Dict
    regressor_group_sizes = []
    regressor_group_starts = []
    regressor_group_stops = []
    regressor_names = []

    stimuli_length = stop_window - start_window
    number_of_stimuli = len(onset_names)


    timewindow_list = list(range(start_window, stop_window))
    timewindow_list = np.multiply(timewindow_list, 36)

    for stimuli_index in range(number_of_stimuli):
        group_start = stimuli_index * stimuli_length
        group_stop = group_start + stimuli_length

        regressor_group_sizes.append(stimuli_length)
        regressor_group_starts.append(group_start)
        regressor_group_stops.append(group_stop)

        # Get Stimuli name
        stimuli_name = onset_names[stimuli_index]
        stimuli_name = stimuli_name.replace("_onsets.npy", "")
        stimuli_name = stimuli_name.replace("_onset_frames.npy", "")

        for timepoint in range(stimuli_length):
            regressor_names.append(stimuli_name + "_" + str(timewindow_list[timepoint]) + "ms")

    stimuli_matrix_dict = {
    "number_of_regressor_groups": number_of_stimuli,
    "coef_group_sizes": regressor_group_sizes,
    "coef_group_starts": regressor_group_starts,
    "coef_group_stops": regressor_group_stops,
    "coefs_names": regressor_names,
    }

    return stimuli_matrix_dict

def scale_continous_regressors(regressor_matrix):

    # Subtract Mean
    regressor_mean = np.mean(regressor_matrix, axis=0)
    regressor_sd = np.std(regressor_matrix, axis=0)

    # Devide By 2x SD
    regressor_matrix = np.subtract(regressor_matrix, regressor_mean)
    regressor_matrix = np.divide(regressor_matrix, 2 * regressor_sd)

    regressor_matrix = np.nan_to_num(regressor_matrix)
    #plt.hist(np.ndarray.flatten(regressor_matrix), bins=100)
    #plt.show()
    return regressor_matrix




def create_combined_design_matrix(base_directory, data_root_directory, tensor_root_directory, onset_file_list, start_window, stop_window):

    full_model_behaviour_matrix = []
    condition_regressor_list = []

    full_tensor_directory = os.path.join(tensor_root_directory, base_directory)
    for onsets_file in onset_file_list:

        # Get Tensor Name
        tensor_name = onsets_file.replace("_onsets.npy", "")
        tensor_name = tensor_name.replace("_onset_frames.npy", "")

        # Load Behaviour Design Matrix Tensor
        behaviour_tensor = np.load(os.path.join(full_tensor_directory, tensor_name + "_behaviour_design_tensor.npy"))

        # Create Stimuli Regressor
        number_of_trials = np.shape(behaviour_tensor)[0]
        trial_length = np.shape(behaviour_tensor)[1]
        condition_regressor = create_condition_regressor(number_of_trials, trial_length)

        # Reshape These
        behaviour_tensor = flatten_tensor(behaviour_tensor)

        # Combine These
        full_model_behaviour_matrix.append(behaviour_tensor)
        condition_regressor_list.append(condition_regressor)

    # Combine All Conditions
    full_model_behaviour_matrix = np.vstack(full_model_behaviour_matrix)
    full_model_stimuli_matrix = combine_condition_tenors(condition_regressor_list)


    # Scale Behavioural Regressors
    full_model_behaviour_matrix = scale_continous_regressors(full_model_behaviour_matrix)
    full_design_matrix = np.hstack([full_model_stimuli_matrix, full_model_behaviour_matrix])

    # Save This Matrix
    np.save(os.path.join(full_tensor_directory, "Full_Model_Design_Matrix.npy"), full_design_matrix)

    # Create Dictionaries
    stimuli_design_matrix_dict = create_stimuli_design_dictionary(onset_file_list, start_window, stop_window)
    behaviour_design_matrix_dict = np.load(os.path.join(data_root_directory, base_directory, "Behaviour_Ridge_Regression", "Behaviour_design_matrix_key_dict.npy"), allow_pickle=True)[()]
    combined_dictionary = merge_dictionaries(behaviour_design_matrix_dict, stimuli_design_matrix_dict)
    np.save(os.path.join(tensor_root_directory, base_directory, "design_matrix_key_dict.npy"), combined_dictionary)



def merge_dictionaries(behaviour_design_matrix_dict, stimuli_design_matrix_dict):

    # Unpack Dictiionaries
    number_of_behaviour_regressor_groups = behaviour_design_matrix_dict["number_of_regressor_groups"]
    behaviour_group_sizes = behaviour_design_matrix_dict[ "coef_group_sizes"]
    behaviour_group_starts = behaviour_design_matrix_dict["coef_group_starts"]
    behaviour_group_stops = behaviour_design_matrix_dict["coef_group_stops"]
    behaviour_group_names = behaviour_design_matrix_dict["coefs_names"]

    number_of_stimuli_regressor_groups = stimuli_design_matrix_dict["number_of_regressor_groups"]
    stimuli_regressor_group_sizes = stimuli_design_matrix_dict["coef_group_sizes"]
    stimuli_group_starts = stimuli_design_matrix_dict["coef_group_starts"]
    stimuli_group_stops = stimuli_design_matrix_dict[ "coef_group_stops"]
    stimuli_group_names = stimuli_design_matrix_dict["coefs_names"]

    # Adjust Behaviour Starts and Stops
    number_of_stimuli_regressors = np.sum(stimuli_regressor_group_sizes)
    behaviour_group_starts = list(np.add(behaviour_group_starts, number_of_stimuli_regressors))
    behaviour_group_stops = list(np.add(behaviour_group_stops, number_of_stimuli_regressors))

    # Concatenate Lists
    number_of_combined_regressor_groups = number_of_behaviour_regressor_groups + number_of_stimuli_regressor_groups
    combined_group_names = stimuli_group_names + behaviour_group_names
    combined_group_sizes = stimuli_regressor_group_sizes + behaviour_group_sizes
    combined_group_starts = stimuli_group_starts + behaviour_group_starts
    combined_group_stops = stimuli_group_stops + behaviour_group_stops

    # Create Combined Dict
    combined_dictionary = {
    "number_of_regressor_groups":number_of_combined_regressor_groups,
    "coef_group_sizes":combined_group_sizes,
    "coef_group_starts":combined_group_starts,
    "coef_group_stops":combined_group_stops,
    "coefs_names":combined_group_names,
    }


    return combined_dictionary


def create_full_model_design_matrix(base_directory, data_root_directory, tensor_root_directory, start_window, stop_window, onset_files, start_cutoff=3000):

    # Check Tensor Directory
    check_tensor_directory(tensor_root_directory, base_directory)

    # Create Behaviour Design Matrix
    Create_Behaviour_Design_Matrix.create_design_matrix(os.path.join(data_root_directory, base_directory))

    # Create Behaviour Tensors
    create_behaviour_tensors(base_directory, data_root_directory, tensor_root_directory, onset_files, start_window, stop_window, start_cutoff)

    # Create Full Model Design Matricies
    create_combined_design_matrix(base_directory, data_root_directory, tensor_root_directory, onset_files, start_window, stop_window)