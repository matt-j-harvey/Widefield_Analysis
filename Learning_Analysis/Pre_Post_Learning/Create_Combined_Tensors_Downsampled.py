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
            frame = Learning_Utils.create_image_from_data(frame, indicies, image_height, image_width)

            # Align Image
            frame = Learning_Utils.transform_image(frame, alignment_dictionary)

            # Downsize
            frame = resize(frame, (100, 100), preserve_range=True)

            reconstructed_trial.append(frame)
        reconstructed_tensor.append(reconstructed_trial)

    reconstructed_tensor = np.array(reconstructed_tensor)
    return reconstructed_tensor


def apply_shared_tight_mask(activity_tensor):

    # Load Tight Mask
    indicies, image_height, image_width = Learning_Utils.load_tight_mask_downsized()

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


def correct_baseline_of_tensor(tensor, start_size):

    print("Tensor shape", np.shape(tensor))
    corrected_tensor = []

    # Load Mask
    indicies, image_height, image_width = Learning_Utils.load_tight_mask_downsized()

    for trial in tensor:
        print("Trial Shape", np.shape(trial))
        trial_baseline = trial[0:np.abs(start_size)]
        print("Baseline Shape", np.shape(trial_baseline))
        trial_baseline = np.mean(trial_baseline, axis=0)
        print("Baseline Shape", np.shape(trial_baseline))
        trial = np.subtract(trial, trial_baseline)
        corrected_tensor.append(trial)

    corrected_tensor = np.array(corrected_tensor)
    print("Corrected Tensor", np.shape(corrected_tensor))
    return corrected_tensor



def create_standard_alignment_tensor(base_directory, onsets_file, start_window, stop_window):

    # Load Mask
    indicies, image_height, image_width = Learning_Utils.load_generous_mask(base_directory)

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


def save_tensor_to_tables_files(base_directory, activity_tensor, timepoint_file_list, timepoint_group_list, number_of_timepoints, condition_index):

    array_name = get_array_name(base_directory)

    # Load Mask
    indicies, image_height, image_width = Learning_Utils.load_tight_mask_downsized()

    # Save To Tables File For Each Timepoint
    for timepoint_index in range(number_of_timepoints):
        timepoint_data = activity_tensor[:, timepoint_index]
        timepoint_data = np.ndarray.astype(timepoint_data, np.int)
        timepoint_file = timepoint_file_list[timepoint_index]
        timepoint_group = timepoint_group_list[timepoint_index][condition_index]
        tensor_storage = timepoint_file.create_carray(where=timepoint_group, name=array_name, atom=tables.Int32Atom(), shape=(np.shape(timepoint_data)))
        tensor_storage[:] = timepoint_data
        tensor_storage.flush()


def visualise_tensor(activity_tensor):
    indicies, height, width = Learning_Utils.load_tight_mask_downsized()
    average_response = np.mean(activity_tensor, axis=0)

    for frame in average_response:
        image = Learning_Utils.create_image_from_data(frame, indicies, height, width)
        plt.imshow(image)
        plt.show()


def create_combined_tensor_unpaired(group_1_list, group_2_list, group_names, group_1_onset_file, group_2_onset_file, start_window, stop_window, save_directory, baseline_correct=False):

    # Check Save Directory Exists
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Get Number Of Timepoints
    number_of_timepoints = stop_window - start_window

    # Create Files
    timepoint_file_list = []
    timepoint_group_list = []
    for timepoint in range(number_of_timepoints):
        filename = os.path.join(save_directory, "Timepoint_" + str(timepoint).zfill(4) + ".h5")
        timepoint_file = tables.open_file(filename, "w")
        group_1 = timepoint_file.create_group(where="/", name=group_names[0])
        group_2 = timepoint_file.create_group(where="/", name=group_names[1])
        timepoint_file_list.append(timepoint_file)
        timepoint_group_list.append([group_1, group_2])

    # Create Group 1 Tensor
    for base_directory in tqdm(group_1_list):
        activity_tensor = create_standard_alignment_tensor(base_directory, group_1_onset_file, start_window, stop_window)
        if baseline_correct == True: activity_tensor = correct_baseline_of_tensor(activity_tensor, start_window)
        save_tensor_to_tables_files(base_directory, activity_tensor, timepoint_file_list, timepoint_group_list, number_of_timepoints, 0)

    # Create Group 2 Tensor
    for base_directory in tqdm(group_2_list):
        activity_tensor = create_standard_alignment_tensor(base_directory, group_2_onset_file, start_window, stop_window)
        if baseline_correct == True: activity_tensor = correct_baseline_of_tensor(activity_tensor, start_window)
        save_tensor_to_tables_files(base_directory, activity_tensor, timepoint_file_list, timepoint_group_list, number_of_timepoints, 1)

    # Close Files
    for timepoint_file in timepoint_file_list:
        timepoint_file.close()


def create_combined_tensor_paired(tuple_list, onset_file, start_window, stop_window, save_directory, baseline_correct=False):

    # Check Save Directory Exists
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Get Number Of Timepoints
    number_of_timepoints = stop_window - start_window

    # Create Files
    timepoint_file_list = []
    timepoint_group_list = []
    for timepoint in range(number_of_timepoints):
        filename = os.path.join(save_directory, "Timepoint_" + str(timepoint).zfill(4) + ".h5")
        timepoint_file = tables.open_file(filename, "w")
        pre_learning_group = timepoint_file.create_group(where="/", name="Pre_Learning")
        post_learning_group = timepoint_file.create_group(where="/", name="Post_Learning")

        timepoint_file_list.append(timepoint_file)
        timepoint_group_list.append([pre_learning_group, post_learning_group])

    # Create Tensor
    for session_tuple in tqdm(tuple_list):

        # Create Pre Learning Tensor
        activity_tensor = create_standard_alignment_tensor(session_tuple[0], onset_file, start_window, stop_window)

        if baseline_correct == True:
            activity_tensor = correct_baseline_of_tensor(activity_tensor, start_window)

        session_name = Learning_Utils.get_session_name(session_tuple[0])
        np.save(os.path.join(r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Tensor_Checking", session_name + "_activity_tensor.npy"), activity_tensor)

        save_tensor_to_tables_files(session_tuple[0], activity_tensor, timepoint_file_list, timepoint_group_list, number_of_timepoints, 0)

        # Create Post Learning Tensor
        activity_tensor = create_standard_alignment_tensor(session_tuple[1], onset_file, start_window, stop_window)
        if baseline_correct == True:
            activity_tensor = correct_baseline_of_tensor(activity_tensor, start_window)

        session_name = Learning_Utils.get_session_name(session_tuple[1])
        np.save(os.path.join(r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Tensor_Checking", session_name + "_activity_tensor.npy"), activity_tensor)

        save_tensor_to_tables_files(session_tuple[1], activity_tensor, timepoint_file_list, timepoint_group_list, number_of_timepoints, 1)

    # Close Files
    for timepoint_file in timepoint_file_list:
        timepoint_file.close()


# Load Session Tuples
"""
control_tuples = Session_List.control_session_tuples
mutant_tuples = Session_List.mutant_session_tuples

control_sessions = Session_List.control_post_learning_session_list
mutant_sessions = Session_List.mutant_post_learning_session_list

# Load Analysis Details
analysis_name = "Hits_Pre_Post_Learning_response"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Learning_Utils.load_analysis_container(analysis_name)

control_sessions = Session_List.control_post_learning_session_list
mutant_sessions = Session_List.mutant_post_learning_session_list

analysis_name = "Matched_Correct_Vis_1"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Learning_Utils.load_analysis_container(analysis_name)
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Genotype_RT_Matched_Post"
group_names = ["Controls", "Mutants"]
create_combined_tensor_unpaired(control_sessions, mutant_sessions, group_names, onset_files[0], start_window, stop_window, save_directory, baseline_correct=True)

analysis_name = "Correct_Rejections_Response"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Learning_Utils.load_analysis_container(analysis_name)
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Genotype_Correct_Rejections_Post_Learning"
group_names = ["Controls", "Mutants"]
create_combined_tensor_unpaired(control_sessions, mutant_sessions, group_names, onset_files[0], start_window, stop_window, save_directory, baseline_correct=True)

control_sessions = Session_List.control_pre_learning_session_list
mutant_sessions = Session_List.mutant_pre_learning_session_list

analysis_name = "Matched_Correct_Vis_1"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Learning_Utils.load_analysis_container(analysis_name)
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Genotype_RT_Matched_Pre"
group_names = ["Controls", "Mutants"]
create_combined_tensor_unpaired(control_sessions, mutant_sessions, group_names, onset_files[0], start_window, stop_window, save_directory, baseline_correct=True)

save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Mutant_Combined_Tensor_Response_Baseline_Corrected"
create_combined_tensor(mutant_tuples, onset_files[0], start_window, stop_window, save_directory, baseline_correct=True)

save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Control_Combined_Tensor_Response_Baseline_Corrected"
create_combined_tensor(control_tuples, onset_files[0], start_window, stop_window, save_directory, baseline_correct=True)

# Create Activity Tensors
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Mutant_Combined_Tensor"
create_combined_tensor(mutant_tuples, onset_files[0], start_window, stop_window, save_directory)

analysis_name = "Trial_Anticipation"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Learning_Utils.load_analysis_container(analysis_name)

save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Anticipation_Analysis/Mutant_Combined_Tensor"
create_combined_tensor(mutant_tuples, onset_files[0], start_window, stop_window, save_directory)
"""


control_session_list = []
mutant_session_list = []

control_mice = ["NRXN78.1D", "NRXK78.1A", "NAXAK4.1B", "NXAK7.1B", "NXAK14.1A", "NXAK22.1A"]
for mouse in control_mice:
    mouse_sessions = Learning_Utils.load_mouse_sessions(mouse, "Switching")
    for session in mouse_sessions:
        control_session_list.append(session)


mutant_mice = ["NRXN71.2A", "NXAK4.1A", "NXAK10.1A", "NXAK16.1B", "NXAK24.1C", "NXAK20.1B"]
for mouse in mutant_mice:
    mouse_sessions = Learning_Utils.load_mouse_sessions(mouse, "Switching")
    for session in mouse_sessions:
        mutant_session_list.append(session)

"""
analysis_name = "Unrewarded_Contextual_Modulation"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Learning_Utils.load_analysis_container(analysis_name)
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Genotype_Context_Vis_2"
group_names = ["Controls", "Mutants"]
create_combined_tensor_unpaired(control_session_list, mutant_session_list, group_names, onset_files[0], start_window, stop_window, save_directory, baseline_correct=True)


analysis_name = "Unrewarded_Contextual_Modulation"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Learning_Utils.load_analysis_container(analysis_name)
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Control_Contextual_Vis_2"
group_names = ["Visual Context", "Odour Context"]
create_combined_tensor_unpaired(control_session_list, control_session_list, group_names, onset_files[0], onset_files[1], start_window, stop_window, save_directory, baseline_correct=True)


analysis_name = "Unrewarded_Contextual_Modulation"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Learning_Utils.load_analysis_container(analysis_name)
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Mutant_Contextual_Vis_2"
group_names = ["Visual Context", "Odour Context"]
create_combined_tensor_unpaired(mutant_session_list, mutant_session_list, group_names, onset_files[0], onset_files[1], start_window, stop_window, save_directory, baseline_correct=True)
"""

analysis_name = "Unrewarded_Contextual_Modulation"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Learning_Utils.load_analysis_container(analysis_name)
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Genotype_Context_Vis_2_Odour_Context"
group_names = ["Controls", "Mutants"]
create_combined_tensor_unpaired(control_session_list, mutant_session_list, group_names, onset_files[1], onset_files[1], start_window, stop_window, save_directory, baseline_correct=True)

