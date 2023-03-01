import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tables
from tqdm.auto import tqdm
import pickle

import Create_Video_From_Tensor
from Widefield_Utils import widefield_utils
from Files import Session_List


def check_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def reconstruct_group_mean(mean_activity, indicies, image_height, image_width):

    reconstructed_mean = []

    for frame in mean_activity:

        # Reconstruct Image
        frame = widefield_utils.create_image_from_data(frame, indicies, image_height, image_width)
        plt.imshow(frame)

        reconstructed_mean.append(frame)

    reconstructed_mean = np.array(reconstructed_mean)
    return reconstructed_mean


def ensure_tensor_structure(tensor):
    if np.ndim(tensor) != 3:
        print("Padding tensor")
        tensor = pad_ragged_tensor_with_nans(tensor)
        print("New shape", np.shape(tensor))
    return tensor


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


def view_mean_activity(base_directory_list, tensor_root_directory, onset_files, plot_titles, save_directory, trial_start):

    # Check Save Directory
    check_directory(save_directory)

    # Load Mask Details
    indicies, image_height, image_width = widefield_utils.load_tight_mask()

    # Get Number Of Conditions
    number_of_conditions = len(tensor_names)

    # Create List To Hold Activity Tensors
    activity_tensor_list = []

    for condition_index in tqdm(range(number_of_conditions), position=0, desc="Condition"):
        trial_tensor_name = onset_files[condition_index].replace("_onsets.npy", ".pickle")

        condition_tensor_list = []
        for base_directory in tqdm(base_directory_list, position=1, desc="Session"):

            # Get Tensor Filename
            session_tensor_directory = widefield_utils.get_session_folder_in_tensor_directory(base_directory, tensor_root_directory)
            session_tensor_file = os.path.join(session_tensor_directory, trial_tensor_name)
            with open(session_tensor_file, 'rb') as handle:
                trial_tensor = pickle.load(handle)
            activity_tensor = trial_tensor["activity_tensor"]

            # Ensure Its In THe Correct Shape
            activity_tensor = ensure_tensor_structure(activity_tensor)

            # Get Average
            mean_activity = np.nanmean(activity_tensor, axis=0)

            # Add To List
            condition_tensor_list.append(mean_activity)

        # Get Group Mean
        condition_tensor_list = np.array(condition_tensor_list)
        condition_tensor_list = ensure_tensor_structure(condition_tensor_list)
        condition_mean_tensor = np.nanmean(condition_tensor_list, axis=0)

        # Reconstruct Group Mean
        #condition_mean_tensor = reconstruct_group_mean(condition_mean_tensor, indicies, image_height, image_width)

        activity_tensor_list.append(condition_mean_tensor)

    # Check Save Directory Exists
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # View Individual Movie
    Create_Video_From_Tensor.create_activity_video(activity_tensor_list, trial_start, plot_titles, save_directory, indicies, image_height, image_width)

# Select Sessions and Directories
#selected_session_list = Session_List.control_transition_sessions
#tensor_root_directory = r"/media/matthew/External_Harddrive_2/Control_Transition_Tensors/Raw_Activity"
#save_directory = r"/media/matthew/External_Harddrive_2/Control_Transition_Tensors/Raw_Activity/Results/Absence Of Expected Odour/Average_Activity_Video"


selected_session_list = Session_List.mutant_transition_sessions
tensor_root_directory = r"/media/matthew/External_Harddrive_2/Neurexin_Transition_Tensors/Raw_Activity"
save_directory = r"/media/matthew/External_Harddrive_2/Neurexin_Transition_Tensors/Raw_Activity/Results/Absence Of Expected Odour/Average_Activity_Video"


# Load Analysis Details
analysis_name = "Absence Of Expected Odour"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)

# Create Video
view_mean_activity(selected_session_list, tensor_root_directory, onset_files, tensor_names, save_directory, start_window)

