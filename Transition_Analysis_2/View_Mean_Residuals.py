import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tables
from tqdm import tqdm


import Create_Video_From_Tensor
import Transition_Utils


def check_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)



def reconstruct_group_mean(mean_activity, indicies, image_height, image_width):

    reconstructed_mean = []

    for frame in mean_activity:

        # Reconstruct Image
        frame = Transition_Utils.create_image_from_data(frame, indicies, image_height, image_width)

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


def uncorrected_workflow_group_extended(base_directory_list, tensor_names, model_name, plot_titles, save_directory, trial_start):

    # Check Save Directory
    check_directory(save_directory)

    # Load Mask Details
    indicies, image_height, image_width = Transition_Utils.load_tight_mask()

    # Get Number Of Conditions
    number_of_conditions = len(tensor_names)

    # Create List To Hold Activity Tensors
    activity_tensor_list = []

    # Extended Residual Tensor Root Directory
    extended_tensor_root_directory = r"/media/matthew/Expansion/Widefield_Analysis/Extended_Tensors"

    for condition_index in range(number_of_conditions):

        condition_name = tensor_names[condition_index].replace("_onsets.npy", "")
        condition_tensor_list = []

        for base_directory in tqdm(base_directory_list):

            # Get Tensor Filename
            session_tensor_directory = Transition_Utils.check_save_directory(base_directory, extended_tensor_root_directory)

            activity_tensor_file = os.path.join(session_tensor_directory, condition_name + "_Extended_Activity_Tensor.npy")
            print("Activity Tensor File", activity_tensor_file)

            if os.path.exists(activity_tensor_file):
                activity_tensor = np.load(activity_tensor_file, allow_pickle=True)
                print("Activity Tensor Shape", np.shape(activity_tensor))

                # Ensure Its In THe Correct Shape
                activity_tensor = ensure_tensor_structure(activity_tensor)

                # Get Average
                mean_activity = np.nanmean(activity_tensor, axis=0)
                print("Mean Activity Shape", np.shape(mean_activity))

                # Add To List
                condition_tensor_list.append(mean_activity)

        # Get Group Mean
        condition_tensor_list = np.array(condition_tensor_list)
        condition_tensor_list = ensure_tensor_structure(condition_tensor_list)
        condition_mean_tensor = np.nanmean(condition_tensor_list, axis=0)

        # Reconstruct Group Mean
        condition_mean_tensor = reconstruct_group_mean(condition_mean_tensor, indicies, image_height, image_width)

        activity_tensor_list.append(condition_mean_tensor)
        print("Condition Mean Tensor", np.shape(condition_mean_tensor))

    # Check Save Directory Exists
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # View Individual Movie
    Create_Video_From_Tensor.create_activity_video(activity_tensor_list, trial_start, plot_titles, save_directory)


# Load Session List
mouse_list = ["NRXN78.1D", "NXAK4.1B", "NXAK7.1B", "NXAK14.1A", "NXAK22.1A"]
session_type = "Transition"
session_list = []
for mouse_name in mouse_list:
    session_list = session_list + Transition_Utils.load_mouse_sessions(mouse_name, session_type)

# Load Analysis Details

analysis_name = "Absence Of Expected Odour"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Transition_Utils.load_analysis_container(analysis_name)

save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Transition_Figure/Mean_Activity_Comic"
"""
# Load Analysis Details
analysis_name = "Perfect_v_Imperfect_Switches"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Transition_Utils.load_analysis_container(analysis_name)
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Transition_Figure/Mean_Activity_Comic_Perfec_v_Imperfect"
"""
# Create
#save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Transition_Figure/Mean_Activity_Comic"
uncorrected_workflow_group_extended(session_list, onset_files, analysis_name, tensor_names, save_directory, start_window)
