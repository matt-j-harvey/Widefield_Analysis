import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from sklearn.decomposition import PCA

import Group_Analysis_Utils
import Create_Video_From_Tensor
import Session_List

def denoise_tensor(activity_tensor):

    n_trials, n_timepoints, n_pixels = np.shape(activity_tensor)
    activity_tensor = np.reshape(activity_tensor, (n_trials * n_timepoints, n_pixels))

    model = PCA(n_components=50)
    activity_tensor = model.inverse_transform(model.fit_transform(activity_tensor))
    activity_tensor = np.reshape(activity_tensor, (n_trials, n_timepoints, n_pixels))
    return  activity_tensor


def get_multiple_group_activity_tensor_list(group_1_session_list, group_2_session_list, tensor_name, tensor_save_directory, paired_or_unpaired='unpaired'):

    activity_tensor_list = []
    mean_activity_tensor_list = []

    combined_session_list = [group_1_session_list, group_2_session_list]

    condition_name = tensor_name
    condition_name = condition_name.replace('_onsets.npy', '')

    for session_list in combined_session_list:

        print("condition name", condition_name)
        condition_tensor_list = []

        for base_directory in session_list:
            print("Session: ", base_directory)

            # Get Path Details
            mouse_name, session_name = Group_Analysis_Utils.get_mouse_name_and_session_name(base_directory)

            # Load Activity Tensor
            activity_tensor = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, condition_name + "_Activity_Tensor.npy"), allow_pickle=True)
            print("Activity Tensor Shape", np.shape(activity_tensor))

            activity_tensor = denoise_tensor(activity_tensor)

            # Get Average
            mean_activity = np.nanmean(activity_tensor, axis=0)

            # Add To List
            condition_tensor_list.append(mean_activity)

        # Get Group Mean
        condition_tensor_list = np.array(condition_tensor_list)
        condition_mean_tensor = np.mean(condition_tensor_list, axis=0)

        activity_tensor_list.append(condition_tensor_list)
        mean_activity_tensor_list.append(condition_mean_tensor)

    # Perform T Test
    if paired_or_unpaired == 'unpaired':
        t_stats, p_values = stats.ttest_ind(activity_tensor_list[0], activity_tensor_list[1], axis=0)

    elif paired_or_unpaired == 'paired':
        t_stats, p_values = stats.ttest_rel(activity_tensor_list[0], activity_tensor_list[1], axis=0)

    print("P values", np.shape(p_values))

    return mean_activity_tensor_list, p_values


def get_single_group_activity_tensor_list(session_list, tensor_names, tensor_save_directory, paired_or_unpaired):

    number_of_conditions = len(tensor_names)

    activity_tensor_list = []
    mean_activity_tensor_list = []
    for condition_index in range(number_of_conditions):

        condition_name = tensor_names[condition_index]
        condition_name = condition_name.replace('_onsets.npy','')
        print("condition name", condition_name)
        condition_tensor_list = []

        for base_directory in session_list:
            print("Session: ", base_directory)

            # Get Path Details
            mouse_name, session_name = Group_Analysis_Utils.get_mouse_name_and_session_name(base_directory)

            # Load Activity Tensor
            activity_tensor = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, condition_name + "_Predicited_Tensor.npy"), allow_pickle=True)
            print("Activity Tensor Shape", np.shape(activity_tensor))
            # Get Average
            mean_activity = np.nanmean(activity_tensor, axis=0)

            # Add To List
            condition_tensor_list.append(mean_activity)

        # Get Group Mean
        condition_tensor_list = np.array(condition_tensor_list)
        condition_mean_tensor = np.mean(condition_tensor_list, axis=0)

        activity_tensor_list.append(condition_tensor_list)
        mean_activity_tensor_list.append(condition_mean_tensor)

    # Perform T Test
    if paired_or_unpaired == 'unpaired':
        t_stats, p_values = stats.ttest_ind(activity_tensor_list[0], activity_tensor_list[1], axis=0)

    elif paired_or_unpaired == 'paired':
        t_stats, p_values = stats.ttest_rel(activity_tensor_list[0], activity_tensor_list[1], axis=0)

    return mean_activity_tensor_list, p_values


def compare_group_averages(group_1_session_list, group_2_session_list, tensor_name, group_names, start_window, save_directory, tensor_save_directory, paired_or_unpaired):

    # Check Save Directory
    Group_Analysis_Utils.check_directory(save_directory)

    # Load Activity Tensors
    print("Loading Activity Tensors")
    activity_tensor_list, p_values = get_multiple_group_activity_tensor_list(group_1_session_list, group_2_session_list, tensor_name, tensor_save_directory, paired_or_unpaired=paired_or_unpaired)

    # Create Activity Video
    indicies, image_height, image_width = Group_Analysis_Utils.load_tight_mask()
    #Create_Video_From_Tensor.create_activity_video(activity_tensor_list, start_window, group_names, save_directory, indicies, image_height, image_width, timestep=36)
    Create_Video_From_Tensor.create_activity_video_with_significance(activity_tensor_list, p_values, start_window, group_names, save_directory, indicies, image_height, image_width, timestep=36)


def single_group_average(session_list, onset_file_list, tensor_names, start_window, save_directory, tensor_save_directory, paired_or_unpaired):

    # Check Save Directory
    Group_Analysis_Utils.check_directory(save_directory)

    # Load Activity Tensors
    print("Loading Activity Tensors")
    activity_tensor_list, p_values = get_single_group_activity_tensor_list(session_list, onset_file_list, tensor_save_directory, paired_or_unpaired)

    # Create Activity Video
    indicies, image_height, image_width = Group_Analysis_Utils.load_tight_mask()
    Create_Video_From_Tensor.create_activity_video_with_significance(activity_tensor_list, p_values, start_window, tensor_names, save_directory, indicies, image_height, image_width, timestep=36)



control_session_list = [

    r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_05_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_09_Switching_Imaging",

    r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_29_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_05_Switching_Imaging",

    r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_21_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_23_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_11_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_13_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_15_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_17_Transition_Imaging",

    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_14_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_20_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_22_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_29_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_03_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_05_Transition_Imaging",

    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_02_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_04_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_06_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_02_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_08_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_10_Transition_Imaging",

    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_26_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_28_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_02_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_23_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_31_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_04_02_Transition_Imaging",

]

mutant_session_list = [

    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_13_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_15_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_17_Switching_Imaging",

    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_31_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_02_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_04_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_08_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_10_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_12_Transition_Imaging",

    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_20_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_22_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_24_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_14_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_16_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_18_Transition_Imaging",

    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_17_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_19_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_23_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_30_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_06_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_08_Transition_Imaging",

    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_17_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_19_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_22_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_24_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_26_Transition_Imaging",

    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_14_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_20_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_26_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_05_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_08_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_10_Transition_Imaging",

]


""" Vis 2 Context """
analysis_name = "Unrewarded_Contextual_Modulation"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Group_Analysis_Utils.load_analysis_container(analysis_name)
tensor_save_directory = r"/media/matthew/External_Harddrive_3/Switching_Analysis_Tensors"

save_directory = r"/media/matthew/Expansion/Thesis_Comittee_Analysis/Uncorrected_Switching_Modulation/Mutants_Predicted"
single_group_average(mutant_session_list, onset_files, tensor_names, start_window, save_directory, tensor_save_directory, paired_or_unpaired='paired')

save_directory = r"/media/matthew/Expansion/Thesis_Comittee_Analysis/Uncorrected_Switching_Modulation/Controls_Predicted"
single_group_average(control_session_list, onset_files, tensor_names, start_window, save_directory, tensor_save_directory, paired_or_unpaired='paired')
