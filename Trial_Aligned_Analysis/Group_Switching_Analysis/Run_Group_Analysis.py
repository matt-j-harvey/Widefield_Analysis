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
    condition_name = condition_name.replace('_onsets', '')
    condition_name = condition_name.replace('.npy', '')
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


def get_single_group_activity_tensor_list(session_list, tensor_names, tensor_save_directory):

    number_of_conditions = len(tensor_names)

    activity_tensor_list = []
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
            activity_tensor = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, condition_name + "_Activity_Tensor.npy"), allow_pickle=True)
            print("Activity Tensor Shape", np.shape(activity_tensor))
            # Get Average
            mean_activity = np.nanmean(activity_tensor, axis=0)

            # Add To List
            condition_tensor_list.append(mean_activity)

        # Get Group Mean
        condition_tensor_list = np.array(condition_tensor_list)
        condition_mean_tensor = np.mean(condition_tensor_list, axis=0)
        activity_tensor_list.append(condition_mean_tensor)

    return activity_tensor_list


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

def single_group_average(session_list, onset_file_list, tensor_names, start_window, save_directory, tensor_save_directory):

    # Check Save Directory
    Group_Analysis_Utils.check_directory(save_directory)

    # Load Activity Tensors
    print("Loading Activity Tensors")
    activity_tensor_list = get_activity_tensor_list(session_list, onset_file_list, tensor_save_directory)

    # Create Activity Video
    indicies, image_height, image_width = Group_Analysis_Utils.load_tight_mask()
    Create_Video_From_Tensor.create_activity_video(activity_tensor_list, start_window, tensor_names, save_directory, indicies, image_height, image_width, timestep=36)

"""
### Control Pre Post Learning ###
analysis_name = "Hits_Pre_Post_Learning_response"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Group_Analysis_Utils.load_analysis_container(analysis_name)
tensor_save_directory = r"//media/matthew/Expansion/Widefield_Analysis/Learning_Analysis"

save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Control_Learning"
group_names = ["Control_Pre", "Control_Post"]
compare_group_averages(Session_List.control_pre_learning_session_list, Session_List.control_post_learning_session_list, tensor_names[0], group_names, start_window, save_directory, tensor_save_directory, paired_or_unpaired='paired')


### Mutants Pre Post Learning ###
analysis_name = "Hits_Pre_Post_Learning_response"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Group_Analysis_Utils.load_analysis_container(analysis_name)
tensor_save_directory = r"//media/matthew/Expansion/Widefield_Analysis/Learning_Analysis"

save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mutant_Learning"
group_names = ["Mutant_Pre", "Mutant_Post"]
compare_group_averages(Session_List.mutant_pre_learning_session_list, Session_List.mutant_post_learning_session_list, tensor_names[0], group_names, start_window, save_directory, tensor_save_directory, paired_or_unpaired='paired')
"""

### Genotype Comparison Pre Learning ###
analysis_name = "Matched_Correct_Vis_1"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Group_Analysis_Utils.load_analysis_container(analysis_name)
tensor_save_directory = r"//media/matthew/Expansion/Widefield_Analysis/Learning_Analysis"

save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Genotype_Pre"
group_names = ["Control_Pre", "Mutant_Pre"]
compare_group_averages(Session_List.control_pre_learning_session_list, Session_List.mutant_pre_learning_session_list, onset_files[0], group_names, start_window, save_directory, tensor_save_directory, paired_or_unpaired='unpaired')


### Genotype Comparison Post Learning ###
analysis_name = "Matched_Correct_Vis_1"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Group_Analysis_Utils.load_analysis_container(analysis_name)
tensor_save_directory = r"//media/matthew/Expansion/Widefield_Analysis/Learning_Analysis"

save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Genotype_Post"
group_names = ["Control_Post", "Mutant_Post"]
compare_group_averages(Session_List.control_post_learning_session_list, Session_List.mutant_post_learning_session_list, onset_files[0], group_names, start_window, save_directory, tensor_save_directory, paired_or_unpaired='unpaired')



### Correct Rejections Post Learning ###
analysis_name = "Correct_Rejections_Response"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Group_Analysis_Utils.load_analysis_container(analysis_name)
tensor_save_directory = r"//media/matthew/Expansion/Widefield_Analysis/Learning_Analysis"
print("tensor name", tensor_names)
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Genotype_Correct_Rejections"
group_names = ["Control_Post", "Mutant_Post"]
compare_group_averages(Session_List.control_post_learning_session_list, Session_List.mutant_post_learning_session_list, tensor_names[0], group_names, start_window, save_directory, tensor_save_directory, paired_or_unpaired='unpaired')
