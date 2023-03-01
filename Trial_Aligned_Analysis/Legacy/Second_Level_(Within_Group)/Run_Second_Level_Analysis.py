import math

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy import stats, ndimage
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import fdrcorrection
import mne
from datetime import datetime

from Files import Session_List
from Widefield_Utils import widefield_utils, Create_Activity_Tensor, Create_Video_From_Tensor

def denoise_tensor(activity_tensor):

    n_trials, n_timepoints, n_pixels = np.shape(activity_tensor)
    activity_tensor = np.reshape(activity_tensor, (n_trials * n_timepoints, n_pixels))

    model = PCA(n_components=50)
    activity_tensor = model.inverse_transform(model.fit_transform(activity_tensor))
    activity_tensor = np.reshape(activity_tensor, (n_trials, n_timepoints, n_pixels))

    return activity_tensor


def smooth_tensor(tensor, indicies, image_height, image_width):

    tensor = np.nan_to_num(tensor)

    smoothed_tensor = []
    for trial in tensor:

        smoothed_trial = []
        for frame in trial:

            frame = widefield_utils.create_image_from_data(frame, indicies, image_height, image_width)
            frame = ndimage.gaussian_filter(frame, sigma=2)
            frame = np.reshape(frame, image_height * image_width)
            frame = frame[indicies]
            smoothed_trial.append(frame)
        smoothed_tensor.append(smoothed_trial)

    smoothed_tensor = np.array(smoothed_tensor)

    return smoothed_tensor



def reconstruct_tensor(tensor, indicies, image_height, image_width):

    reconstructed_tensor = []
    for trial in tensor:

        reconstructed_trial = []
        for frame in trial:
            frame = widefield_utils.create_image_from_data(frame, indicies, image_height, image_width)
            reconstructed_trial.append(frame)

        reconstructed_tensor.append(reconstructed_trial)

    reconstructed_tensor = np.array(reconstructed_tensor)

    return reconstructed_tensor



def run_second_level_analysis(session_list, tensor_list, tensor_save_directory, paired_or_unpaired='unpaired'):

    activity_tensor_list = []

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()

    for condition_name in tensor_list:
        condition_tensors = []
        condition_name = condition_name.replace('_onsets', '')
        condition_name = condition_name.replace('.npy', '')

        for base_directory in tqdm(session_list):

            # Get Path Details
            mouse_name, session_name = widefield_utils.get_mouse_name_and_session_name(base_directory)

            # Load Activity Tensor
            activity_tensor = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, condition_name + "_Activity_Tensor.npy"), allow_pickle=True)
            print("Activity tensor shape", np.shape(activity_tensor))

            # Smooth tensor
            activity_tensor = smooth_tensor(activity_tensor, indicies, image_height, image_width)

            # Get Means
            activity_mean = np.mean(activity_tensor, axis=0)

            # Add To List
            condition_tensors.append(activity_mean)

        activity_tensor_list.append(condition_tensors)

    activity_tensor_list = np.array(activity_tensor_list)
    print("Activity tensor list shape", np.shape(activity_tensor_list))

    p_value_tensor = []
    t_stat_tensor = []

    # Perform T Test
    number_of_timepoints = np.shape(activity_tensor_list)[2]
    for timepoint in range(number_of_timepoints):

        condition_1_data = activity_tensor_list[0, :, timepoint]
        condition_2_data = activity_tensor_list[1, :, timepoint]
        print("condition 1 data shape", np.shape(condition_1_data))

        if paired_or_unpaired == 'unpaired':
            t_stats, p_values = stats.ttest_ind(condition_1_data, condition_2_data, axis=0)

        elif paired_or_unpaired == 'paired':
            t_stats, p_values = stats.ttest_rel(condition_1_data, condition_2_data, axis=0)

        print("P vzlues", np.shape(p_values))
        p_value_tensor.append(p_values)
        t_stat_tensor.append(t_stats)

    p_value_tensor = np.array(p_value_tensor)
    t_stat_tensor = np.array(t_stat_tensor)
    print("T stats shape", np.shape(p_value_tensor))

    return t_stat_tensor, p_value_tensor



def run_second_level_analysis_cluster(session_list, tensor_list, tensor_save_directory):

    activity_tensor_list = []

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()

    for condition_name in tensor_list:
        condition_tensors = []
        condition_name = condition_name.replace('_onsets', '')
        condition_name = condition_name.replace('.npy', '')

        for base_directory in tqdm(session_list):

            # Get Path Details
            mouse_name, session_name = widefield_utils.get_mouse_name_and_session_name(base_directory)

            # Load Activity Tensor
            activity_tensor = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, condition_name + "_Activity_Tensor.npy"), allow_pickle=True)
            print("Activity tensor shape", np.shape(activity_tensor))

            # Smooth tensor
            activity_tensor = smooth_tensor(activity_tensor, indicies, image_height, image_width)

            # Reconstruct Tensor
            activity_tensor = reconstruct_tensor(activity_tensor, indicies, image_height, image_width)
            print("Activity Tensor", np.shape(activity_tensor))

            # Get Means
            activity_mean = np.mean(activity_tensor, axis=0)
            print("Activity Mean", np.mean(activity_tensor))

            # Add To List
            condition_tensors.append(activity_mean)

        activity_tensor_list.append(condition_tensors)

    activity_tensor_list = np.array(activity_tensor_list)
    print("Activity tensor list shape", np.shape(activity_tensor_list))

    p_value_tensor = []

    threshold_tfce = dict(start=0, step=0.2)

    # Perform T Test
    number_of_timepoints = np.shape(activity_tensor_list)[2]
    for timepoint in range(number_of_timepoints):

        condition_1_data = activity_tensor_list[0, :, timepoint]
        condition_2_data = activity_tensor_list[1, :, timepoint]
        print("condition 1 data shape", np.shape(condition_1_data))
        print("Condition 2 Data Shape", np.shape(condition_2_data))

        F_obs, clusters, cluster_pvs, H0 = mne.stats.permutation_cluster_test(X=[condition_1_data, condition_2_data], threshold=threshold_tfce, out_type='mask', n_permutations=1024)

        # Reshape P Values
        p_map = np.reshape(cluster_pvs, (image_height, image_width))

        print("P vzlues", np.shape(p_map))
        p_value_tensor.append(p_map)

    p_value_tensor = np.array(p_value_tensor)

    return p_value_tensor


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


def view_p_and_t_values(p_values, t_stats):

    # Remove NaNs
    p_values = np.nan_to_num(p_values)
    t_stats = np.nan_to_num(t_stats)

    # Load Tight Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()

    # Get Data Strcuture
    number_of_frames, number_of_pixels = np.shape(p_values)

    t_colourmap = widefield_utils.get_musall_cmap()

    # Plot Each Frame
    for frame_index in range(number_of_frames):

        # Create Axes
        figure_1 = plt.figure()
        p_axis = figure_1.add_subplot(1,3,1)
        t_axis = figure_1.add_subplot(1,3,2)
        fdr_axis = figure_1.add_subplot(1,3,3)

        # Get FDR Ps
        rejected, p_corrected = fdrcorrection(p_values[frame_index])

        # Reconstruct image
        p_frame = widefield_utils.create_image_from_data(p_values[frame_index], indicies, image_height, image_width)
        t_frame = widefield_utils.create_image_from_data(t_stats[frame_index], indicies, image_height, image_width)
        fdr_frame = widefield_utils.create_image_from_data(rejected, indicies, image_height, image_width)

        # Inverse
        p_frame = 1.0 / p_frame

        # Plot
        plt.title(str(frame_index))
        p_axis.imshow(p_frame, vmin=0, vmax=10000, cmap='jet')
        t_axis.imshow(t_frame, vmin=-10, vmax=10, cmap=t_colourmap)
        fdr_axis.imshow(fdr_frame, vmin=0, vmax=1)
        #p_axis.colorbar()
        #t_axis.colorbar()
        plt.show()


def get_mean_session_tensor_reconstructed(session_list, tensor_list, tensor_save_directory):

    """
    Will Return A Tensor Of Shape - (C x S x T x H x W):

    C x M x T x P
    C - Conditions
    S - Sessions
    T - Trial Length
    H - Height
    W - Width
    """

    activity_tensor_list = []

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()

    for condition_name in tensor_list:
        condition_list = []

        condition_name = condition_name.replace('_onsets', '')
        condition_name = condition_name.replace('.npy', '')

        for session in tqdm(session_list):

            # Get Path Details
            mouse_name, session_name = widefield_utils.get_mouse_name_and_session_name(session)

            # Load Activity Tensor
            activity_tensor = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, condition_name + "_Activity_Tensor.npy"), allow_pickle=True)

            # Smooth tensor
            activity_tensor = smooth_tensor(activity_tensor, indicies, image_height, image_width)

            # Reconstruct Tensor
            activity_tensor = reconstruct_tensor(activity_tensor, indicies, image_height, image_width)

            # Get Average
            mean_activity = np.nanmean(activity_tensor, axis=0)

            # Add To List
            condition_list.append(mean_activity)

        # Add Condition List To Activity Tensor
        activity_tensor_list.append(condition_list)

    # Convert To Array
    activity_tensor_list = np.array(activity_tensor_list)
    print("Mean Actiivt Tensor Shope sessions", np.shape(activity_tensor_list))

    return activity_tensor_list



def get_mean_mouse_tensor_reconstructed(nested_session_list, tensor_list, tensor_save_directory):

    """
    Will Return A Tensor Of Shape - (C x M x T x H x W):

    C x M x T x P
    C - Conditions
    M - Mice
    T - Trial Length
    H - Height
    W - Width
    """

    mean_activity_tensor = []

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()

    for condition_name in tensor_list:
        condition_list = []

        condition_name = condition_name.replace('_onsets', '')
        condition_name = condition_name.replace('.npy', '')

        for mouse in tqdm(nested_session_list):
            mouse_condition_list = []

            for session in mouse:

                # Get Path Details
                mouse_name, session_name = widefield_utils.get_mouse_name_and_session_name(session)

                # Load Activity Tensor
                activity_tensor = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, condition_name + "_Activity_Tensor.npy"), allow_pickle=True)

                # Smooth tensor
                activity_tensor = smooth_tensor(activity_tensor, indicies, image_height, image_width)

                # Reconstruct Tensor
                activity_tensor = reconstruct_tensor(activity_tensor, indicies, image_height, image_width)
                print("Activity Tensor Shape", np.shape(activity_tensor))

                # Get Average
                mean_activity = np.nanmean(activity_tensor, axis=0)
                print("Mean activity Shape", np.shape(mean_activity))

                # Add To List
                mouse_condition_list.append(mean_activity)

            # Get Mean Mouse Condition
            mean_mouse_condition = np.mean(mouse_condition_list, axis=0)
            print("Mean Mouse Condition", np.shape(mean_mouse_condition))

            # Add Mean Mouse To Condition List
            condition_list.append(mean_mouse_condition)

        # Add Condition List To Activity Tensor
        mean_activity_tensor.append(condition_list)

    # Convert To Array
    mean_activity_tensor = np.array(mean_activity_tensor)
    print("Mean Actiivt Tensor Shope", np.shape(mean_activity_tensor))

    return mean_activity_tensor



def paired_cluster_signficance_test(activity_tensor_list):

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()

    activity_tensor_list = np.nan_to_num(activity_tensor_list)

    # Convert To Diffference Tensor For 1 Sample T Test
    modulation_list = np.subtract(activity_tensor_list[0], activity_tensor_list[1])

    # Convert To Array
    modulation_list = np.array(modulation_list)
    print("Modulation list shape", np.shape(modulation_list))
    threshold_tfce = dict(start=0, step=0.2)
    number_of_timepoints = np.shape(modulation_list)[1]


    # Get Window
    modulation_window = modulation_list[:, 10:38]
    modulation_window = np.mean(modulation_window, axis=1)

    F_obs, clusters, cluster_pvs, H0 = mne.stats.permutation_cluster_1samp_test(X=modulation_window, threshold=threshold_tfce, out_type='mask', n_permutations=1024, n_jobs=4)

    # Reshape P Values
    p_map = np.reshape(cluster_pvs, (image_height, image_width))

    np.save("/media/matthew/29D46574463D2856/Significance_Testing/Cluster_Testing/Session_Control_Modulation/Window_p_map.npy", p_map)
    plt.imshow(p_map)
    plt.show()

    # For Each Timepoint
    p_value_tensor = []
    for timepoint in range(number_of_timepoints):
        F_obs, clusters, cluster_pvs, H0 = mne.stats.permutation_cluster_1samp_test(X=modulation_list[:, timepoint], threshold=threshold_tfce, out_type='mask', n_permutations=1024, n_jobs=2)

        # Reshape P Values
        p_map = np.reshape(cluster_pvs, (image_height, image_width))

        # Save This
        np.save(os.path.join("/media/matthew/29D46574463D2856/Significance_Testing/Cluster_Testing/Session_Control_Modulation", str(timepoint).zfill(3) + ".npy"), p_map)

        p_value_tensor.append(p_map)

    p_value_tensor = np.array(p_value_tensor)

    return p_value_tensor



def run_second_level_analysis_paired_nested(nested_session_list, tensor_list, tensor_save_directory):

    modulation_list = []

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()

    for mouse in tqdm(nested_session_list):
        mouse_difference_values = []

        for session in mouse:
            session_tensors = []

            for condition_name in tensor_list:

                condition_name = condition_name.replace('_onsets', '')
                condition_name = condition_name.replace('.npy', '')

                # Get Path Details
                mouse_name, session_name = widefield_utils.get_mouse_name_and_session_name(session)

                # Load Activity Tensor
                activity_tensor = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, condition_name + "_Activity_Tensor.npy"), allow_pickle=True)

                # Smooth tensor
                activity_tensor = smooth_tensor(activity_tensor, indicies, image_height, image_width)

                # Reconstruct Tensor
                activity_tensor = reconstruct_tensor(activity_tensor, indicies, image_height, image_width)

                # Get Average
                mean_activity = np.nanmean(activity_tensor, axis=0)

                # Add To List
                session_tensors.append(mean_activity)

            # Get Session Difference
            session_difference = np.subtract(session_tensors[0], session_tensors[1])
            mouse_difference_values.append(session_difference)

        # Get Mouse Modulation
        mouse_modulation = np.mean(mouse_difference_values, axis=0)
        modulation_list.append(mouse_modulation)

    # Test Significance
    # Perform T Test
    modulation_list = np.array(modulation_list)
    print("Modulation list shape", np.shape(modulation_list))
    threshold_tfce = dict(start=0, step=0.2)
    number_of_timepoints = np.shape(modulation_list)[1]



    p_value_tensor = []
    for timepoint in range(number_of_timepoints):

        F_obs, clusters, cluster_pvs, H0 = mne.stats.permutation_cluster_1samp_test(X=modulation_list[:, timepoint], threshold=threshold_tfce, out_type='mask', n_permutations=1024)

        # Reshape P Values
        p_map = np.reshape(cluster_pvs, (image_height, image_width))

        p_value_tensor.append(p_map)

    p_value_tensor = np.array(p_value_tensor)

    return p_value_tensor


def view_p_maps(p_values):

    # Remove NaNs
    p_values = np.nan_to_num(p_values)

    # Get Data Strcuture
    number_of_frames = np.shape(p_values)[0]
    # Plot Each Frame
    for frame_index in range(number_of_frames):

        # Create Axes
        figure_1 = plt.figure()
        p_axis = figure_1.add_subplot(1,1,1)

        # Reconstruct image
        p_frame = p_values[frame_index]

        # Inverse
        #p_frame = 1.0 / p_frame

        p_frame = 1 - p_frame

        # Plot
        plt.title(str(frame_index))
        p_axis.imshow(p_frame, vmin=0.95, vmax=1, cmap='inferno')

        plt.show()





### Correct Rejections Post Learning ###
analysis_name = "Unrewarded_Contextual_Modulation"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)
tensor_save_directory = r"//media/matthew/Expansion/Widefield_Analysis/Learning_Analysis"

# Intermediate Significance Resting Folder
signficance_testing_folder = r"/media/matthew/29D46574463D2856/Significance_Testing"

# Load Session List
nested_session_list = Session_List.control_switching_nested
session_list = Session_List.control_switching_sessions

"""
# Get Difference Tensor
mean_mouse_tensor = get_mean_mouse_tensor_reconstructed(nested_session_list, onset_files, tensor_save_directory)

# Save This
np.save(os.path.join(signficance_testing_folder, "Controls_Contextual_Modulation_Mouse_Average.npy"), mean_mouse_tensor)

# Load This
mean_mouse_tensor = np.load(os.path.join(signficance_testing_folder, "Controls_Contextual_Modulation_Mouse_Average.npy"))

# Test Significance
p_tensor = paired_cluster_signficance_test(mean_mouse_tensor)
np.save(r"/media/matthew/Expansion/Widefield_Analysis/Signficiance_Testings/mouse_average_p_tensor.npy", p_tensor)

# View
p_tensor = np.load(r"/media/matthew/Expansion/Widefield_Analysis/Signficiance_Testings/mouse_average_p_tensor.npy")
view_p_maps(p_tensor)
"""



# Get Difference Tensor
mean_session_tensor = get_mean_session_tensor_reconstructed(session_list, onset_files, tensor_save_directory)

# Save This
np.save(os.path.join(signficance_testing_folder, "Controls_Contextual_Modulation_Session_Average.npy"), mean_session_tensor)

# Load This
mean_session_tensor = np.load(os.path.join(signficance_testing_folder, "Controls_Contextual_Modulation_Session_Average.npy"))

# Test Significance
p_tensor = paired_cluster_signficance_test(mean_session_tensor)
np.save(r"/media/matthew/Expansion/Widefield_Analysis/Signficiance_Testings/session_average_p_tensor.npy", p_tensor)



print("p tensor shhape", np.shape(p_tensor))


# View P Tensor
view_p_maps(p_tensor)

#run_second_level_analysis_paired_nested(nested_session_list, onset_files, tensor_save_directory)

"""
print("Start: ", datetime.now())
p_tensor = run_second_level_analysis_cluster(session_list, onset_files, tensor_save_directory)
print("Done: ", datetime.now())

np.save(r"/media/matthew/Expansion/Widefield_Analysis/Signficiance_Testings/p_tensor.npy", p_tensor)
print("p tensor shhape", np.shape(p_tensor))


p_tensor = np.load(r"/media/matthew/Expansion/Widefield_Analysis/Signficiance_Testings/p_tensor.npy")

view_p_maps(p_tensor)

"""

#t_stats, p_values = run_second_level_analysis(session_list, onset_files, tensor_save_directory, paired_or_unpaired='paired')

