import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec
from matplotlib import cm
from matplotlib.pyplot import Normalize
from matplotlib.colors import LogNorm
import os
from scipy import stats
from tqdm import tqdm
from skimage.feature import canny

import Trial_Aligned_Utils




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

def load_activity_tensors(session_list, onsets_file, tensor_save_directory, analysis_name, return_mean=False):

    activity_tensor_list = []
    for base_directory in tqdm(session_list):
        activity_tensor_name = onsets_file.replace("_onsets.npy", "")
        activity_tensor_name = activity_tensor_name + "_Extended_Activity_Tensor.npy"
        session_tensor_directory = Trial_Aligned_Utils.check_save_directory(base_directory, tensor_save_directory)
        session_tensor_file = os.path.join(session_tensor_directory, activity_tensor_name)
        activity_tensor = np.load(session_tensor_file, allow_pickle=True)
        print("Activity tensor shape", np.shape(activity_tensor))
        if return_mean == True:
            activity_tensor = pad_ragged_tensor_with_nans(activity_tensor)
            print("Activity tensor shape", np.shape(activity_tensor))
            activity_tensor = np.nanmean(activity_tensor, axis=0)

            """
            indicies, image_height, image_width = Trial_Aligned_Utils.load_tight_mask()
            for timepoint in activity_tensor:
                timepoint = Trial_Aligned_Utils.create_image_from_data(timepoint, indicies, image_height, image_width)
                plt.imshow(timepoint)
                plt.show()
            """

            activity_tensor = np.reshape(activity_tensor, (1, np.shape(activity_tensor)[0], np.shape(activity_tensor)[1]))

        activity_tensor_list.append(activity_tensor)

    return activity_tensor_list


def asses_significance(condition_1_tensor_list, condition_2_tensor_list):

    t_stats, p_values = stats.ttest_rel(condition_1_tensor_list, condition_2_tensor_list, axis=0)
    print("P Values", np.shape(p_values))



def create_three_way_signfiance_map(session_list, onsets_files_list, analysis_name, tensor_root_directory, save_directory):

    # Load Activity Tensors
    condition_1_tensor_list = load_activity_tensors(session_list, onsets_files_list[0], tensor_root_directory, analysis_name, return_mean=True)
    condition_2_tensor_list = load_activity_tensors(session_list, onsets_files_list[1], tensor_root_directory, analysis_name, return_mean=True)
    condition_3_tensor_list = load_activity_tensors(session_list, onsets_files_list[2], tensor_root_directory, analysis_name, return_mean=True)

    # Concatenate Activity Tensors
    condition_1_tensor_list = np.vstack(condition_1_tensor_list)
    condition_2_tensor_list = np.vstack(condition_2_tensor_list)
    condition_3_tensor_list = np.vstack(condition_3_tensor_list)
    print("Condition 2 tensor list", np.shape(condition_2_tensor_list))

    # Asses Significance
    cond_2_v_1_t, cond_2_v_1_p = stats.ttest_rel(condition_2_tensor_list, condition_1_tensor_list, axis=0)
    cond_2_v_3_t, cond_2_v_3_p = stats.ttest_rel(condition_2_tensor_list, condition_3_tensor_list, axis=0)
    #asses_significance(condition_1_tensor_list, condition_2_tensor_list)
    print("Cond 2 v 1 t", np.shape(cond_2_v_1_t))

    # Save These Values
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    np.save(os.path.join(save_directory, "2_v_1_t.npy"), cond_2_v_1_t)
    np.save(os.path.join(save_directory, "2_v_1_p.npy"), cond_2_v_1_p)
    np.save(os.path.join(save_directory, "2_v_3_t.npy"), cond_2_v_3_t)
    np.save(os.path.join(save_directory, "2_v_3_p.npy"), cond_2_v_3_p)


    #return cmap


def get_background_pixels():

    indicies, image_height, image_width = Trial_Aligned_Utils.load_tight_mask()

    template = np.ones(image_height * image_width)
    template[indicies] = 0
    template = np.reshape(template, (image_height, image_width))
    background_indicies = np.nonzero(template)
    return background_indicies

def get_mask_edge_pixels():

    indicies, image_height, image_width = Trial_Aligned_Utils.load_tight_mask()

    template = np.zeros(image_height * image_width)
    template[indicies] = 1
    template = np.reshape(template, (image_height, image_width))
    edges = canny(template)
    #plt.imshow(edges)
    #plt.show()
    edge_indicies = np.nonzero(edges)
    return edge_indicies


def get_atlas_outline_pixels():

    # Load Atlas
    atlas_outline = np.load("/home/matthew/Documents/Github_Code_Clean/Transition_Analysis/Brain_Registration/Atlas_Outlines.npy")

    # Load Atlas Transformation Dict
    transformation_dict = np.load("/home/matthew/Documents/Github_Code_Clean/Transition_Analysis/Brain_Registration/Atlas_Alignment_Dictionary.npy", allow_pickle=True)[()]

    # Transform Atlas
    atlas_outline = Trial_Aligned_Utils.transform_mask_or_atlas(atlas_outline, transformation_dict)

    #plt.imshow(atlas_outline)
    #plt.show()

    atlas_pixels = np.nonzero(atlas_outline)
    return atlas_pixels



def view_signficance_maps(save_folder):

    root_folder = save_folder
    cond_2_v_1_t = np.load(os.path.join(root_folder, "2_v_1_t.npy"))
    cond_2_v_1_p = np.load(os.path.join(root_folder, "2_v_1_p.npy"))
    cond_2_v_3_t = np.load(os.path.join(root_folder, "2_v_3_t.npy"))
    cond_2_v_3_p = np.load(os.path.join(root_folder, "2_v_3_p.npy"))

    # Load Tight Mask Details
    indicies, image_height, image_width = Trial_Aligned_Utils.load_tight_mask()

    # Get Background Pixels
    background_pixels = get_background_pixels()
    edge_indicies = get_mask_edge_pixels()

    # Get Atlas Pixels
    atlas_pixels = get_atlas_outline_pixels()


    rows = 1
    columns = 4
    figure_1 = plt.figure(figsize=(15, 9))
    gridspec_1 = GridSpec(nrows=rows, ncols=columns, figure=figure_1)

    number_of_timepoints, number_of_pixels = np.shape(cond_2_v_3_p)

    # Create Colourmap
    sig_vmax = 8
    sig_vmin = 2
    sig_colourmap = cm.ScalarMappable(cmap=cm.get_cmap('inferno_r'), norm=Normalize(vmin=sig_vmin, vmax=sig_vmax))
    sig_colourmap = cm.ScalarMappable(cmap=cm.get_cmap('viridis_r'), norm=LogNorm(vmax=1, vmin=0.000001))

    average_period_start = 100
    average_period_stop = 140
    average_period_frames = []

    for timepoint_index in range(number_of_timepoints):

        comparison_1_axis = figure_1.add_subplot(gridspec_1[0, 0])
        comparison_2_axis = figure_1.add_subplot(gridspec_1[0, 1])
        joint_axis = figure_1.add_subplot(gridspec_1[0, 2])
        cbar_axis = figure_1.add_subplot(gridspec_1[0, 3])

        comparison_1_image = cond_2_v_1_p[timepoint_index]
        comparison_2_image = cond_2_v_3_p[timepoint_index]

        #comparison_1_image = np.where(cond_2_v_1_p[timepoint_index] < 0.05, comparison_1_image, 0)
        #comparison_2_image = np.where(cond_2_v_3_p[timepoint_index] < 0.05, comparison_2_image, 0)

        joint_image = np.where(comparison_1_image > 0, comparison_1_image, 1)
        joint_image = np.where(comparison_2_image > 0, joint_image, 1)

        comparison_1_image = Trial_Aligned_Utils.create_image_from_data(comparison_1_image, indicies, image_height, image_width)
        comparison_2_image = Trial_Aligned_Utils.create_image_from_data(comparison_2_image, indicies, image_height, image_width)
        joint_image = Trial_Aligned_Utils.create_image_from_data(joint_image, indicies, image_height, image_width)

        if timepoint_index > average_period_start and timepoint_index < average_period_stop:
            average_period_frames.append(joint_image)

        # Convert To RGBA
        comparison_1_image = sig_colourmap.to_rgba(comparison_1_image)
        comparison_2_image = sig_colourmap.to_rgba(comparison_2_image)
        joint_image = sig_colourmap.to_rgba(joint_image)

        # Add Atlas Outline
        comparison_1_image[atlas_pixels] = [1,1,1,1]
        comparison_2_image[atlas_pixels] =[1,1,1,1]
        joint_image[atlas_pixels] = [1,1,1,1]

        # Remove Background
        comparison_1_image[background_pixels] = [1,1,1,1]
        comparison_2_image[background_pixels] = [1,1,1,1]
        joint_image[background_pixels] = [1,1,1,1]

        comparison_1_axis.imshow(comparison_1_image)
        comparison_2_axis.imshow(comparison_2_image)
        joint_axis.imshow(joint_image)
        cbar_axis.imshow(np.ones(np.shape(joint_image)))

        # Remove Axes
        comparison_1_axis.axis('off')
        comparison_2_axis.axis('off')
        joint_axis.axis('off')
        cbar_axis.axis('off')

        # Add Titles
        comparison_1_axis.set_title("Expected_Present_V_Absent")
        comparison_2_axis.set_title("Absent_V_Not_Expected")
        joint_axis.set_title("Intersection")
        cbar_axis.set_title("log p value", loc='left')

        figure_1.suptitle(str((timepoint_index - 10)*36))
        figure_1.colorbar(sig_colourmap, ax=cbar_axis, fraction=0.05, location='left')

        plt.draw()
        plt.savefig(os.path.join(save_folder, "Individual_Timepoints", str(timepoint_index).zfill(3) + ".svg"))
        plt.clf()

    plt.ioff()

    average_window = np.mean(np.array(average_period_frames), axis=0)
    normaliser = LogNorm(vmax=1, vmin=0.00001)
    alpha_values = normaliser(average_window)
    alpha_values = 1 / alpha_values
    #alpha_values = np.divide(alpha_values, np.percentile(alpha_values, q=95))
    #alpha_values = np.clip(alpha_values, a_min=0, a_max=1)

    average_sig_colourmap = cm.ScalarMappable(cmap=cm.get_cmap('Purples_r'), norm=normaliser)
    average_window = average_sig_colourmap.to_rgba(average_window, alpha=alpha_values)
    average_window[atlas_pixels] = [0, 0, 0, 1]
    #average_window[background_pixels] = [1, 1, 1, 1]

    #average_window[edge_indicies] = [0, 0, 0, 1]
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.axis('off')
    axis_1.imshow(average_window)
    figure_1.colorbar(average_sig_colourmap, ax=axis_1)
    plt.savefig(r"/media/matthew/Expansion/Widefield_Analysis/Transition_Figure/Signficance_Map/TIme_Averaged_Window/Average_t_map.svg")
    plt.show()


# Get Analysis Details
analysis_name = "Absence Of Expected Odour"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Trial_Aligned_Utils.load_analysis_container(analysis_name)
stop_stimuli_list = [["Odour 1", "Visual 1", "Visual 2"], ["Odour 1", "Odour 2", "Visual 1", "Visual 2"], ["Odour 1", "Odour 2", "Visual 1", "Visual 2"]]
tensor_save_directory = r"/media/matthew/29D46574463D2856/Nature_Transition_Analysis_Results/Extended_Tensors"

session_list = [

    r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_23_Transition_Imaging",
    r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_31_Transition_Imaging",
    r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_04_02_Transition_Imaging",

    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_02_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_08_Transition_Imaging",
    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_10_Transition_Imaging",

    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_13_Transition_Imaging",
    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_15_Transition_Imaging",
    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_17_Transition_Imaging",

    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_10_29_Transition_Imaging",
    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_11_03_Transition_Imaging",
    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_11_05_Transition_Imaging"

]

# Run Analysis
save_folder = r"/media/matthew/29D46574463D2856/Nature_Transition_Analysis_Results/Significance_Map"
#create_three_way_signfiance_map(session_list, onset_files, analysis_name, tensor_save_directory, save_folder)
view_signficance_maps(save_folder)