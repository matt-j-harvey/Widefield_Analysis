import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec
from matplotlib import cm
from matplotlib.pyplot import Normalize
import os
from scipy import stats
from tqdm import tqdm
from skimage.feature import canny
import pickle

import Transition_Utils
from Files import Session_List
from Widefield_Utils import widefield_utils



def load_activity_tensors(session_list, onsets_file, tensor_save_directory, analysis_name, return_mean=False):

    activity_tensor_list = []
    for base_directory in tqdm(session_list):

        # Load Trial Tensor Dict
        trial_tensor_name = onsets_file.replace("_onsets.npy", ".pickle")
        session_tensor_directory = Transition_Utils.check_save_directory(base_directory, tensor_save_directory)
        session_tensor_file = os.path.join(session_tensor_directory, trial_tensor_name)
        print("Session tensor file", session_tensor_file)

        with open(session_tensor_file, 'rb') as handle:
            trial_tensor = pickle.load(handle)

        #trial_tensor = np.load(os.path.join(session_tensor_file), allow_pickle=True)[()]
        activity_tensor = trial_tensor["activity_tensor"]

        print("Activity tensor shape", np.shape(activity_tensor))
        if return_mean == True:
            activity_tensor = np.nanmean(activity_tensor, axis=0)
            activity_tensor = np.reshape(activity_tensor, (1, np.shape(activity_tensor)[0], np.shape(activity_tensor)[1]))
        activity_tensor_list.append(activity_tensor)

    """
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    for frame in activity_tensor:
        frame = widefield_utils.create_image_from_data(frame, indicies, image_height, image_width)

        plt.imshow(frame)
        plt.show()
    """
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
    asses_significance(condition_1_tensor_list, condition_2_tensor_list)

    # Save These Values
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    np.save(os.path.join(save_directory, "2_v_1_t.npy"), cond_2_v_1_t)
    np.save(os.path.join(save_directory, "2_v_1_p.npy"), cond_2_v_1_p)
    np.save(os.path.join(save_directory, "2_v_3_t.npy"), cond_2_v_3_t)
    np.save(os.path.join(save_directory, "2_v_3_p.npy"), cond_2_v_3_p)




def get_background_pixels():

    indicies, image_height, image_width = Transition_Utils.load_tight_mask()

    template = np.ones(image_height * image_width)
    template[indicies] = 0
    template = np.reshape(template, (image_height, image_width))
    background_indicies = np.nonzero(template)
    return background_indicies

def get_mask_edge_pixels():

    indicies, image_height, image_width = Transition_Utils.load_tight_mask()

    template = np.zeros(image_height * image_width)
    template[indicies] = 1
    template = np.reshape(template, (image_height, image_width))
    edges = canny(template)
    plt.imshow(edges)
    plt.show()
    edge_indicies = np.nonzero(edges)
    return edge_indicies


def get_atlas_outline_pixels():

    # Load Atlas
    atlas_outline = np.load("/home/matthew/Documents/Allen_Atlas_Templates/New_Outline.npy")

    # Load Atlas Transformation Dict
    transformation_dict = np.load(r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Consensus_Cluster_Atlas_Alignment_Dictionary.npy", allow_pickle=True)[()]

    # Transform Atlas
    atlas_outline = Transition_Utils.transform_mask_or_atlas(atlas_outline, transformation_dict)



    atlas_pixels = np.nonzero(atlas_outline)
    return atlas_pixels


def view_signficance_maps(results_directory, figure_directory):

    cond_2_v_1_t = np.load(os.path.join(results_directory, "2_v_1_t.npy"))
    cond_2_v_1_p = np.load(os.path.join(results_directory, "2_v_1_p.npy"))
    cond_2_v_3_t = np.load(os.path.join(results_directory, "2_v_3_t.npy"))
    cond_2_v_3_p = np.load(os.path.join(results_directory, "2_v_3_p.npy"))

    # Load Tight Mask Details
    indicies, image_height, image_width = widefield_utils.load_tight_mask()

    # Get Background Pixels
    background_pixels = widefield_utils.get_background_pixels(indicies, image_height, image_width )
    edge_indicies = widefield_utils.get_mask_edge_pixels(indicies, image_height, image_width )

    # Get Atlas Pixels
    atlas_pixels = widefield_utils.get_atlas_outline_pixels()

    # Create Save Directories
    individual_timepoint_directory = os.path.join(figure_directory, "Individual_Timepoints")
    if not os.path.exists(individual_timepoint_directory):
        os.mkdir(individual_timepoint_directory)

    time_average_directory = os.path.join(figure_directory, "Time_Averaged_Window")
    if not os.path.exists(time_average_directory):
        os.mkdir(time_average_directory)

    rows = 1
    columns = 4
    figure_1 = plt.figure(figsize=(15, 9))
    gridspec_1 = GridSpec(nrows=rows, ncols=columns, figure=figure_1)

    number_of_timepoints, number_of_pixels = np.shape(cond_2_v_3_p)

    # Create Colourmap
    sig_vmax = 8
    sig_vmin = 2
    sig_colourmap = cm.ScalarMappable(cmap=cm.get_cmap('viridis'), norm=Normalize(vmin=sig_vmin, vmax=sig_vmax))


    average_period_start = 93
    average_period_stop = 135
    average_period_frames = []


    for timepoint_index in range(number_of_timepoints):

        comparison_1_axis = figure_1.add_subplot(gridspec_1[0, 0])
        comparison_2_axis = figure_1.add_subplot(gridspec_1[0, 1])
        joint_axis = figure_1.add_subplot(gridspec_1[0, 2])
        cbar_axis = figure_1.add_subplot(gridspec_1[0, 3])

        comparison_1_image = cond_2_v_1_t[timepoint_index]
        comparison_2_image = cond_2_v_3_t[timepoint_index]

        comparison_1_image = np.where(cond_2_v_1_p[timepoint_index] < 0.05, comparison_1_image, 0)
        comparison_2_image = np.where(cond_2_v_3_p[timepoint_index] < 0.05, comparison_2_image, 0)

        joint_image = np.where(comparison_1_image > 0, comparison_1_image, 0)
        joint_image = np.where(comparison_2_image > 0, joint_image, 0)

        comparison_1_image = Transition_Utils.create_image_from_data(comparison_1_image, indicies, image_height, image_width)
        comparison_2_image = Transition_Utils.create_image_from_data(comparison_2_image, indicies, image_height, image_width)
        joint_image = Transition_Utils.create_image_from_data(joint_image, indicies, image_height, image_width)

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
        cbar_axis.set_title("t statistic", loc='left')

        figure_1.suptitle(str((timepoint_index - 10)*36))
        figure_1.colorbar(cm.ScalarMappable(norm=Normalize(vmin=sig_vmin, vmax=sig_vmax), cmap='viridis'), ax=cbar_axis, fraction=0.05, location='left')

        plt.draw()

        plt.savefig(os.path.join(individual_timepoint_directory, str(timepoint_index).zfill(3) + ".svg"))
        plt.clf()

    plt.ioff()

    average_window = np.mean(np.array(average_period_frames), axis=0)
    alpha_values = np.clip(average_window, a_min=2, a_max=8)
    alpha_values = np.divide(alpha_values, 8)

    sig_vmax = 3
    sig_vmin = 0
    average_sig_colourmap = cm.ScalarMappable(cmap=cm.get_cmap('Purples'), norm=Normalize(vmin=sig_vmin, vmax=sig_vmax))
    average_window = average_sig_colourmap.to_rgba(average_window, alpha=alpha_values)
    average_window[atlas_pixels] = [0, 0, 0, 1]
    average_window[background_pixels] = [1, 1, 1, 1]

    average_window[edge_indicies] = [0, 0, 0, 1]
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.axis('off')
    axis_1.imshow(average_window)
    figure_1.colorbar(cm.ScalarMappable(norm=Normalize(vmin=sig_vmin, vmax=sig_vmax), cmap='Purples'), ax=axis_1)
    plt.savefig(os.path.join(time_average_directory, "Average_t_map.svg"))
    plt.show()


# Load Session List
selected_session_list = Session_List.mutant_transition_sessions
tensor_root_directory = r"//media/matthew/External_Harddrive_2/Neurexin_Transition_Tensors/Raw_Activity"


#selected_session_list = Session_List.control_transition_sessions
#tensor_root_directory = r"//media/matthew/External_Harddrive_2/Control_Transition_Tensors/Raw_Activity"

# Load Analysis Details
analysis_name = "Absence Of Expected Odour"
[start_window, stop_window, onsets_files_list, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)


results_directory = os.path.join(tensor_root_directory, "Results", analysis_name)
if not os.path.exists(results_directory):
    os.mkdir(results_directory)

create_three_way_signfiance_map(selected_session_list, onsets_files_list, analysis_name, tensor_root_directory, results_directory)

# Plot Map
figure_directory = os.path.join(results_directory, "Figure")
if not os.path.exists(figure_directory):
    os.mkdir(figure_directory)

view_signficance_maps(results_directory, figure_directory)