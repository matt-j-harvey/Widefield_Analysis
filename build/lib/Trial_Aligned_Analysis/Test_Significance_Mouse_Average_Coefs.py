import matplotlib.pyplot as plt
import tables
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.gridspec as gridspec
import os

from Widefield_Utils import widefield_utils


def load_analysis_data(tensor_directory, analysis_name):

    # Open Analysis Dataframe
    analysis_file = tables.open_file(os.path.join(tensor_directory, analysis_name + "_Trialwise_.h5"), mode="r")
    activity_dataset = analysis_file.root["Data"]
    metadata_dataset = analysis_file.root["Trial_Details"]
    metadata_dataset = np.array(metadata_dataset)
    activity_dataset = np.array(activity_dataset)
    print("metadata_dataset", np.shape(metadata_dataset))
    print("activity_dataset", np.shape(activity_dataset))

    return activity_dataset, metadata_dataset


def get_mouse_data(activity_list, metadata_dataset, selected_mouse, condition_1, condition_2):

    # Unpack Metadata
    mouse_list = metadata_dataset[:, 1]
    condition_list = metadata_dataset[:, 2]

    # Get Mouse Sessions
    condition_1_mouse_indicies = np.where((mouse_list == selected_mouse) & (condition_list == condition_1))
    condition_2_mouse_indicies = np.where((mouse_list == selected_mouse) & (condition_list == condition_2))

    # Get Mouse Data
    mouse_condition_1_data = activity_list[condition_1_mouse_indicies]
    mouse_condition_2_data = activity_list[condition_2_mouse_indicies]

    return mouse_condition_1_data, mouse_condition_2_data


def baseline_correct_data(mouse_condition_data, baseline_window, response_window):

    corrected_data = []

    # Baseline Correct
    for session in mouse_condition_data:
        print("Session Shape", np.shape(session))
        session_baseline = session[baseline_window]
        session_baseline = np.mean(session_baseline, axis=0)
        session_response = session[response_window]
        session_response = np.mean(session_response, axis=0)
        session_response = np.subtract(session_response, session_baseline)
        corrected_data.append(session_response)

    return corrected_data

def get_mouse_data_baseline_correct(activity_list, metadata_dataset, selected_mouse, condition_1, condition_2, baseline_window, response_window):

    # Unpack Metadata
    mouse_list = metadata_dataset[:, 1]
    condition_list = metadata_dataset[:, 2]

    # Get Mouse Sessions
    condition_1_mouse_indicies = np.where((mouse_list == selected_mouse) & (condition_list == condition_1))
    condition_2_mouse_indicies = np.where((mouse_list == selected_mouse) & (condition_list == condition_2))

    # Get Mouse Data
    mouse_condition_1_data = activity_list[condition_1_mouse_indicies]
    mouse_condition_2_data = activity_list[condition_2_mouse_indicies]

    # Baseline Correct
    mouse_condition_1_data = baseline_correct_data(mouse_condition_1_data, baseline_window, response_window)
    mouse_condition_2_data = baseline_correct_data(mouse_condition_2_data, baseline_window, response_window)

    return mouse_condition_1_data, mouse_condition_2_data


def get_mouse_averages(activity_list, metadata_dataset, condition_1, condition_2):

    condition_1_mouse_average_list = []
    condition_2_mouse_average_list = []

    mouse_list = metadata_dataset[:, 1]
    unique_mice = np.unique(mouse_list)

    for selected_mouse in unique_mice:
        mouse_condition_1_data, mouse_condition_2_data = get_mouse_data(activity_list, metadata_dataset, selected_mouse, condition_1, condition_2, baseline_correct, baseline_window)
        mouse_condition_1_mean = np.mean(mouse_condition_1_data, axis=0)
        mouse_condition_2_mean = np.mean(mouse_condition_2_data, axis=0)
        condition_1_mouse_average_list.append(mouse_condition_1_mean)
        condition_2_mouse_average_list.append(mouse_condition_2_mean)

    return condition_1_mouse_average_list, condition_2_mouse_average_list



def get_mouse_averages_learning(activity_list, metadata_dataset):

    condition_1_mouse_average_list = []
    condition_2_mouse_average_list = []

    mouse_list = metadata_dataset[:, 1]
    unique_mice = np.unique(mouse_list)

    for selected_mouse in unique_mice:
        mouse_condition_1_data, mouse_condition_2_data = get_mouse_data_learning(activity_list, metadata_dataset, selected_mouse)

        mouse_condition_1_mean = np.mean(mouse_condition_1_data, axis=0)
        mouse_condition_2_mean = np.mean(mouse_condition_2_data, axis=0)
        condition_1_mouse_average_list.append(mouse_condition_1_mean)
        condition_2_mouse_average_list.append(mouse_condition_2_mean)

    return condition_1_mouse_average_list, condition_2_mouse_average_list


def get_mouse_averages_baseline_correct(activity_list, metadata_dataset, condition_1, condition_2, baseline_window, response_window):

    condition_1_mouse_average_list = []
    condition_2_mouse_average_list = []

    mouse_list = metadata_dataset[:, 1]
    unique_mice = np.unique(mouse_list)

    for selected_mouse in unique_mice:
        mouse_condition_1_data, mouse_condition_2_data = get_mouse_data_baseline_correct(activity_list, metadata_dataset, selected_mouse, condition_1, condition_2, baseline_window, response_window)
        mouse_condition_1_mean = np.mean(mouse_condition_1_data, axis=0)
        mouse_condition_2_mean = np.mean(mouse_condition_2_data, axis=0)
        condition_1_mouse_average_list.append(mouse_condition_1_mean)
        condition_2_mouse_average_list.append(mouse_condition_2_mean)

    return condition_1_mouse_average_list, condition_2_mouse_average_list




def test_signficance_mouse_average_window(tensor_directory, analysis_name, window, condition_1, condition_2):

    """
    This Test Is Run Pixelwise - All Brains Must Be In Same Pixel Space

    Metadata Structure -  group_index, mouse_index, session_index, condition_index

    :return:
    Tensor of P Values
    """

    # Open Analysis Dataframe
    activity_dataset, metadata_dataset = load_analysis_data(tensor_directory, analysis_name)
    number_of_sessions, number_of_timepoints, number_of_pixels = np.shape(activity_dataset)

    # Get Mouse Averages
    condition_1_mouse_average_list, condition_2_mouse_average_list = get_mouse_averages(activity_dataset, metadata_dataset, condition_1, condition_2)
    condition_1_mouse_average_list = np.array(condition_1_mouse_average_list)
    condition_2_mouse_average_list = np.array(condition_2_mouse_average_list)
    print("Condition 1 average list", np.shape(condition_1_mouse_average_list))
    print("Condition 2 average list", np.shape(condition_2_mouse_average_list))

    # Get Window Means
    condition_1_mouse_average_list = condition_1_mouse_average_list[:, window]
    condition_2_mouse_average_list = condition_2_mouse_average_list[:, window]
    condition_1_mouse_average_list = np.mean(condition_1_mouse_average_list, axis=1)
    condition_2_mouse_average_list = np.mean(condition_2_mouse_average_list, axis=1)

    # Test Significance
    t_vector, p_vector = stats.ttest_rel(condition_1_mouse_average_list, condition_2_mouse_average_list, axis=0)

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

    # Load Colourmap
    colourmap = widefield_utils.get_musall_cmap()

    # Visualise T Stats
    figure_1 = plt.figure()

    raw_axis = figure_1.add_subplot(1,3,1)
    thresholded_axis = figure_1.add_subplot(1,3,2)
    corrected_axis = figure_1.add_subplot(1,3,3)

    # Perform FDR Correction
    rejected, corrected_p_values = fdrcorrection(p_vector, alpha=0.1)

    # Threshold T Vectors
    thresholded_t_vector = np.where(p_vector < 0.05, t_vector, 0)
    corrected_t_vector = np.where(rejected == 1, t_vector, 0)

    # Reconstruct Into Maps
    raw_t_map = widefield_utils.create_image_from_data(t_vector, indicies, image_height, image_width)
    thresholded_t_map = widefield_utils.create_image_from_data(thresholded_t_vector, indicies, image_height, image_width)
    corrected_t_map = widefield_utils.create_image_from_data(corrected_t_vector, indicies, image_height, image_width)

    # Display These
    raw_axis.imshow(raw_t_map, cmap=colourmap, vmin=-6, vmax=6)
    thresholded_axis.imshow(thresholded_t_map, cmap=colourmap, vmin=-6, vmax=6)
    corrected_axis.imshow(corrected_t_map, cmap=colourmap, vmin=-6, vmax=6)


    plt.show()







def test_signficance_mouse_average_window_baseline_correct(tensor_directory, analysis_name, window, condition_1, condition_2, baseline_window):

    """
    This Test Is Run Pixelwise - All Brains Must Be In Same Pixel Space

    Metadata Structure -  group_index, mouse_index, session_index, condition_index

    :return:
    Tensor of P Values
    """

    # Open Analysis Dataframe
    activity_dataset, metadata_dataset = load_analysis_data(tensor_directory, analysis_name)
    number_of_sessions, number_of_timepoints, number_of_pixels = np.shape(activity_dataset)

    # Get Mouse Averages
    condition_1_mouse_average_list, condition_2_mouse_average_list = get_mouse_averages_baseline_correct(activity_dataset, metadata_dataset, condition_1, condition_2, baseline_window, window)
    condition_1_mouse_average_list = np.array(condition_1_mouse_average_list)
    condition_2_mouse_average_list = np.array(condition_2_mouse_average_list)
    print("Condition 1 average list", np.shape(condition_1_mouse_average_list))
    print("Condition 2 average list", np.shape(condition_2_mouse_average_list))

    # Test Significance
    t_vector, p_vector = stats.ttest_rel(condition_1_mouse_average_list, condition_2_mouse_average_list, axis=0)

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

    # Load Colourmap
    colourmap = widefield_utils.get_musall_cmap()

    # Visualise T Stats
    figure_1 = plt.figure()

    raw_axis = figure_1.add_subplot(1,3,1)
    thresholded_axis = figure_1.add_subplot(1,3,2)
    corrected_axis = figure_1.add_subplot(1,3,3)

    # Perform FDR Correction
    rejected, corrected_p_values = fdrcorrection(p_vector, alpha=0.1)

    # Threshold T Vectors
    thresholded_t_vector = np.where(p_vector < 0.05, t_vector, 0)
    corrected_t_vector = np.where(rejected == 1, t_vector, 0)

    # Reconstruct Into Maps
    raw_t_map = widefield_utils.create_image_from_data(t_vector, indicies, image_height, image_width)
    thresholded_t_map = widefield_utils.create_image_from_data(thresholded_t_vector, indicies, image_height, image_width)
    corrected_t_map = widefield_utils.create_image_from_data(corrected_t_vector, indicies, image_height, image_width)

    # Display These
    raw_axis.imshow(raw_t_map, cmap=colourmap, vmin=-6, vmax=6)
    thresholded_axis.imshow(thresholded_t_map, cmap=colourmap, vmin=-6, vmax=6)
    corrected_axis.imshow(corrected_t_map, cmap=colourmap, vmin=-6, vmax=6)


    plt.show()




def get_mouse_data_learning(activity_list, metadata_dataset, selected_mouse):

    # Unpack Metadata
    mouse_list = metadata_dataset[:, 1]
    learning_stage_list = metadata_dataset[:, 2]

    # Get Mouse Sessions
    mouse_pre_learning_indicies = np.where((mouse_list == selected_mouse) & (learning_stage_list == 0))
    mouse_post_learning_indicies = np.where((mouse_list == selected_mouse) & (learning_stage_list == 1))

    # Get Mouse Data
    mouse_pre_learning_data = activity_list[mouse_pre_learning_indicies]
    mouse_post_learning_data = activity_list[mouse_post_learning_indicies]

    return mouse_pre_learning_data, mouse_post_learning_data



def test_signficance_mouse_average_learning(tensor_directory, analysis_name, window):

    """
    This Test Is Run Pixelwise - All Brains Must Be In Same Pixel Space

    Metadata Structure -  group_index, mouse_index, session_index, condition_index

    :return:
    Tensor of P Values
    """

    # Open Analysis Dataframe
    activity_dataset, metadata_dataset = load_analysis_data(tensor_directory, analysis_name)

    # Get Mouse Averages
    condition_1_mouse_average_list, condition_2_mouse_average_list = get_mouse_averages_learning(activity_dataset, metadata_dataset)
    condition_1_mouse_average_list = np.array(condition_1_mouse_average_list)
    condition_2_mouse_average_list = np.array(condition_2_mouse_average_list)
    print("Condition 1 average list", np.shape(condition_1_mouse_average_list))
    print("Condition 2 average list", np.shape(condition_2_mouse_average_list))

    # Get Window Means
    condition_1_mouse_average_list = condition_1_mouse_average_list[:, window]
    condition_2_mouse_average_list = condition_2_mouse_average_list[:, window]
    condition_1_mouse_average_list = np.mean(condition_1_mouse_average_list, axis=1)
    condition_2_mouse_average_list = np.mean(condition_2_mouse_average_list, axis=1)

    # Test Significance
    t_vector, p_vector = stats.ttest_rel(condition_1_mouse_average_list, condition_2_mouse_average_list, axis=0)

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

    # Load Colourmap
    colourmap = widefield_utils.get_musall_cmap()

    # Visualise T Stats
    figure_1 = plt.figure()

    raw_axis = figure_1.add_subplot(1,3,1)
    thresholded_axis = figure_1.add_subplot(1,3,2)
    corrected_axis = figure_1.add_subplot(1,3,3)

    # Perform FDR Correction
    rejected, corrected_p_values = fdrcorrection(p_vector, alpha=0.1)

    # Threshold T Vectors
    thresholded_t_vector = np.where(p_vector < 0.05, t_vector, 0)
    corrected_t_vector = np.where(rejected == 1, t_vector, 0)

    # Reconstruct Into Maps
    raw_t_map = widefield_utils.create_image_from_data(t_vector, indicies, image_height, image_width)
    thresholded_t_map = widefield_utils.create_image_from_data(thresholded_t_vector, indicies, image_height, image_width)
    corrected_t_map = widefield_utils.create_image_from_data(corrected_t_vector, indicies, image_height, image_width)

    # Display These
    raw_axis.imshow(raw_t_map, cmap=colourmap, vmin=-6, vmax=6)
    thresholded_axis.imshow(thresholded_t_map, cmap=colourmap, vmin=-6, vmax=6)
    corrected_axis.imshow(corrected_t_map, cmap=colourmap, vmin=-6, vmax=6)


    plt.show()

