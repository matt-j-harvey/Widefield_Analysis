import matplotlib.pyplot as plt
import tables
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.gridspec as gridspec
import os

from Widefield_Utils import widefield_utils






def test_signficance_mouse_average(tensor_directory, analysis_name):

    """
    This Test Is Run Pixelwise - All Brains Must Be In Same Pixel Space

    Metadata Structure -  group_index, mouse_index, session_index, condition_index

    :return:
    Tensor of P Values
    """

    # Open Analysis Dataframe
    analysis_file = tables.open_file(os.path.join(tensor_directory, analysis_name + "_Trialwise_.h5"), mode="r")
    activity_dataset = analysis_file.root["Data"]
    metadata_dataset = analysis_file.root["Trial_Details"]
    activity_dataset = np.array(activity_dataset)
    metadata_dataset = np.array(metadata_dataset)
    number_of_trials, number_of_timepoints, number_of_pixels = np.shape(activity_dataset)

    print("Number of timepoints", number_of_timepoints)
    print("number of pixels", number_of_pixels)
    print("number of trials", number_of_trials)

    # Get Mouse Averages
    condition_1_mouse_average_list, condition_2_mouse_average_list = get_mouse_averages(activity_dataset, metadata_dataset)

    # Test Significance
    t_stats, p_values = stats.ttest_rel(condition_1_mouse_average_list, condition_2_mouse_average_list, axis=0)

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

    # Load Colourmap
    colourmap = widefield_utils.get_musall_cmap()

    # Visualise T Stats
    number_of_timepoints = np.shape(t_stats)[0]
    for timepoint_index in range(number_of_timepoints):


        figure_1 = plt.figure()

        raw_axis = figure_1.add_subplot(1,3,1)
        thresholded_axis = figure_1.add_subplot(1,3,2)
        corrected_axis = figure_1.add_subplot(1,3,3)

        # Perform FDR Correction
        p_vector = p_values[timepoint_index]
        rejected, corrected_p_values = fdrcorrection(p_vector, alpha=0.05)

        # Threshold T Vectors
        t_vector = t_stats[timepoint_index]
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

        # Set Title
        raw_axis.set_title(str(timepoint_index))

        plt.show()






def split_trials_by_condition(activity_dataset, metata_dataset):

    condition_list = metata_dataset[:, 3]
    unique_conditions = np.unique(condition_list)

    combined_activity_list = []

    for condition in unique_conditions:
        condition_indicies = np.where(condition_list == condition)[0]
        combined_activity_list.append(activity_dataset[condition_indicies])

    return combined_activity_list



def get_session_averages(activity_dataset, metadata_dataset):

    # Load Session List
    session_list = metadata_dataset[:, 2]
    unique_sessions = np.unique(session_list)
    print("Unique Sessions", unique_sessions)
    condition_1_session_average_list = []
    condition_2_session_average_list = []

    for session in unique_sessions:
        session_indicies = np.where(session_list == session)[0]

        session_trials = activity_dataset[session_indicies]
        session_metadata = metadata_dataset[session_indicies]

        [condition_1_trials, condition_2_trials] = split_trials_by_condition(session_trials, session_metadata)
        print("Condition 1 trials", np.shape(condition_1_trials))
        print("Condition 2 trials", np.shape(condition_2_trials))

        condition_1_mean = np.mean(condition_1_trials, axis=0)
        condition_2_mean = np.mean(condition_2_trials, axis=0)

        print("COndition 1 mean", np.shape(condition_1_mean))

        condition_1_session_average_list.append(condition_1_mean)
        condition_2_session_average_list.append(condition_2_mean)

    return condition_1_session_average_list, condition_2_session_average_list


def get_mouse_averages(activity_dataset, metadata_dataset):

    # Load Session List
    mouse_list = metadata_dataset[:, 1]
    unique_mice = np.unique(mouse_list)

    condition_1_mouse_average_list = []
    condition_2_mouse_average_list = []

    for mouse in unique_mice:
        print("Mouse", mouse)

        mouse_indicies = np.where(mouse_list == mouse)[0]

        mouse_activity_data = activity_dataset[mouse_indicies]
        mouse_metadata = metadata_dataset[mouse_indicies]

        # Get Session Averages
        condition_1_session_averages, condition_2_session_averages = get_session_averages(mouse_activity_data, mouse_metadata)

        # Get Mouse Averages
        condition_1_mouse_average = np.mean(condition_1_session_averages, axis=0)
        condition_2_mouse_average = np.mean(condition_2_session_averages, axis=0)

        # Add To List
        condition_1_mouse_average_list.append(condition_1_mouse_average)
        condition_2_mouse_average_list.append(condition_2_mouse_average)

    return condition_1_mouse_average_list, condition_2_mouse_average_list


def visualise_individual_mice(tensor_directory, analysis_name, window):

    # Open Analysis Dataframe
    analysis_file = tables.open_file(os.path.join(tensor_directory, analysis_name + "_Trialwise_.h5"), mode="r")
    activity_dataset = analysis_file.root["Data"]
    metadata_dataset = analysis_file.root["Trial_Details"]
    activity_dataset = np.array(activity_dataset)
    metadata_dataset = np.array(metadata_dataset)
    number_of_trials, number_of_timepoints, number_of_pixels = np.shape(activity_dataset)

    print("Activity dataset shape", np.shape(activity_dataset))
    print("Number of timepoints", number_of_timepoints)
    print("number of pixels", number_of_pixels)
    print("number of trials", number_of_trials)

    # Get Mouse Averages
    condition_1_mouse_average_list, condition_2_mouse_average_list = get_mouse_averages(activity_dataset, metadata_dataset)

    # Get Window Means
    condition_1_mouse_average_list = condition_1_mouse_average_list[:, window]
    condition_2_mouse_average_list = condition_2_mouse_average_list[:, window]
    condition_1_mouse_average_list = np.mean(condition_1_mouse_average_list, axis=1)
    condition_2_mouse_average_list = np.mean(condition_2_mouse_average_list, axis=1)

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

    # Load Colourmap
    colourmap = widefield_utils.get_musall_cmap()


    number_of_mice = len(condition_1_mouse_average_list)
    figure_1 = plt.figure()
    gridspec_1 = gridspec.GridSpec(ncols=number_of_mice, nrows=3, figure=figure_1)

    for mouse_index in range(number_of_mice):
        condition_1_average = condition_1_mouse_average_list[mouse_index]
        condition_2_average = condition_2_mouse_average_list[mouse_index]
        difference = np.subtract(condition_1_average, condition_2_average)

        condition_1_average = widefield_utils.create_image_from_data(condition_1_average, indicies, image_height, image_width)
        condition_2_average = widefield_utils.create_image_from_data(condition_2_average, indicies, image_height, image_width)
        difference = widefield_utils.create_image_from_data(difference, indicies, image_height, image_width)

        condition_1_axis = figure_1.add_subplot(gridspec_1[0, mouse_index])
        condition_2_axis = figure_1.add_subplot(gridspec_1[1, mouse_index])
        difference_axis = figure_1.add_subplot(gridspec_1[2, mouse_index])

        condition_1_axis.imshow(condition_1_average, cmap=colourmap, vmin=-0.05, vmax=0.05)
        condition_2_axis.imshow(condition_2_average, cmap=colourmap, vmin=-0.05, vmax=0.05)
        difference_axis.imshow(difference_axis, cmap=colourmap, vmin=-0.02, vmax=0.02)

    plt.show()


def test_signficance_mouse_average_window(tensor_directory, analysis_name, window):

    """
    This Test Is Run Pixelwise - All Brains Must Be In Same Pixel Space

    Metadata Structure -  group_index, mouse_index, session_index, condition_index

    :return:
    Tensor of P Values
    """

    # Open Analysis Dataframe
    analysis_file = tables.open_file(os.path.join(tensor_directory, analysis_name + "_Trialwise_.h5"), mode="r")
    activity_dataset = analysis_file.root["Data"]
    metadata_dataset = analysis_file.root["Trial_Details"]
    activity_dataset = np.array(activity_dataset)
    metadata_dataset = np.array(metadata_dataset)
    number_of_trials, number_of_timepoints, number_of_pixels = np.shape(activity_dataset)

    print("Activity dataset shape", np.shape(activity_dataset))
    print("Number of timepoints", number_of_timepoints)
    print("number of pixels", number_of_pixels)
    print("number of trials", number_of_trials)

    # Get Mouse Averages
    condition_1_mouse_average_list, condition_2_mouse_average_list = get_mouse_averages(activity_dataset, metadata_dataset)
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
    rejected, corrected_p_values = fdrcorrection(p_vector, alpha=0.05)

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
