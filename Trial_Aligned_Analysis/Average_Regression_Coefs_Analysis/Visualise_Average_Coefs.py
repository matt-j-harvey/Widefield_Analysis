import os
import h5py
from tqdm import tqdm
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import tables
from datetime import datetime
from collections import Counter
import matplotlib.gridspec as gridspec

"""
    Level 1 - Group
    Level 2 - Mouse
    Level 3 - Learning Stage
    Level 4 - Condition
"""

from Widefield_Utils import widefield_utils


# View For Each Session
# View For Each Mouse
# View Overall Average

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




def view_average_difference(tensor_directory, analysis_name, condition_1_index, condition_2_index, vmin=-0.05, vmax=0.05):

    # Load Data
    activity_dataset, metadata_dataset = load_analysis_data(tensor_directory, analysis_name)

    number_of_sessions, number_of_timepoints, number_of_pixels = np.shape(activity_dataset)

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further( indicies, image_height, image_width)

    # Split By Condition
    condition_details = metadata_dataset[:, 2]
    condition_1_indicies = np.where(condition_details == condition_1_index)[0]
    condition_2_indicies = np.where(condition_details == condition_2_index)[0]

    condition_1_data = activity_dataset[condition_1_indicies]
    condition_2_data = activity_dataset[condition_2_indicies]
    print("Condition 1 data", np.shape(condition_1_data))
    print("condition 2 data", np.shape(condition_2_data))

    # Get MEans
    condition_1_data = np.mean(condition_1_data, axis=0)
    condition_2_data = np.mean(condition_2_data, axis=0)

    # Load Colourmap
    colourmap = widefield_utils.get_musall_cmap()

    for timepoint_index in tqdm(range(number_of_timepoints), position=0, desc="Timepoint"):
        figure_1 = plt.figure()

        condition_1_axis = figure_1.add_subplot(1,3,1)
        condition_2_axis = figure_1.add_subplot(1, 3, 2)
        diff_axis = figure_1.add_subplot(1, 3, 3)

        # Recreate Images
        condition_1_image = widefield_utils.create_image_from_data(condition_1_data[timepoint_index],  indicies, image_height, image_width)
        condition_2_image = widefield_utils.create_image_from_data(condition_2_data[timepoint_index], indicies, image_height, image_width)

        # Plot These
        condition_1_axis.imshow(condition_1_image, cmap=colourmap, vmin=vmin, vmax=vmax)
        condition_2_axis.imshow(condition_2_image, cmap=colourmap, vmin=vmin, vmax=vmax)
        diff_axis.imshow(np.subtract(condition_1_image, condition_2_image), cmap=colourmap, vmin=vmin/2, vmax=vmax/2)


        plt.title(str(timepoint_index))
        plt.show()




def view_average_difference(tensor_directory, analysis_name, condition_1_index, condition_2_index, vmin=-0.05, vmax=0.05):

    # Load Data
    activity_dataset, metadata_dataset = load_analysis_data(tensor_directory, analysis_name)

    number_of_sessions, number_of_timepoints, number_of_pixels = np.shape(activity_dataset)

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further( indicies, image_height, image_width)

    # Split By Condition
    condition_details = metadata_dataset[:, 2]
    condition_1_indicies = np.where(condition_details == condition_1_index)[0]
    condition_2_indicies = np.where(condition_details == condition_2_index)[0]

    condition_1_data = activity_dataset[condition_1_indicies]
    condition_2_data = activity_dataset[condition_2_indicies]
    print("Condition 1 data", np.shape(condition_1_data))
    print("condition 2 data", np.shape(condition_2_data))

    # Get MEans
    condition_1_data = np.mean(condition_1_data, axis=0)
    condition_2_data = np.mean(condition_2_data, axis=0)

    # Load Colourmap
    colourmap = widefield_utils.get_musall_cmap()

    for timepoint_index in tqdm(range(number_of_timepoints), position=0, desc="Timepoint"):
        figure_1 = plt.figure()

        condition_1_axis = figure_1.add_subplot(1,3,1)
        condition_2_axis = figure_1.add_subplot(1, 3, 2)
        diff_axis = figure_1.add_subplot(1, 3, 3)

        # Recreate Images
        condition_1_image = widefield_utils.create_image_from_data(condition_1_data[timepoint_index],  indicies, image_height, image_width)
        condition_2_image = widefield_utils.create_image_from_data(condition_2_data[timepoint_index], indicies, image_height, image_width)

        # Plot These
        condition_1_axis.imshow(condition_1_image, cmap=colourmap, vmin=vmin, vmax=vmax)
        condition_2_axis.imshow(condition_2_image, cmap=colourmap, vmin=vmin, vmax=vmax)
        diff_axis.imshow(np.subtract(condition_1_image, condition_2_image), cmap=colourmap, vmin=vmin/2, vmax=vmax/2)


        plt.title(str(timepoint_index))
        plt.show()



def get_largest_session_number(metadata_datset):
    mouse_list = metadata_datset[:, 1]
    frequency_dict = Counter(mouse_list)
    highest_n = np.max(frequency_dict.values())
    return highest_n


def view_average_difference_per_mouse(tensor_directory, analysis_name, condition_1_index, condition_2_index, vmin=-0.05, vmax=0.05):

    # Load Data
    activity_dataset, metadata_dataset = load_analysis_data(tensor_directory, analysis_name)
    number_of_sessions, number_of_timepoints, number_of_pixels = np.shape(activity_dataset)

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

    # Load Colourmap
    colourmap = widefield_utils.get_musall_cmap()

    mouse_list = metadata_dataset[:, 1]
    unique_mice = np.unique(mouse_list)

    # Get Mouse Data
    condition_1_list_per_mouse = []
    condition_2_list_per_mouse = []

    for mouse in unique_mice:
        mouse_condition_1_data, mouse_condition_2_data = get_mouse_data(activity_dataset, metadata_dataset, mouse, condition_1_index, condition_2_index)
        mouse_condition_1_mean = np.mean(mouse_condition_1_data, axis=0)
        mouse_condition_2_mean = np.mean(mouse_condition_2_data, axis=0)
        condition_1_list_per_mouse.append(mouse_condition_1_mean)
        condition_2_list_per_mouse.append(mouse_condition_2_mean)


    # Plot This
    n_mice = len(condition_1_list_per_mouse)

    for timepoint_index in tqdm(range(number_of_timepoints), position=0, desc="Timepoint"):
        figure_1 = plt.figure(figsize=[8, 8])
        gridspec_1 = gridspec.GridSpec(ncols=n_mice, nrows=3, figure=figure_1)

        for mouse_index in range(n_mice):

            # Create Axes
            condition_1_axis = figure_1.add_subplot(gridspec_1[0, mouse_index])
            condition_2_axis = figure_1.add_subplot(gridspec_1[1, mouse_index])
            diff_axis = figure_1.add_subplot(gridspec_1[2, mouse_index])

            # Get Data
            condition_1_vector = condition_1_list_per_mouse[mouse_index][timepoint_index]
            condition_2_vector = condition_2_list_per_mouse[mouse_index][timepoint_index]
            difference_vector = np.subtract(condition_1_vector, condition_2_vector)

            # Recreate Images
            condition_1_image = widefield_utils.create_image_from_data(condition_1_vector, indicies, image_height, image_width)
            condition_2_image = widefield_utils.create_image_from_data(condition_2_vector, indicies, image_height, image_width)
            difference_image = widefield_utils.create_image_from_data(difference_vector, indicies, image_height, image_width)

            # Plot These
            condition_1_axis.imshow(condition_1_image, cmap=colourmap, vmin=vmin, vmax=vmax)
            condition_2_axis.imshow(condition_2_image, cmap=colourmap, vmin=vmin, vmax=vmax)
            diff_axis.imshow(difference_image, cmap=colourmap, vmin=vmin / 2, vmax=vmax / 2)

        figure_1.suptitle(str(timepoint_index))
        plt.show()






def view_average_difference_per_mouse(tensor_directory, analysis_name, condition_1_index, condition_2_index, vmin=-0.05, vmax=0.05):

    # Load Data
    activity_dataset, metadata_dataset = load_analysis_data(tensor_directory, analysis_name)
    number_of_sessions, number_of_timepoints, number_of_pixels = np.shape(activity_dataset)

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

    # Load Colourmap
    colourmap = widefield_utils.get_musall_cmap()

    mouse_list = metadata_dataset[:, 1]
    unique_mice = np.unique(mouse_list)

    # Get Mouse Data
    condition_1_list_per_mouse = []
    condition_2_list_per_mouse = []

    for mouse in unique_mice:
        mouse_condition_1_data, mouse_condition_2_data = get_mouse_data(activity_dataset, metadata_dataset, mouse, condition_1_index, condition_2_index)
        mouse_condition_1_mean = np.mean(mouse_condition_1_data, axis=0)
        mouse_condition_2_mean = np.mean(mouse_condition_2_data, axis=0)
        condition_1_list_per_mouse.append(mouse_condition_1_mean)
        condition_2_list_per_mouse.append(mouse_condition_2_mean)


    # Plot This
    n_mice = len(condition_1_list_per_mouse)

    for timepoint_index in tqdm(range(number_of_timepoints), position=0, desc="Timepoint"):
        figure_1 = plt.figure(figsize=[8, 8])
        gridspec_1 = gridspec.GridSpec(ncols=n_mice, nrows=3, figure=figure_1)

        for mouse_index in range(n_mice):

            # Create Axes
            condition_1_axis = figure_1.add_subplot(gridspec_1[0, mouse_index])
            condition_2_axis = figure_1.add_subplot(gridspec_1[1, mouse_index])
            diff_axis = figure_1.add_subplot(gridspec_1[2, mouse_index])

            # Get Data
            condition_1_vector = condition_1_list_per_mouse[mouse_index][timepoint_index]
            condition_2_vector = condition_2_list_per_mouse[mouse_index][timepoint_index]
            difference_vector = np.subtract(condition_1_vector, condition_2_vector)

            # Recreate Images
            condition_1_image = widefield_utils.create_image_from_data(condition_1_vector, indicies, image_height, image_width)
            condition_2_image = widefield_utils.create_image_from_data(condition_2_vector, indicies, image_height, image_width)
            difference_image = widefield_utils.create_image_from_data(difference_vector, indicies, image_height, image_width)

            # Plot These
            condition_1_axis.imshow(condition_1_image, cmap=colourmap, vmin=vmin, vmax=vmax)
            condition_2_axis.imshow(condition_2_image, cmap=colourmap, vmin=vmin, vmax=vmax)
            diff_axis.imshow(difference_image, cmap=colourmap, vmin=vmin / 2, vmax=vmax / 2)

        figure_1.suptitle(str(timepoint_index))
        plt.show()


def view_average_difference_per_mouse_learning(tensor_directory, analysis_name, vmin=-0.05, vmax=0.05):

    # Load Data
    activity_dataset, metadata_dataset = load_analysis_data(tensor_directory, analysis_name)
    number_of_sessions, number_of_timepoints, number_of_pixels = np.shape(activity_dataset)

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

    # Load Colourmap
    colourmap = widefield_utils.get_musall_cmap()

    mouse_list = metadata_dataset[:, 1]
    print("Mouse List", mouse_list)

    unique_mice = np.unique(mouse_list)
    print("Unique Mice", unique_mice)

    # Get Mouse Data
    condition_1_list_per_mouse = []
    condition_2_list_per_mouse = []

    for mouse in unique_mice:
        mouse_condition_1_data, mouse_condition_2_data = get_mouse_data_learning(activity_dataset, metadata_dataset, mouse)
        print("Condition 1 data", np.shape(mouse_condition_1_data))
        print("Condition 2 data", np.shape(mouse_condition_2_data))
        mouse_condition_1_mean = np.mean(mouse_condition_1_data, axis=0)
        mouse_condition_2_mean = np.mean(mouse_condition_2_data, axis=0)
        condition_1_list_per_mouse.append(mouse_condition_1_mean)
        condition_2_list_per_mouse.append(mouse_condition_2_mean)


    # Plot This
    n_mice = len(condition_1_list_per_mouse)

    for timepoint_index in tqdm(range(number_of_timepoints), position=0, desc="Timepoint"):
        figure_1 = plt.figure(figsize=[8, 8])
        gridspec_1 = gridspec.GridSpec(ncols=n_mice, nrows=3, figure=figure_1)

        for mouse_index in range(n_mice):

            # Create Axes
            condition_1_axis = figure_1.add_subplot(gridspec_1[0, mouse_index])
            condition_2_axis = figure_1.add_subplot(gridspec_1[1, mouse_index])
            diff_axis = figure_1.add_subplot(gridspec_1[2, mouse_index])

            # Get Data
            condition_1_vector = condition_1_list_per_mouse[mouse_index][timepoint_index]
            condition_2_vector = condition_2_list_per_mouse[mouse_index][timepoint_index]
            difference_vector = np.subtract(condition_1_vector, condition_2_vector)

            # Recreate Images
            condition_1_image = widefield_utils.create_image_from_data(condition_1_vector, indicies, image_height, image_width)
            condition_2_image = widefield_utils.create_image_from_data(condition_2_vector, indicies, image_height, image_width)
            difference_image = widefield_utils.create_image_from_data(difference_vector, indicies, image_height, image_width)

            # Plot These
            condition_1_axis.imshow(condition_1_image, cmap=colourmap, vmin=vmin, vmax=vmax)
            condition_2_axis.imshow(condition_2_image, cmap=colourmap, vmin=vmin, vmax=vmax)
            diff_axis.imshow(difference_image, cmap=colourmap, vmin=vmin / 2, vmax=vmax / 2)

        figure_1.suptitle(str(timepoint_index))
        plt.show()