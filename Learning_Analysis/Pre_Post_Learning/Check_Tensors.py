import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec
import os
import tables

import Learning_Utils




def repackage_into_numpy_arrays(file_container):

    pre_learning_activity_list = []
    post_learning_activity_list = []

    for array in file_container.list_nodes(where="/Pre_Learning"):
        array = np.array(array)
        for trial in array:
            pre_learning_activity_list.append(trial)

    for array in file_container.list_nodes(where="/Post_Learning"):
        array = np.array(array)
        for trial in array:
            post_learning_activity_list.append(trial)

    pre_learning_activity_list = np.array(pre_learning_activity_list)
    post_learning_activity_list = np.array(post_learning_activity_list)

    return pre_learning_activity_list, post_learning_activity_list



def load_numpy_data(base_directory):

    file_list = os.listdir(base_directory)

    combined_tensor = []
    for activity_tensor_file in file_list:

        activity_tensor = np.load(os.path.join(base_directory, activity_tensor_file))
        combined_tensor.append(activity_tensor)

    combined_tensor = np.vstack(combined_tensor)
    print("Combined Tensor Shape", np.shape(combined_tensor))
    return combined_tensor


def check_numpy_arrays(base_directory):

    # Load Data
    pre_learning_data = load_numpy_data(os.path.join(base_directory, "Pre_Learning"))
    post_learning_data = load_numpy_data(os.path.join(base_directory, "Post_Learning"))

    # Load Mask
    indicies, image_height, image_width = Learning_Utils.load_tight_mask_downsized()

    # Get Mean
    pre_learning_mean = np.mean(pre_learning_data, axis=0)
    post_learning_mean = np.mean(post_learning_data, axis=0)

    # Display Data

    # Create Figure
    figure_1 = plt.figure()
    gridspec_1 = GridSpec(nrows=1, ncols=3, figure=figure_1)
    difference_colourmap = Learning_Utils.get_mussall_cmap()

    number_of_timepoints = np.shape(pre_learning_mean)[0]

    for timepoint_index in range(number_of_timepoints):
        condition_1_axis = figure_1.add_subplot(gridspec_1[0, 0])
        condition_2_axis = figure_1.add_subplot(gridspec_1[0, 1])
        difference_axis = figure_1.add_subplot(gridspec_1[0, 2])

        pre_learning_frame = pre_learning_mean[timepoint_index]
        post_learning_frame = post_learning_mean[timepoint_index]
        difference_frame = np.subtract(post_learning_frame, pre_learning_frame)

        pre_learning_frame = Learning_Utils.create_image_from_data(pre_learning_frame, indicies, image_height, image_width)
        post_learning_frame = Learning_Utils.create_image_from_data(post_learning_frame, indicies, image_height, image_width)
        difference_frame = Learning_Utils.create_image_from_data(difference_frame, indicies, image_height, image_width)

        plt.title(" Timepoint: " + str(timepoint_index))
        condition_1_axis.imshow(pre_learning_frame, vmin=0, vmax=15000)
        condition_2_axis.imshow(post_learning_frame, vmin=0, vmax=15000)
        difference_axis.imshow(difference_frame, vmin=-6000, vmax=6000, cmap=difference_colourmap)

        plt.draw()
        plt.pause(0.1)
        plt.clf()


def check_tables_file(base_directory):

    # load Mask
    indicies, image_height, image_width = Learning_Utils.load_tight_mask_downsized()

    # Load Colourmaps
    difference_colourmap = Learning_Utils.get_mussall_cmap()
    difference_magntidue = 6000

    # Load File List
    file_list = os.listdir(base_directory)

    # Iterate Though Timepoints
    count = 0

    # Create Figure
    figure_1 = plt.figure()
    gridspec_1 = GridSpec(nrows=1, ncols=3, figure=figure_1)

    plt.ion()
    for data_file in file_list[count:]:

        condition_1_axis = figure_1.add_subplot(gridspec_1[0, 0])
        condition_2_axis = figure_1.add_subplot(gridspec_1[0, 1])
        difference_axis = figure_1.add_subplot(gridspec_1[0, 2])

        # Extract Data
        file_container = tables.open_file(os.path.join(base_directory, data_file), "r")
        pre_learning_array, post_learning_array = repackage_into_numpy_arrays(file_container)
        file_container.close()

        # Get Mean Pre and Post Matricies
        pre_learning_mean = np.mean(pre_learning_array, axis=0)
        post_learning_mean = np.mean(post_learning_array, axis=0)
        difference = np.subtract(post_learning_mean, pre_learning_mean)

        # Create Images
        pre_learning_image = Learning_Utils.create_image_from_data(pre_learning_mean, indicies, image_height, image_width)
        post_learning_image = Learning_Utils.create_image_from_data(post_learning_mean, indicies, image_height, image_width)
        difference_image = Learning_Utils.create_image_from_data(difference, indicies, image_height, image_width)

        # Plot Images
        condition_1_axis.imshow(pre_learning_image, vmin=0, vmax=15000)
        condition_2_axis.imshow(post_learning_image, vmin=0, vmax=15000)
        difference_axis.imshow(difference_image, cmap=difference_colourmap, vmax=difference_magntidue, vmin=-1 * difference_magntidue)

        # Remove Axis
        condition_1_axis.axis('off')
        condition_2_axis.axis('off')
        difference_axis.axis('off')

        # Set Titles
        difference_axis.set_title("Differences")

        # Save Figure
        plt.draw()
        plt.pause(0.1)
        plt.clf()

    count += 1


numpy_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Tensor_Checking"
tables_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Control_Combined_Tensor_Response_Baseline_Corrected"

#check_numpy_arrays(numpy_directory)
check_tables_file(tables_directory)