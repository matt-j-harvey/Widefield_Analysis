import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn.linear_model import Ridge
import os
import math
import scipy
import tables
from bisect import bisect_left
import cv2
from sklearn.decomposition import TruncatedSVD
from pathlib import Path
import joblib
from scipy import signal, ndimage, stats
from skimage.transform import resize
from scipy.interpolate import interp1d
import sys
import matplotlib.gridspec as gridspec

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions


def get_activity_tensor(activity_matrix, onsets, start_window, stop_window):

    number_of_pixels = np.shape(activity_matrix)[1]
    number_of_trials = np.shape(onsets)[0]
    number_of_timepoints = stop_window - start_window

    # Create Empty Tensor To Hold Data
    activity_tensor = np.zeros((number_of_trials, number_of_timepoints, number_of_pixels))

    # Get Correlation Matrix For Each Trial
    for trial_index in range(number_of_trials):

        # Get Trial Activity
        trial_start = onsets[trial_index] + start_window
        trial_stop = onsets[trial_index] + stop_window
        trial_activity = activity_matrix[trial_start:trial_stop]
        activity_tensor[trial_index] = trial_activity

    return activity_tensor


def reconstruct_images_from_activity(activity_matrix, base_directory):

    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    number_of_frames = np.shape(activity_matrix)[0]
    image_matrix = np.zeros([number_of_frames, image_height, image_width])
    for frame_index in range(number_of_frames):
        activity_vector = activity_matrix[frame_index]
        image = Widefield_General_Functions.create_image_from_data(activity_vector, indicies, image_height, image_width)
        image_matrix[frame_index] = image

    return image_matrix


def get_average_response(session_list, onsets_file_list, trial_start, trial_stop, save_directory):

    average_response_list = []

    for base_directory in session_list:
        print(base_directory.split('/')[-1])

        # Load Delta F Matrix
        delta_f_matrix_filepath = os.path.join(base_directory, "Delta_F.h5")
        delta_f_matrix_container = tables.open_file(delta_f_matrix_filepath, mode='r')
        delta_f_matrix = delta_f_matrix_container.root['Data']

        # Load Onsets
        onsets = []
        for onsets_file in onsets_file_list:
            full_onset_file_path = os.path.join(base_directory, "Stimuli_Onsets", onsets_file)
            onsets_contents = np.load(full_onset_file_path)
            for onset in onsets_contents:
                onsets.append(onset)

        # Create Trial Tensor
        activity_tensor = get_activity_tensor(delta_f_matrix, onsets, trial_start, trial_stop)

        # Get Mean Activity
        mean_activity = np.mean(activity_tensor, axis=0)

        # Add To List
        average_response_list.append(mean_activity)

    np.save(save_directory, average_response_list)

    return average_response_list



def plot_response_differences(base_directory_list, visual_responses, odour_responses):

    number_of_mice = len(base_directory_list)

    # Convert Activity Vectors Into Images
    odour_responses_reconstructed_list = []
    visual_responses_reconstructed_list = []
    for mouse in range(number_of_mice):
        odour_response_images = reconstruct_images_from_activity(odour_responses[mouse], base_directory_list[mouse])
        visual_response_images = reconstruct_images_from_activity(visual_responses[mouse], base_directory_list[mouse])
        odour_responses_reconstructed_list.append(odour_response_images)
        visual_responses_reconstructed_list.append(visual_response_images)

    number_of_columns = number_of_mice
    number_of_rows = 3
    figure_1 = plt.figure(constrained_layout=True)


    number_of_timepoints = np.shape(odour_responses[0])[0]
    print("Number of mice", number_of_mice)
    print("Number of timepoints", number_of_timepoints)

    for timepoint in range(number_of_timepoints):
        grid_spec_1 = gridspec.GridSpec(ncols=number_of_columns, nrows=number_of_rows, figure=figure_1)

        for mouse in range(number_of_mice):

            # Add Axes
            visual_axis     = figure_1.add_subplot(grid_spec_1[0, mouse])
            odour_axis      = figure_1.add_subplot(grid_spec_1[1, mouse])
            difference_axis = figure_1.add_subplot(grid_spec_1[2, mouse])

            # Select Images
            visual_response_image = visual_responses_reconstructed_list[mouse][timepoint]
            odour_reponse_image = odour_responses_reconstructed_list[mouse][timepoint]
            difference_image = np.diff([visual_response_image, odour_reponse_image], axis=0)[0]

            # Plot These Images
            visual_axis.imshow(visual_response_image, cmap='jet', vmin=0, vmax=1)
            odour_axis.imshow(odour_reponse_image,    cmap='jet', vmin=0, vmax=1)
            difference_axis.imshow(difference_image, cmap='bwr', vmin=-0.5, vmax=0.5)

            # Remove Axes
            visual_axis.axis('off')
            odour_axis.axis('off')
            difference_axis.axis('off')


            # Set Title
            visual_axis.set_title("Visual " + str(timepoint))
            odour_axis.set_title("Odour " + str(timepoint))
            difference_axis.set_title("Difference " + str(timepoint))

        plt.draw()
        plt.pause(0.1)
        plt.clf()


def add_region_boundaries(base_directory):

    # Load Atlas Regions
    atlas_region_mapping = np.load(r"/home/matthew/Documents/Allen_Atlas_Templates/Atlas_Template_V2.npy")

    # Load Atlas Transformation Details
    atlas_alignment_dictionary = np.load(os.path.join(base_directory, "Atlas_Alignment_Dictionary.npy"), allow_pickle=True)
    atlas_alignment_dictionary = atlas_alignment_dictionary[()]
    atlas_rotation = atlas_alignment_dictionary['rotation']
    atlas_x_scale_factor = atlas_alignment_dictionary['x_scale_factor']
    atlas_y_scale_factor = atlas_alignment_dictionary['y_scale_factor']
    atlas_x_shift = atlas_alignment_dictionary['x_shift']
    atlas_y_shift = atlas_alignment_dictionary['y_shift']

    # Rotate Atlas
    atlas_region_mapping = ndimage.rotate(atlas_region_mapping, atlas_rotation, reshape=False, )
    atlas_region_mapping = np.clip(atlas_region_mapping, a_min=0, a_max=None)

    # Scale Atlas
    atlas_height = np.shape(atlas_region_mapping)[0]
    atlas_width = np.shape(atlas_region_mapping)[1]
    atlas_region_mapping = resize(atlas_region_mapping, (int(atlas_y_scale_factor * atlas_height), int(atlas_x_scale_factor * atlas_width)), preserve_range=True)

    #plt.imshow(atlas_region_mapping)
    #plt.show()

    # Load mask
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    # Place Into Bounding Box
    bounding_array = np.zeros((800, 800))
    x_start = 100
    y_start = 100
    atlas_height = np.shape(atlas_region_mapping)[0]
    atlas_width = np.shape(atlas_region_mapping)[1]
    bounding_array[y_start + atlas_y_shift: y_start + atlas_y_shift + atlas_height,
    x_start + atlas_x_shift: x_start + atlas_x_shift + atlas_width] = atlas_region_mapping
    bounded_atlas = bounding_array[y_start:y_start + image_height, x_start:x_start + image_width]

    # Mask Atlas
    bounded_atlas = np.ndarray.flatten(bounded_atlas)
    masked_atlas = np.zeros((image_height * image_width))
    for pixel_index in indicies:
        masked_atlas[pixel_index] = bounded_atlas[pixel_index]
    masked_atlas = np.ndarray.reshape(masked_atlas, (image_height, image_width))

    # Binaise Masked Atlas
    masked_atlas = np.where(masked_atlas > 0.1, 1, 0)

    # Save Mapping
    #np.save(os.path.join(base_directory, "Pixel_Assignmnets_Image.npy"), masked_atlas),
    #plt.imshow(masked_atlas)

    #plt.savefig(os.path.join(base_directory, "Pixel_Region_Assignmnet.png"))
    #plt.show()


    mask_indicies = np.nonzero(np.ndarray.flatten(masked_atlas))


    fromindex = np.zeros((image_height * image_width))
    for index in mask_indicies:
        fromindex[index] = 1

    fromindex = np.ndarray.reshape(fromindex, (image_height, image_width))
    #plt.imshow(fromindex)
    #plt.show()

    return masked_atlas, mask_indicies


def scale_difference_image(difference_image):

    difference_magnitude = 0.2
    difference_image = np.clip(difference_image, a_min=-1 * difference_magnitude, a_max=difference_magnitude)
    difference_image = np.divide(difference_image, difference_magnitude)
    difference_image = np.divide(difference_image, 2)
    difference_image = np.add(difference_image, 0.5)
    return difference_image


def gaussian_smooth_image(data, image_height, image_width):
    data = np.ndarray.reshape(data, (image_height, image_width))
    data = ndimage.gaussian_filter(data, sigma=1)
    data = np.ndarray.reshape(data, (image_height * image_width))
    return data




def create_single_mouse_comparison_video_3_conditions(base_directory, tensor_filenames, trial_start, trial_stop, plot_titles, save_directory):

    # Get Region Boundaries
    masked_atlas, atlas_indicies = add_region_boundaries(base_directory)

    # Load Tensors
    condition_1_activity_tensor = np.load(os.path.join(base_directory, "Activity_Tensors", tensor_filenames[0] + "_Activity_Tensor.npy"))
    condition_1_activity_mean = np.mean(condition_1_activity_tensor, axis=0)
    condition_1_activity_tensor = reconstruct_images_from_activity(condition_1_activity_mean, base_directory)

    condition_2_activity_tensor = np.load(os.path.join(base_directory, "Activity_Tensors", tensor_filenames[1] + "_Activity_Tensor.npy"))
    condition_2_activity_mean = np.mean(condition_2_activity_tensor, axis=0)
    condition_2_activity_tensor = reconstruct_images_from_activity(condition_2_activity_mean, base_directory)

    condition_3_activity_tensor = np.load(os.path.join(base_directory, "Activity_Tensors", tensor_filenames[2] + "_Activity_Tensor.npy"))
    condition_3_activity_mean = np.mean(condition_3_activity_tensor, axis=0)
    condition_3_activity_tensor = reconstruct_images_from_activity(condition_3_activity_mean, base_directory)

    # Plot
    number_of_timepoints = np.shape(condition_1_activity_tensor)[0]
    number_of_columns = 4
    number_of_rows = 1
    figure_1 = plt.figure(constrained_layout=True, figsize=(80, 60))

    timepoint_count = 0
    timepoint_list = list(range(trial_start, trial_stop))
    timepoint_list = np.multiply(timepoint_list, 36)

    # Get Full Save Path
    full_save_path = os.path.join(base_directory, save_directory)
    if not os.path.exists(full_save_path):
        os.mkdir(full_save_path)

    for timepoint in range(number_of_timepoints):
        print(timepoint)
        grid_spec_1 = gridspec.GridSpec(ncols=number_of_columns, nrows=number_of_rows, figure=figure_1)

        # Add Axes
        condition_1_activity_axis = figure_1.add_subplot(grid_spec_1[0, 0])
        condition_2_activity_axis = figure_1.add_subplot(grid_spec_1[0, 1])
        condition_3_activity_axis = figure_1.add_subplot(grid_spec_1[0, 2])
        difference_axis = figure_1.add_subplot(grid_spec_1[0, 3])

        # Select Images
        condition_1_activity_image = condition_1_activity_tensor[timepoint]
        condition_2_activity_image = condition_2_activity_tensor[timepoint]
        condition_3_activity_image = condition_3_activity_tensor[timepoint]
        difference_image = np.diff([condition_1_activity_image, condition_2_activity_image], axis=0)[0]

        image_height = np.shape(condition_1_activity_image)[0]
        image_width = np.shape(condition_1_activity_image)[1]

        # Gaussian Smoothing
        condition_1_activity_image = gaussian_smooth_image(condition_1_activity_image, image_height, image_width)
        condition_2_activity_image = gaussian_smooth_image(condition_2_activity_image, image_height, image_width)
        condition_3_activity_image = gaussian_smooth_image(condition_3_activity_image, image_height, image_width)
        difference_image = gaussian_smooth_image(difference_image, image_height, image_width)

        # Scale Difference Images To Between 0 and 1
        difference_image = scale_difference_image(difference_image)

        # Convert These To Colours
        delta_f_colourmap = cm.get_cmap('jet')
        difference_colourmap = cm.get_cmap('bwr')

        condition_1_activity_image = delta_f_colourmap(condition_1_activity_image)
        condition_2_activity_image = delta_f_colourmap(condition_2_activity_image)
        condition_3_activity_image = delta_f_colourmap(condition_3_activity_image)
        difference_image = difference_colourmap(difference_image)

        # Flatten Arrays
        condition_1_activity_image = np.ndarray.reshape(condition_1_activity_image, (image_height * image_width, 4))
        condition_2_activity_image = np.ndarray.reshape(condition_2_activity_image, (image_height * image_width, 4))
        condition_3_activity_image = np.ndarray.reshape(condition_3_activity_image, (image_height * image_width, 4))
        difference_image = np.ndarray.reshape(difference_image, (image_height * image_width, 4))

        # Add Outlines
        condition_1_activity_image[atlas_indicies] = (0, 0, 0, 1)
        condition_2_activity_image[atlas_indicies] = (0, 0, 0, 1)
        condition_3_activity_image[atlas_indicies] = (0, 0, 0, 1)
        difference_image[atlas_indicies] = (0, 0, 0, 1)

        # Put Back Into Squares
        condition_1_activity_image = np.ndarray.reshape(condition_1_activity_image, (image_height, image_width, 4))
        condition_2_activity_image = np.ndarray.reshape(condition_2_activity_image, (image_height, image_width, 4))
        condition_3_activity_image = np.ndarray.reshape(condition_3_activity_image, (image_height, image_width, 4))
        difference_image = np.ndarray.reshape(difference_image, (image_height, image_width, 4))

        # Plot These Images
        condition_1_activity_axis.imshow(condition_1_activity_image)
        condition_2_activity_axis.imshow(condition_2_activity_image)
        condition_3_activity_axis.imshow(condition_3_activity_image)
        difference_axis.imshow(difference_image)

        # Remove Axes
        condition_1_activity_axis.axis('off')
        condition_2_activity_axis.axis('off')
        condition_3_activity_axis.axis('off')
        difference_axis.axis('off')

        # Set Title
        condition_1_activity_axis.set_title(plot_titles[0] + "_Raw_Activity")
        condition_2_activity_axis.set_title(plot_titles[1] + "_Raw_Activity")
        condition_3_activity_axis.set_title(plot_titles[2] + "_Raw_Activity")
        difference_axis.set_title("Difference")

        # Add Time Text
        time_text = str(timepoint_list[timepoint]) + "ms"
        x_pos = 30
        y_pos = 75
        condition_1_activity_axis.text(x_pos, y_pos, time_text, color='White')
        condition_2_activity_axis.text(x_pos, y_pos, time_text, color='White')
        condition_3_activity_axis.text(x_pos, y_pos, time_text, color='White')
        difference_axis.text(x_pos, y_pos, time_text)

        plt.draw()
        plt.savefig(os.path.join(full_save_path, str(timepoint_count).zfill(3) + ".png"))
        plt.pause(0.1)
        plt.clf()

         # plt.show()
        timepoint_count += 1





def create_single_mouse_comparison_video(base_directory, tensor_filenames, trial_start, trial_stop, plot_titles, save_directory):

    # Get Region Boundaries
    masked_atlas, atlas_indicies = add_region_boundaries(base_directory)

    # Load Tensors
    condition_1_activity_tensor = np.load(os.path.join(base_directory, "Activity_Tensors", tensor_filenames[0] + "_Activity_Tensor.npy"))
    condition_1_activity_mean = np.mean(condition_1_activity_tensor, axis=0)
    condition_1_activity_tensor = reconstruct_images_from_activity(condition_1_activity_mean, base_directory)

    condition_2_activity_tensor = np.load(os.path.join(base_directory, "Activity_Tensors", tensor_filenames[1] + "_Activity_Tensor.npy"))
    condition_2_activity_mean = np.mean(condition_2_activity_tensor, axis=0)
    condition_2_activity_tensor = reconstruct_images_from_activity(condition_2_activity_mean, base_directory)

    # Plot
    number_of_timepoints = np.shape(condition_1_activity_tensor)[0]
    number_of_columns = 3
    number_of_rows = 1
    figure_1 = plt.figure(constrained_layout=True, figsize=(80, 60))

    timepoint_count = 0
    timepoint_list = list(range(trial_start, trial_stop))
    timepoint_list = np.multiply(timepoint_list, 36)

    # Get Full Save Path
    full_save_path = os.path.join(base_directory, save_directory)
    if not os.path.exists(full_save_path):
        os.mkdir(full_save_path)

    for timepoint in range(number_of_timepoints):
        print(timepoint)
        grid_spec_1 = gridspec.GridSpec(ncols=number_of_columns, nrows=number_of_rows, figure=figure_1)

        # Add Axes
        condition_1_activity_axis = figure_1.add_subplot(grid_spec_1[0, 0])
        condition_2_activity_axis = figure_1.add_subplot(grid_spec_1[0, 1])
        difference_axis = figure_1.add_subplot(grid_spec_1[0, 2])

        # Select Images
        condition_1_activity_image = condition_1_activity_tensor[timepoint]
        condition_2_activity_image = condition_2_activity_tensor[timepoint]
        difference_image = np.diff([condition_1_activity_image, condition_2_activity_image], axis=0)[0]

        image_height = np.shape(condition_1_activity_image)[0]
        image_width = np.shape(condition_1_activity_image)[1]

        # Gaussian Smoothing
        condition_1_activity_image = gaussian_smooth_image(condition_1_activity_image, image_height, image_width)
        condition_2_activity_image = gaussian_smooth_image(condition_2_activity_image, image_height, image_width)
        difference_image = gaussian_smooth_image(difference_image, image_height, image_width)

        # Scale Difference Images To Between 0 and 1
        difference_image = scale_difference_image(difference_image)

        # Convert These To Colours
        delta_f_colourmap = cm.get_cmap('jet')
        difference_colourmap = cm.get_cmap('bwr')

        condition_1_activity_image = delta_f_colourmap(condition_1_activity_image)
        condition_2_activity_image = delta_f_colourmap(condition_2_activity_image)
        difference_image = difference_colourmap(difference_image)

        # Flatten Arrays
        condition_1_activity_image = np.ndarray.reshape(condition_1_activity_image, (image_height * image_width, 4))
        condition_2_activity_image = np.ndarray.reshape(condition_2_activity_image, (image_height * image_width, 4))
        difference_image = np.ndarray.reshape(difference_image, (image_height * image_width, 4))

        # Add Outlines
        condition_1_activity_image[atlas_indicies] = (0, 0, 0, 1)
        condition_2_activity_image[atlas_indicies] = (0, 0, 0, 1)
        difference_image[atlas_indicies] = (0, 0, 0, 1)

        # Put Back Into Squares
        condition_1_activity_image = np.ndarray.reshape(condition_1_activity_image, (image_height, image_width, 4))
        condition_2_activity_image = np.ndarray.reshape(condition_2_activity_image, (image_height, image_width, 4))
        difference_image = np.ndarray.reshape(difference_image, (image_height, image_width, 4))

        # Plot These Images
        condition_1_activity_axis.imshow(condition_1_activity_image)
        condition_2_activity_axis.imshow(condition_2_activity_image)
        difference_axis.imshow(difference_image)

        # Remove Axes
        condition_1_activity_axis.axis('off')
        condition_2_activity_axis.axis('off')
        difference_axis.axis('off')

        # Set Title
        condition_1_activity_axis.set_title(plot_titles[0] + "_Raw_Activity")
        condition_2_activity_axis.set_title(plot_titles[1] + "_Raw_Activity")
        difference_axis.set_title("Difference")

        # Add Time Text
        time_text = str(timepoint_list[timepoint]) + "ms"
        x_pos = 30
        y_pos = 75
        condition_1_activity_axis.text(x_pos, y_pos, time_text, color='White')
        condition_2_activity_axis.text(x_pos, y_pos, time_text, color='White')
        difference_axis.text(x_pos, y_pos, time_text)

        plt.draw()
        plt.savefig(os.path.join(full_save_path, str(timepoint_count).zfill(3) + ".png"))
        plt.pause(0.1)
        plt.clf()

         # plt.show()
        timepoint_count += 1






def create_single_mouse_comparison_video_with_correction(base_directory, condition_1_tensor_filename, condition_2_tensor_filename, trial_start, trial_stop, plot_titles, save_directory):

    # Get Region Boundaries
    masked_atlas, atlas_indicies = add_region_boundaries(base_directory)

    # Load Tensors
    condition_1_activity_tensor = np.load(os.path.join(base_directory, "Activity_Tensors", condition_1_tensor_filename + "_Activity_Tensor.npy"))
    condition_1_activity_mean = np.mean(condition_1_activity_tensor, axis=0)
    condition_1_activity_tensor = reconstruct_images_from_activity(condition_1_activity_mean, base_directory)

    condition_1_predicted_tensor = np.load(os.path.join(base_directory, "Activity_Tensors", condition_1_tensor_filename + "_Predicted_Tensor.npy"))
    condition_1_predicted_mean = np.mean(condition_1_predicted_tensor, axis=0)
    condition_1_predicted_tensor = reconstruct_images_from_activity(condition_1_predicted_mean, base_directory)

    condition_1_corrected_tensor = np.load(os.path.join(base_directory, "Activity_Tensors", condition_1_tensor_filename + "_Corrected_Tensor.npy"))
    condition_1_corrected_mean = np.mean(condition_1_corrected_tensor, axis=0)
    condition_1_corrected_tensor = reconstruct_images_from_activity(condition_1_corrected_mean, base_directory)

    condition_2_activity_tensor  = np.load(os.path.join(base_directory, "Activity_Tensors", condition_2_tensor_filename + "_Activity_Tensor.npy"))
    condition_2_activity_mean = np.mean(condition_2_activity_tensor, axis=0)
    condition_2_activity_tensor = reconstruct_images_from_activity(condition_2_activity_mean, base_directory)

    condition_2_predicted_tensor = np.load(os.path.join(base_directory, "Activity_Tensors", condition_2_tensor_filename + "_Predicted_Tensor.npy"))
    condition_2_predicted_mean = np.mean(condition_2_predicted_tensor, axis=0)
    condition_2_predicted_tensor = reconstruct_images_from_activity(condition_2_predicted_mean, base_directory)

    condition_2_corrected_tensor = np.load(os.path.join(base_directory, "Activity_Tensors", condition_2_tensor_filename + "_Corrected_Tensor.npy"))
    condition_2_corrected_mean = np.mean(condition_2_corrected_tensor, axis=0)
    condition_2_corrected_tensor = reconstruct_images_from_activity(condition_2_corrected_mean, base_directory)


    # Plot
    number_of_timepoints = np.shape(condition_1_activity_tensor)[0]
    number_of_columns = 4
    number_of_rows = 2
    figure_1 = plt.figure(constrained_layout=True, figsize=(80, 60))

    timepoint_count = 0
    timepoint_list = list(range(trial_start, trial_stop))
    timepoint_list = np.multiply(timepoint_list, 36)

    # Get Full Save Path
    full_save_path = os.path.join(base_directory, save_directory)
    if not os.path.exists(full_save_path):
        os.mkdir(full_save_path)

    for timepoint in range(number_of_timepoints):
        print(timepoint)
        grid_spec_1 = gridspec.GridSpec(ncols=number_of_columns, nrows=number_of_rows, figure=figure_1)

        # Add Axes
        condition_1_activity_axis  = figure_1.add_subplot(grid_spec_1[0, 0])
        condition_1_predicted_axis = figure_1.add_subplot(grid_spec_1[0, 1])
        condition_1_corrected_axis = figure_1.add_subplot(grid_spec_1[0, 2])

        condition_2_activity_axis  = figure_1.add_subplot(grid_spec_1[1, 0])
        condition_2_predicted_axis = figure_1.add_subplot(grid_spec_1[1, 1])
        condition_2_corrected_axis = figure_1.add_subplot(grid_spec_1[1, 2])

        difference_axis = figure_1.add_subplot(grid_spec_1[0, 3])

        # Select Images
        condition_1_activity_image  = condition_1_activity_tensor[timepoint]
        condition_1_predicted_image = condition_1_predicted_tensor[timepoint]
        condition_1_corrected_image = condition_1_corrected_tensor[timepoint]

        condition_2_activity_image  = condition_2_activity_tensor[timepoint]
        condition_2_predicted_image = condition_2_predicted_tensor[timepoint]
        condition_2_corrected_image = condition_2_corrected_tensor[timepoint]

        difference_image = np.diff([condition_1_corrected_image, condition_2_corrected_image], axis=0)[0]


        image_height = np.shape(condition_1_activity_image)[0]
        image_width = np.shape(condition_1_activity_image)[1]

        # Gaussian Smoothing
        #control_difference_image = gaussian_smooth_image(control_difference_image, image_height, image_width)
        #mutant_difference_image = gaussian_smooth_image(mutant_difference_image, image_height, image_width)

        # Scale Difference Images To Between 0 and 1
        difference_image = scale_difference_image(difference_image)

        # Convert These To Colours
        delta_f_colourmap = cm.get_cmap('jet')
        difference_colourmap = cm.get_cmap('bwr')

        condition_1_activity_image = delta_f_colourmap(condition_1_activity_image)
        condition_1_predicted_image = delta_f_colourmap(condition_1_predicted_image)
        condition_1_corrected_image = delta_f_colourmap(condition_1_corrected_image)

        condition_2_activity_image = delta_f_colourmap(condition_2_activity_image)
        condition_2_predicted_image = delta_f_colourmap(condition_2_predicted_image)
        condition_2_corrected_image = delta_f_colourmap(condition_2_corrected_image)

        difference_image = difference_colourmap(difference_image)


        # Flatten Arrays
        condition_1_activity_image = np.ndarray.reshape(condition_1_activity_image, (image_height * image_width, 4))
        condition_1_predicted_image = np.ndarray.reshape(condition_1_predicted_image, (image_height * image_width, 4))
        condition_1_corrected_image = np.ndarray.reshape(condition_1_corrected_image, (image_height * image_width, 4))

        condition_2_activity_image = np.ndarray.reshape(condition_2_activity_image, (image_height * image_width, 4))
        condition_2_predicted_image = np.ndarray.reshape(condition_2_predicted_image, (image_height * image_width, 4))
        condition_2_corrected_image = np.ndarray.reshape(condition_2_corrected_image, (image_height * image_width, 4))

        difference_image = np.ndarray.reshape(difference_image, (image_height * image_width, 4))


        # Add Outlines
        condition_1_activity_image[atlas_indicies] = (0, 0, 0, 1)
        condition_1_predicted_image[atlas_indicies] = (0, 0, 0, 1)
        condition_1_corrected_image[atlas_indicies] = (0, 0, 0, 1)

        condition_2_activity_image[atlas_indicies] = (0, 0, 0, 1)
        condition_2_predicted_image[atlas_indicies] = (0, 0, 0, 1)
        condition_2_corrected_image[atlas_indicies] = (0, 0, 0, 1)

        difference_image[atlas_indicies] = (0, 0, 0, 1)

        # Put Back Into Squares
        condition_1_activity_image = np.ndarray.reshape(condition_1_activity_image, (image_height, image_width, 4))
        condition_1_predicted_image = np.ndarray.reshape(condition_1_predicted_image, (image_height, image_width, 4))
        condition_1_corrected_image = np.ndarray.reshape(condition_1_corrected_image, (image_height, image_width, 4))

        condition_2_activity_image = np.ndarray.reshape(condition_2_activity_image, (image_height, image_width, 4))
        condition_2_predicted_image = np.ndarray.reshape(condition_2_predicted_image, (image_height, image_width, 4))
        condition_2_corrected_image = np.ndarray.reshape(condition_2_corrected_image, (image_height, image_width, 4))

        difference_image = np.ndarray.reshape(difference_image, (image_height, image_width, 4))


        # Plot These Images
        condition_1_activity_axis.imshow(condition_1_activity_image)
        condition_1_predicted_axis.imshow(condition_1_predicted_image)
        condition_1_corrected_axis.imshow(condition_1_corrected_image)

        condition_2_activity_axis.imshow(condition_2_activity_image)
        condition_2_predicted_axis.imshow(condition_2_predicted_image)
        condition_2_corrected_axis.imshow(condition_2_corrected_image)

        difference_axis.imshow(difference_image)

        # Remove Axes
        condition_1_activity_axis.axis('off')
        condition_1_predicted_axis.axis('off')
        condition_1_corrected_axis.axis('off')
        condition_2_activity_axis.axis('off')
        condition_2_predicted_axis.axis('off')
        condition_2_corrected_axis.axis('off')
        difference_axis.axis('off')

        # Set Title
        condition_1_activity_axis.set_title(plot_titles[0] + "_Raw_Activity")
        condition_1_predicted_axis.set_title(plot_titles[0] + "_Predicted_Activity")
        condition_1_corrected_axis.set_title(plot_titles[0] + "_Corrected_Activity")

        condition_2_activity_axis.set_title(plot_titles[1] + "_Raw_Activity")
        condition_2_predicted_axis.set_title(plot_titles[1] + "_Predicted_Activity")
        condition_2_corrected_axis.set_title(plot_titles[1] + "_Corrected_Activity")

        difference_axis.set_title("Corrected Difference")


        #horizontalalignment='center', verticalalignment='center', transform=control_visual_axis.transAxes
        time_text = str(timepoint_list[timepoint]) + "ms"
        x_pos = 30
        y_pos = 75
        condition_1_activity_axis.text(x_pos, y_pos, time_text, color='White')
        condition_1_predicted_axis.text(x_pos, y_pos, time_text, color='White')
        condition_1_corrected_axis.text(x_pos, y_pos, time_text, color='White')
        condition_2_activity_axis.text(x_pos, y_pos, time_text, color='White')
        condition_2_predicted_axis.text(x_pos, y_pos, time_text, color='White')
        condition_2_corrected_axis.text(x_pos, y_pos, time_text, color='White')
        difference_axis.text(x_pos, y_pos, time_text)


        plt.draw()
        plt.savefig(os.path.join(full_save_path, str(timepoint_count).zfill(3) + ".png"))
        plt.pause(0.1)
        plt.clf()

        #plt.show()
        timepoint_count += 1

