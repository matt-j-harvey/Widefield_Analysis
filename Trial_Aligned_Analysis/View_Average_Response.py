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

    plt.imshow(atlas_region_mapping)
    plt.show()

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
    plt.imshow(masked_atlas)

    #plt.savefig(os.path.join(base_directory, "Pixel_Region_Assignmnet.png"))
    plt.show()


    mask_indicies = np.nonzero(np.ndarray.flatten(masked_atlas))


    fromindex = np.zeros((image_height * image_width))
    for index in mask_indicies:
        fromindex[index] = 1

    fromindex = np.ndarray.reshape(fromindex, (image_height, image_width))
    plt.imshow(fromindex)
    plt.show()

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


def plot_cross_mouse_average_response(controls, mutants,
                                  control_visual_responses,
                                  control_odour_responses,
                                  mutant_visual_responses,
                                  mutant_odour_responses,
                                  plot_save_directory):

    # Get Region Boundaries
    masked_atlas, atlas_indicies = add_region_boundaries(controls[0])

    # Convert Lists to Arrays
    control_visual_responses = np.array(control_visual_responses)
    control_odour_responses = np.array(control_odour_responses)
    mutant_visual_responses = np.array(mutant_visual_responses)
    mutant_odour_responses = np.array(mutant_odour_responses)


    # Get Average Responses
    control_mean_visual_response = np.mean(control_visual_responses, axis=0)
    control_mean_odour_response = np.mean(control_odour_responses, axis=0)

    mutant_mean_visual_response = np.mean(mutant_visual_responses, axis=0)
    mutant_mean_odour_response = np.mean(mutant_odour_responses, axis=0)


    # Reconstruct Average Responses Into Images
    control_visual_response_reconstructed = reconstruct_images_from_activity(control_mean_visual_response, controls[0])
    control_odour_response_reconstructed = reconstruct_images_from_activity(control_mean_odour_response, controls[0])

    mutant_visual_response_reconstructed = reconstruct_images_from_activity(mutant_mean_visual_response, controls[0])
    mutant_odour_response_reconstructed = reconstruct_images_from_activity(mutant_mean_odour_response, controls[0])


    # Plot
    number_of_timepoints = np.shape(control_mean_visual_response)[0]
    number_of_columns = 3
    number_of_rows = 2
    figure_1 = plt.figure(constrained_layout=True)

    timepoint_count = 0
    timepoint_list = list(range(trial_start, trial_stop))
    timepoint_list = np.multiply(timepoint_list, 36)

    for timepoint in range(number_of_timepoints):
        grid_spec_1 = gridspec.GridSpec(ncols=number_of_columns, nrows=number_of_rows, figure=figure_1)

        # Add Axes
        control_visual_axis     = figure_1.add_subplot(grid_spec_1[0, 0])
        control_odour_axis      = figure_1.add_subplot(grid_spec_1[0, 1])
        control_difference_axis = figure_1.add_subplot(grid_spec_1[0, 2])

        mutant_visual_axis     = figure_1.add_subplot(grid_spec_1[1, 0])
        mutant_odour_axis      = figure_1.add_subplot(grid_spec_1[1, 1])
        mutant_difference_axis = figure_1.add_subplot(grid_spec_1[1, 2])

        # Select Images
        control_visual_response_image = control_visual_response_reconstructed[timepoint]
        control_odour_response_image = control_odour_response_reconstructed[timepoint]
        control_difference_image = np.diff([control_visual_response_image, control_odour_response_image], axis=0)[0]
        print("First Control Diff Image", np.shape(control_difference_image))

        mutant_visual_response_image = mutant_visual_response_reconstructed[timepoint]
        mutant_odour_response_image = mutant_odour_response_reconstructed[timepoint]
        mutant_difference_image = np.diff([mutant_visual_response_image, mutant_odour_response_image], axis=0)[0]

        image_height = np.shape(control_visual_response_image)[0]
        image_width = np.shape(control_visual_response_image)[1]

        # Gaussian Smoothing
        control_difference_image = gaussian_smooth_image(control_difference_image, image_height, image_width)
        mutant_difference_image = gaussian_smooth_image(mutant_difference_image, image_height, image_width)

        # Scale Difference Images To Between 0 and 1
        control_difference_image = scale_difference_image(control_difference_image)
        mutant_difference_image = scale_difference_image(mutant_difference_image)

        # Convert These To Colours
        delta_f_colourmap = cm.get_cmap('jet')
        difference_colourmap = cm.get_cmap('bwr')

        control_visual_response_image = delta_f_colourmap(control_visual_response_image)
        control_odour_response_image = delta_f_colourmap(control_odour_response_image)
        control_difference_image = difference_colourmap(control_difference_image)

        mutant_visual_response_image = delta_f_colourmap(mutant_visual_response_image)
        mutant_odour_response_image = delta_f_colourmap(mutant_odour_response_image)
        mutant_difference_image = difference_colourmap(mutant_difference_image)

        print("COntrol Diff Image", np.shape(control_difference_image))

        # Add Outlines

        control_visual_response_image = np.ndarray.reshape(control_visual_response_image, (image_height * image_width, 4))
        control_odour_response_image = np.ndarray.reshape(control_odour_response_image, (image_height * image_width, 4))
        control_difference_image = np.ndarray.reshape(control_difference_image, (image_height * image_width, 4))

        mutant_visual_response_image = np.ndarray.reshape(mutant_visual_response_image, (image_height * image_width, 4))
        mutant_odour_response_image = np.ndarray.reshape(mutant_odour_response_image, (image_height * image_width, 4))
        mutant_difference_image = np.ndarray.reshape(mutant_difference_image, (image_height * image_width, 4))

        for index in atlas_indicies:
            control_visual_response_image[index] = (0, 0, 0, 1)
            control_odour_response_image[index] = (0, 0, 0, 1)
            control_difference_image[index] = (0, 0, 0, 1)

            mutant_visual_response_image[index] = (0, 0, 0, 1)
            mutant_odour_response_image[index] = (0, 0, 0, 1)
            mutant_difference_image[index] = (0, 0, 0, 1)

        control_visual_response_image = np.ndarray.reshape(control_visual_response_image, (image_height, image_width, 4))
        control_odour_response_image = np.ndarray.reshape(control_odour_response_image, (image_height, image_width, 4))
        control_difference_image = np.ndarray.reshape(control_difference_image, (image_height, image_width, 4))

        mutant_visual_response_image = np.ndarray.reshape(mutant_visual_response_image, (image_height, image_width, 4))
        mutant_odour_response_image = np.ndarray.reshape(mutant_odour_response_image, (image_height, image_width, 4))
        mutant_difference_image = np.ndarray.reshape(mutant_difference_image, (image_height, image_width, 4))

        # Plot These Images
        control_visual_axis.imshow(control_visual_response_image)
        control_odour_axis.imshow(control_odour_response_image)
        control_difference_axis.imshow(control_difference_image)

        mutant_visual_axis.imshow(mutant_visual_response_image)
        mutant_odour_axis.imshow(mutant_odour_response_image)
        mutant_difference_axis.imshow(mutant_difference_image)

        # Remove Axes
        control_visual_axis.axis('off')
        control_odour_axis.axis('off')
        control_difference_axis.axis('off')
        mutant_visual_axis.axis('off')
        mutant_odour_axis.axis('off')
        mutant_difference_axis.axis('off')

        # Set Title
        control_visual_axis.set_title("Control Visual Context")
        control_odour_axis.set_title("Control Odour Context")
        control_difference_axis.set_title("Control Difference")

        mutant_visual_axis.set_title("Mutant Visual Context")
        mutant_odour_axis.set_title("Mutant Odour Context")
        mutant_difference_axis.set_title("Mutant Difference")

        #horizontalalignment='center', verticalalignment='center', transform=control_visual_axis.transAxes
        time_text = str(timepoint_list[timepoint]) + "ms"
        x_pos = 30
        y_pos = 75
        control_visual_axis.text(       x_pos, y_pos, time_text, color='White')
        control_odour_axis.text(        x_pos, y_pos, time_text, color='White')
        control_difference_axis.text(   x_pos, y_pos, time_text)
        mutant_visual_axis.text(        x_pos, y_pos, time_text, color='White')
        mutant_odour_axis.text(         x_pos, y_pos, time_text, color='White')
        mutant_difference_axis.text(    x_pos, y_pos, time_text)

        plt.draw()
        plt.savefig(plot_save_directory + str(timepoint_count).zfill(3) + ".png")
        plt.pause(0.1)
        plt.clf()

        timepoint_count += 1


controls = ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN78.1A/2020_12_09_Switching_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN78.1D/2020_11_29_Switching_Imaging"]

mutants = [ "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK12.1F/2021_09_22_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN71.2A/2020_12_17_Switching_Imaging"]


"""
trial_start = -10
trial_stop = 40
"""

trial_start = -65
trial_stop = -4

root_directory = "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/Mean_Responses"

#visual_onsets_file = "visual_context_stable_vis_2_frame_onsets_Matched.npy"
#odour_onsets_file = "odour_context_stable_vis_2_frame_onsets_Matched.npy"
#plot_save_directory = "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/Mean_Responses/Matched_Images/"

visual_onsets_file_list = ["visual_context_stable_vis_1_frame_onsets.npy", "visual_context_stable_vis_2_frame_onsets.npy"]
odour_onsets_file_list = ["odour_context_stable_vis_1_frame_onsets.npy", "odour_context_stable_vis_2_frame_onsets.npy"]
plot_save_directory = "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/Baseline_Responses/Image/"

# Get Control Responses
control_visual_response_save_directory = os.path.join(root_directory, "Controls_Visual_Pre_Average.npy")
control_odour_response_save_directory  = os.path.join(root_directory, "Controls_Odour_Pre_Average.npy")

control_visual_average_responses = get_average_response(controls, visual_onsets_file_list, trial_start, trial_stop, control_visual_response_save_directory)
control_odour_average_responses  = get_average_response(controls, odour_onsets_file_list,  trial_start, trial_stop, control_odour_response_save_directory)


# Get Mutant Responses
mutant_visual_response_save_directory = os.path.join(root_directory, "Mutants_Visual_Pre_Average.npy")
mutant_odour_response_save_directory  = os.path.join(root_directory, "Mutants_Odour_Pre_Average.npy")

mutant_visual_average_responses = get_average_response(mutants, visual_onsets_file_list, trial_start, trial_stop, mutant_visual_response_save_directory)
mutant_odour_average_responses  = get_average_response(mutants, odour_onsets_file_list,  trial_start, trial_stop, mutant_odour_response_save_directory)


# Load Responses
control_visual_responses = np.load(control_visual_response_save_directory)
control_odour_responses = np.load(control_odour_response_save_directory)

mutant_visual_responses = np.load(mutant_visual_response_save_directory)
mutant_odour_responses = np.load(mutant_odour_response_save_directory)




plot_cross_mouse_average_response(controls,
                                  mutants,
                                  control_visual_responses,
                                  control_odour_responses,
                                  mutant_visual_responses,
                                  mutant_odour_responses,
                                  plot_save_directory)
