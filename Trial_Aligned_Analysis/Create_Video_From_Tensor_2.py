import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
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
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions


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



def reconstruct_images_from_activity(activity_matrix, indicies, image_height, image_width):

    number_of_frames = np.shape(activity_matrix)[0]
    image_matrix = np.zeros([number_of_frames, image_height, image_width])
    for frame_index in range(number_of_frames):
        activity_vector = activity_matrix[frame_index]
        image = Widefield_General_Functions.create_image_from_data(activity_vector, indicies, image_height, image_width)
        image_matrix[frame_index] = image

    return image_matrix

def normalise_trace(trace):

    # Subtract Min
    trace = np.subtract(trace, np.min(trace))

    # Divide Max
    trace = np.divide(trace, np.max(trace))

    return trace


def jointly_scale_trace(trace_list):

    # Get Group Min
    min_value_list = []
    for trace in trace_list:
        min_value_list.append(np.min(trace))
    group_min = np.min(min_value_list)

    # Subtract Group Min From Trace
    min_subtracted_trace_list = []
    for trace in trace_list:
        subtracted_trace = np.subtract(trace, group_min)
        min_subtracted_trace_list.append(subtracted_trace)

    # Get Group Max
    max_value_list = []
    for trace in min_subtracted_trace_list:
        max_value_list.append(np.max(trace))
    group_max = np.max(max_value_list)

    # Divide By Group max
    normalised_trace_list = []
    for trace in min_subtracted_trace_list:
        normalised_trace = np.divide(trace, group_max)
        normalised_trace_list.append(normalised_trace)

    return normalised_trace_list


def remove_axis_border(axis):
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    #axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)




def jointly_scale_behaviour_traces(behaviour_dict_list, selected_trace_list, number_of_traces, number_of_conditions):

    # Each Ai Trace then Each Condition
    jointly_scaled_trace_list = []

    # Jointly Normalise Traces
    for trace_index in range(number_of_traces):

        # Get Selected Trace Name
        selected_trace = selected_trace_list[trace_index]

        # Get List of Unscaled Traces
        unscaled_trace_list = []
        for condition_index in range(number_of_conditions):

            # Get Trace From Dict
            condition_trace_data = behaviour_dict_list[condition_index][selected_trace]
            condition_trace_data = np.mean(condition_trace_data, axis=0)
            unscaled_trace_list.append(condition_trace_data)

        # Jointly Scale These
        scaled_trace_list = jointly_scale_trace(unscaled_trace_list)
        jointly_scaled_trace_list.append(scaled_trace_list)

    jointly_scaled_trace_list = np.array(jointly_scaled_trace_list)

    return jointly_scaled_trace_list


def load_metadata(base_directory):

    delta_f_file = tables.open_file(os.path.join(base_directory, "Delta_F.h5"))
    pixel_baseline_list = delta_f_file.root.pixel_baseline_list
    pixel_maximum_list = delta_f_file.root.pixel_maximum_list

    pixel_baseline_list = np.array(pixel_baseline_list)
    pixel_maximum_list = np.array(pixel_maximum_list)

    pixel_baseline_list = np.ndarray.flatten(pixel_baseline_list)
    pixel_maximum_list = np.ndarray.flatten(pixel_maximum_list)


    return pixel_baseline_list, pixel_maximum_list


def get_mussal_cmap():
    cmap = LinearSegmentedColormap.from_list('mycmap', [

        (0, 0.87, 0.9, 1),
        (0, 0, 1, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 1, 0, 1),
        
    ])

    return cmap



def create_activity_video(indicies, image_height, image_width, mean_activity_tensors, trial_start, trial_stop, plot_titles, save_directory, behaviour_dict_list, selected_behaviour_traces, timestep=36, difference_conditions=False, cmap_type='positive'):

    # Check Save Directory Exists
    print("Save Directory", save_directory)
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Get Region Boundaries
    #masked_atlas, atlas_indicies = add_region_boundaries(base_directory)

    # Load Number Of Condition
    number_of_conditions = len(mean_activity_tensors)
    print("Number of Conditions: ", number_of_conditions)

    # Construct Images
    reconstructed_activity_tensors = []
    for condition_index in range(number_of_conditions):
        condition_activity_tensor = reconstruct_images_from_activity(mean_activity_tensors[condition_index], indicies, image_height, image_width)
        reconstructed_activity_tensors.append(condition_activity_tensor)
    reconstructed_activity_tensors = np.array(reconstructed_activity_tensors)


    # Create Figure
    number_of_timepoints = trial_stop - trial_start
    print("Numbe of timepoints", number_of_timepoints)
    number_of_columns = number_of_conditions + 1
    number_of_rows = 2
    figure_1 = plt.figure(constrained_layout=True, figsize=(80, 60))
    grid_spec_1 = gridspec.GridSpec(ncols=number_of_columns, nrows=number_of_rows, figure=figure_1)

    # Plot For Each Timepoint
    timepoint_count = 0
    timepoint_list = list(range(trial_start, trial_stop))
    timepoint_list = np.multiply(timepoint_list, timestep)

    # Create Colourmaps
    delta_f_vmax = np.percentile(mean_activity_tensors, q=99)

    if cmap_type == 'positive':
        delta_f_vmin = 0
        diff_v_max = delta_f_vmax * 0.5
        diff_v_min = -1 * delta_f_vmax * 0.5
        print("Delta F Vmax", delta_f_vmax)
        print("Delta F Vmin", delta_f_vmin)
        delta_f_colourmap = cm.ScalarMappable(cmap=cm.get_cmap('inferno'), norm=Normalize(vmin=delta_f_vmin, vmax=delta_f_vmax))
        difference_colourmap = cm.ScalarMappable(cmap=cm.get_cmap('bwr'), norm=Normalize(vmin=diff_v_min, vmax=diff_v_max))

    elif cmap_type == 'zero_centered':
        delta_f_vmin = -1 * delta_f_vmax
        delta_f_colourmap = cm.ScalarMappable(cmap=get_mussal_cmap(), norm=Normalize(vmin=delta_f_vmin, vmax=delta_f_vmax))




    # Get X Values
    x_values = list(range(trial_start * timestep, trial_stop * timestep, timestep))

    # Get Number of Behaviour Traces
    selected_trace_list = list(behaviour_dict_list[0].keys())
    number_of_traces = len(selected_trace_list)

    # Select Behaviour Trace Colours
    behaviour_trace_colour_list = []
    behaviour_colour_map = cm.get_cmap('jet')
    for trace_index in range(number_of_traces):
        trace_index = float(trace_index) / number_of_traces
        trace_colour = behaviour_colour_map(trace_index)
        behaviour_trace_colour_list.append(trace_colour)

    # Set Behaviour Trace Offset
    trace_offset = 1.5

    # Get Jointly Scaled Beahviour Traces
    scaled_behaviour_trace_list = jointly_scale_behaviour_traces(behaviour_dict_list, selected_behaviour_traces, number_of_traces, number_of_conditions)

    # Plot Each Timepoint
    for timepoint in range(number_of_timepoints):

        print(timepoint)
        time_text = str(timepoint_list[timepoint]) + "ms"
        x_pos = 30
        y_pos = 75

        # Add Axes
        for condition_index in range(number_of_conditions):
            condition_activity_axis = figure_1.add_subplot(grid_spec_1[0, condition_index])
            condition_behaviour_axis = figure_1.add_subplot(grid_spec_1[1, condition_index])

            # Select Images
            condition_activity_image = reconstructed_activity_tensors[condition_index, timepoint]

            # Gaussian Smoothing
            condition_activity_image = ndimage.gaussian_filter(condition_activity_image, sigma=1)

            # Add Colourmap
            condition_activity_image = delta_f_colourmap.to_rgba(condition_activity_image)


            # Add Outlines
            """
            condition_activity_image = np.ndarray.reshape(condition_activity_image, (image_height * image_width, 4))
            condition_activity_image[atlas_indicies] = (0, 0, 0, 1)
            condition_activity_image = np.ndarray.reshape(condition_activity_image, (image_height, image_width, 4))
            """

            # Plot Image
            condition_activity_axis.imshow(condition_activity_image)

            # Remove Axis
            condition_activity_axis.axis('off')

            # Set Title
            condition_activity_axis.set_title(plot_titles[condition_index] + "_Raw_Activity")

            # Add Time Text
            condition_activity_axis.text(x_pos, y_pos, time_text, color='White')

            for trace_index in range(number_of_traces):

                # Load Scaled Trace
                scaled_trace = scaled_behaviour_trace_list[trace_index, condition_index]

                # Add Offset
                offset = trace_index * trace_offset
                scaled_trace = np.add(scaled_trace, offset)

                # Get Trace Colour
                trace_colour = behaviour_trace_colour_list[trace_index]

                # Plot Traces
                condition_behaviour_axis.plot(x_values, scaled_trace, c=trace_colour)

            # Remove Behaviour Axis Y Values
            condition_behaviour_axis.get_yaxis().set_visible(False)

            # Remove Behaviour Axis Boxes
            remove_axis_border(condition_behaviour_axis)

            # PLace Vline At Trial Offsets
            condition_behaviour_axis.axvline(x=0, ymin=0, ymax=1, c='k', linestyle=(0, (5,5)))

            current_time = 0 + (trial_start * 36) + (timepoint * 36)
            condition_behaviour_axis.axvline(x=current_time, ymin=0, ymax=1, c='b')

        # Plot Beahviour Lengend
        legend_axis = figure_1.add_subplot(grid_spec_1[1, number_of_conditions])
        patch_list = []

        for trace_index in range(number_of_traces):
            trace_name = selected_trace_list[trace_index]
            trace_colour = behaviour_trace_colour_list[trace_index]

            patch = mpatches.Patch(color=trace_colour, label=trace_name)
            patch_list.append(patch)

        patch_list.reverse()
        legend_axis.legend(handles=patch_list, fontsize='xx-large', loc='center')
        legend_axis.axis('off')

        # Plot Difference Map
        if difference_conditions != False:

            difference_axis = figure_1.add_subplot(grid_spec_1[0, number_of_conditions])

            condition_1_activity_frame = reconstructed_activity_tensors[difference_conditions[0], timepoint]
            condition_2_activity_frame = reconstructed_activity_tensors[difference_conditions[1], timepoint]
            difference_activity_map = np.subtract(condition_1_activity_frame, condition_2_activity_frame)
            difference_activity_map = ndimage.gaussian_filter(difference_activity_map, sigma=1)
            difference_activity_map = difference_colourmap.to_rgba(difference_activity_map)

            difference_axis.imshow(difference_activity_map)



        plt.draw()
        plt.savefig(os.path.join(save_directory, str(timepoint_count).zfill(3) + ".png"))
        plt.pause(0.1)
        plt.clf()

        # plt.show()
        timepoint_count += 1






def create_activity_video_over_learning(base_directory_list, mean_activity_tensors, trial_start, trial_stop, plot_titles, save_directory, behaviour_dict_list, selected_behaviour_traces, timestep=36):

    # Check Save Directory Exists
    print("Save Directory", save_directory)
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Construct Images
    number_of_sessions = len(base_directory_list)
    reconstructed_activity_tensors = []
    for session_index in range(number_of_sessions):

        # Load mask
        base_directory = base_directory_list[session_index]
        indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

        reconstructed_activity_tensor = reconstruct_images_from_activity(mean_activity_tensors[session_index], indicies, image_height, image_width)
        reconstructed_activity_tensors.append(reconstructed_activity_tensor)
    reconstructed_activity_tensors = np.array(reconstructed_activity_tensors)

    # Create Figure
    number_of_timepoints = trial_stop - trial_start
    print("Numbe of timepoints", number_of_timepoints)
    number_of_columns = number_of_sessions
    number_of_rows = 2
    figure_1 = plt.figure(constrained_layout=True, figsize=(80, 60))
    grid_spec_1 = gridspec.GridSpec(ncols=number_of_columns + 1, nrows=number_of_rows, figure=figure_1)

    # Plot For Each Timepoint
    timepoint_count = 0
    timepoint_list = list(range(trial_start, trial_stop))
    timepoint_list = np.multiply(timepoint_list, timestep)

    # Create Colourmaps
    print("Mean Activity Tensor Shape", np.shape(reconstructed_activity_tensors))
    delta_f_vmax = np.percentile(reconstructed_activity_tensors, q=99)
    delta_f_vmin = 0
    diff_v_max = delta_f_vmax * 0.5
    diff_v_min = -1 * delta_f_vmax * 0.5
    delta_f_colourmap = cm.ScalarMappable(cmap=cm.get_cmap('inferno'), norm=Normalize(vmin=delta_f_vmin, vmax=delta_f_vmax))
    difference_colourmap = cm.ScalarMappable(cmap=cm.get_cmap('bwr'), norm=Normalize(vmin=diff_v_min, vmax=diff_v_max))

    # Get X Values
    x_values = list(range(trial_start * timestep, trial_stop * timestep, timestep))

    # Get Number of Behaviour Traces
    selected_trace_list = list(behaviour_dict_list[0].keys())
    number_of_traces = len(selected_trace_list)

    # Select Behaviour Trace Colours
    behaviour_trace_colour_list = []
    behaviour_colour_map = cm.get_cmap('jet')
    for trace_index in range(number_of_traces):
        trace_index = float(trace_index) / number_of_traces
        trace_colour = behaviour_colour_map(trace_index)
        behaviour_trace_colour_list.append(trace_colour)

    # Set Behaviour Trace Offset
    trace_offset = 1.5

    # Get Jointly Scaled Beahviour Traces
    scaled_behaviour_trace_list = jointly_scale_behaviour_traces(behaviour_dict_list, selected_behaviour_traces, number_of_traces, number_of_sessions)

    # Plot Each Timepoint
    for timepoint in range(number_of_timepoints):

        print(timepoint)
        time_text = str(timepoint_list[timepoint]) + "ms"
        x_pos = 30
        y_pos = 75

        # Add Axes
        for session_index in range(number_of_sessions):

            session_activity_axis = figure_1.add_subplot(grid_spec_1[0, session_index])
            session_behaviour_axis = figure_1.add_subplot(grid_spec_1[1, session_index])

            # Select Images
            session_activity_image = reconstructed_activity_tensors[session_index, timepoint]

            # Gaussian Smoothing
            session_activity_image = ndimage.gaussian_filter(session_activity_image, sigma=1)

            # Add Colourmap
            session_activity_image = delta_f_colourmap.to_rgba(session_activity_image)

            # Plot Image
            session_activity_axis.imshow(session_activity_image)

            # Remove Axis
            session_activity_axis.axis('off')

            # Set Title
            session_activity_axis.set_title(plot_titles[session_index] + "_Raw_Activity")

            # Add Time Text
            session_activity_axis.text(x_pos, y_pos, time_text, color='White')

            for trace_index in range(number_of_traces):

                # Load Scaled Trace
                scaled_trace = scaled_behaviour_trace_list[trace_index, session_index]

                # Add Offset
                offset = trace_index * trace_offset
                scaled_trace = np.add(scaled_trace, offset)

                # Get Trace Colour
                trace_colour = behaviour_trace_colour_list[trace_index]

                # Plot Traces
                session_behaviour_axis.plot(x_values, scaled_trace, c=trace_colour)

            # Remove Behaviour Axis Y Values
            session_behaviour_axis.get_yaxis().set_visible(False)

            # Remove Behaviour Axis Boxes
            remove_axis_border(session_behaviour_axis)

            # PLace Vline At Trial Offsets
            session_behaviour_axis.axvline(x=0, ymin=0, ymax=1, c='k', linestyle=(0, (5,5)))

            current_time = 0 + (trial_start * 36) + (timepoint * 36)
            session_behaviour_axis.axvline(x=current_time, ymin=0, ymax=1, c='b')

        # Plot Beahviour Lengend
        legend_axis = figure_1.add_subplot(grid_spec_1[1, number_of_sessions])
        patch_list = []

        for trace_index in range(number_of_traces):
            trace_name = selected_trace_list[trace_index]
            trace_colour = behaviour_trace_colour_list[trace_index]

            patch = mpatches.Patch(color=trace_colour, label=trace_name)
            patch_list.append(patch)

        patch_list.reverse()
        legend_axis.legend(handles=patch_list, fontsize='xx-large', loc='center')
        legend_axis.axis('off')

        plt.draw()
        plt.savefig(os.path.join(save_directory, str(timepoint_count).zfill(3) + ".png"))
        plt.pause(0.1)
        plt.clf()

        # plt.show()
        timepoint_count += 1

