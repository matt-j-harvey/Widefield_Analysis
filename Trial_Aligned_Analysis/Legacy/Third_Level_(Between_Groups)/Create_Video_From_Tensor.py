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
from tqdm import tqdm
from skimage.feature import canny

import Group_Analysis_Utils


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


def get_background_pixels():

    indicies, image_height, image_width = Group_Analysis_Utils.load_tight_mask()

    template = np.ones(image_height * image_width)
    template[indicies] = 0
    template = np.reshape(template, (image_height, image_width))
    background_indicies = np.nonzero(template)
    return background_indicies


def get_mask_edge_pixels():

    indicies, image_height, image_width = Group_Analysis_Utils.load_tight_mask()

    template = np.zeros(image_height * image_width)
    template[indicies] = 1
    template = np.reshape(template, (image_height, image_width))
    edges = canny(template)

    edge_indicies = np.nonzero(edges)
    return edge_indicies


def get_atlas_outline_pixels():

    # Load Atlas
    atlas_outline = np.load("/home/matthew/Documents/Github_Code_Clean/Transition_Analysis/Brain_Registration/Atlas_Outlines.npy")

    # Load Atlas Transformation Dict
    transformation_dict = np.load(r"/home/matthew/Documents/Github_Code_Clean/Transition_Analysis/Brain_Registration/Atlas_Alignment_Dictionary.npy", allow_pickle=True)[()]

    # Transform Atlas
    atlas_outline = Group_Analysis_Utils.transform_mask_or_atlas(atlas_outline, transformation_dict)



    atlas_pixels = np.nonzero(atlas_outline)
    return atlas_pixels




def create_activity_video(mean_activity_tensors, trial_start, plot_titles, save_directory, indicies, image_height, image_width, timestep=36):

    # Check Save Directory Exists
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Get Region Boundaries
    background_indicies = get_background_pixels()
    edge_indicies = get_mask_edge_pixels()
    atlas_indicies = get_atlas_outline_pixels()

    # Load Number Of Condition
    number_of_conditions = len(mean_activity_tensors)

    # Get Size Of Smallest Tensor
    tensor_sizes = []
    for tensor in mean_activity_tensors:
        tensor_length = np.shape(tensor)[0]
        tensor_sizes.append(tensor_length)
    smallest_tensor_size = np.min(tensor_sizes)
    print("Smallest Condition", smallest_tensor_size)

    # Create Figure
    number_of_timepoints = smallest_tensor_size
    number_of_columns = number_of_conditions + 1
    number_of_rows = 1
    figure_1 = plt.figure(constrained_layout=True)
    grid_spec_1 = gridspec.GridSpec(ncols=number_of_columns, nrows=number_of_rows, figure=figure_1)

    # Plot For Each Timepoint
    timepoint_count = 0
    trial_stop = trial_start + number_of_timepoints
    timepoint_list = list(range(trial_start, trial_stop))
    timepoint_list = np.multiply(timepoint_list, timestep)
    print("Trial Start: ", trial_start)
    print("Trial Stop: ", trial_stop)
    print("Timepoint List", timepoint_list)

    # Create Colourmaps
    delta_f_vmax = 0.03
    delta_f_vmin = -0.03
    diff_scale_factor = 0.5
    diff_v_max = delta_f_vmax * diff_scale_factor
    diff_v_min = -1 * delta_f_vmax * diff_scale_factor
    blue_black_red_cmap = Group_Analysis_Utils.get_musall_cmap()
    delta_f_colourmap = cm.ScalarMappable(cmap=blue_black_red_cmap, norm=Normalize(vmin=delta_f_vmin, vmax=delta_f_vmax))
    difference_colourmap = cm.ScalarMappable(cmap=blue_black_red_cmap, norm=Normalize(vmin=diff_v_min, vmax=diff_v_max))

    # Get X Values
    x_values = list(range(trial_start * timestep, trial_stop * timestep, timestep))

    # Plot Each Timepoint
    for timepoint in tqdm(range(number_of_timepoints)):

        time_text = str(timepoint_list[timepoint]) + "ms"
        x_pos = 30
        y_pos = 75

        # Add Axes
        for condition_index in range(number_of_conditions):

            condition_activity_axis = figure_1.add_subplot(grid_spec_1[0, condition_index])

            # Select Images
            condition_activity_image = mean_activity_tensors[condition_index][timepoint]
            condition_activity_image = Group_Analysis_Utils.create_image_from_data(condition_activity_image, indicies, image_height, image_width)

            # Gaussian Smoothing
            #condition_activity_image = ndimage.gaussian_filter(condition_activity_image, sigma=1)

            # Add Colourmap
            condition_activity_image = delta_f_colourmap.to_rgba(condition_activity_image)

            # Add Outlines
            condition_activity_image[atlas_indicies] = [1, 1, 1, 1]

            # Set Background To White
            condition_activity_image[background_indicies] = [1, 1, 1, 1]

            # Plot Image
            condition_activity_axis.imshow(condition_activity_image)

            # Remove Axis
            condition_activity_axis.axis('off')

            # Set Title
            condition_activity_axis.set_title(plot_titles[condition_index].replace("_", " "))

            # Add Time Text
            condition_activity_axis.text(x_pos, y_pos, time_text, color='Black')

        colourbar = figure_1.colorbar(cm.ScalarMappable(norm=Normalize(vmin=delta_f_vmin * 100, vmax=delta_f_vmax * 100), cmap=blue_black_red_cmap), ax=condition_activity_axis, fraction=0.046, pad=0.04)
        #colourbar.set_title("dF/F (%)")

        # Get Difference Image
        difference_axis = figure_1.add_subplot(grid_spec_1[0, -1])

        # Select Images
        difference_activity_image = np.subtract(mean_activity_tensors[0][timepoint], mean_activity_tensors[1][timepoint])
        difference_activity_image = Group_Analysis_Utils.create_image_from_data(difference_activity_image, indicies, image_height, image_width)
        difference_activity_image = ndimage.gaussian_filter(difference_activity_image, sigma=1)

        # Add Colourmap
        difference_activity_image = difference_colourmap.to_rgba(difference_activity_image)

        # Add Outlines
        difference_activity_image[atlas_indicies] = [1, 1, 1, 1]

        # Set Background To White
        difference_activity_image[background_indicies] = [1, 1, 1, 1]

        # Plot Image
        difference_axis.imshow(difference_activity_image)

        # Remove Axis
        difference_axis.axis('off')

        # Set Title
        difference_axis.set_title("Difference")

        # Add Time Text
        difference_axis.text(x_pos, y_pos, time_text, color='Black')


        plt.draw()
        figure_1.set_size_inches(15, 9)
        plt.savefig(os.path.join(save_directory, str(timepoint_count).zfill(3) + ".png"))
        plt.clf()


        timepoint_count += 1






def create_activity_video_with_significance(mean_activity_tensors, p_values, trial_start, plot_titles, save_directory, indicies, image_height, image_width, timestep=36):

    # Check Save Directory Exists
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Get Region Boundaries
    background_indicies = get_background_pixels()
    edge_indicies = get_mask_edge_pixels()
    atlas_indicies = get_atlas_outline_pixels()

    # Load Number Of Condition
    number_of_conditions = len(mean_activity_tensors)

    # Get Size Of Smallest Tensor
    tensor_sizes = []
    for tensor in mean_activity_tensors:
        tensor_length = np.shape(tensor)[0]
        tensor_sizes.append(tensor_length)
    smallest_tensor_size = np.min(tensor_sizes)
    print("Smallest Condition", smallest_tensor_size)

    # Create Figure
    number_of_timepoints = smallest_tensor_size
    number_of_columns = number_of_conditions + 1
    number_of_rows = 1
    figure_1 = plt.figure(constrained_layout=True)
    grid_spec_1 = gridspec.GridSpec(ncols=number_of_columns, nrows=number_of_rows, figure=figure_1)

    # Plot For Each Timepoint
    timepoint_count = 0
    trial_stop = trial_start + number_of_timepoints
    timepoint_list = list(range(trial_start, trial_stop))
    timepoint_list = np.multiply(timepoint_list, timestep)
    print("Trial Start: ", trial_start)
    print("Trial Stop: ", trial_stop)
    print("Timepoint List", timepoint_list)

    # Create Colourmaps
    delta_f_vmax = 0.03
    delta_f_vmin = -0.03
    diff_scale_factor = 0.5
    diff_v_max = delta_f_vmax * diff_scale_factor
    diff_v_min = -1 * delta_f_vmax * diff_scale_factor
    blue_black_red_cmap = Group_Analysis_Utils.get_musall_cmap()
    delta_f_colourmap = cm.ScalarMappable(cmap=blue_black_red_cmap, norm=Normalize(vmin=delta_f_vmin, vmax=delta_f_vmax))
    difference_colourmap = cm.ScalarMappable(cmap=blue_black_red_cmap, norm=Normalize(vmin=diff_v_min, vmax=diff_v_max))

    # Get X Values
    x_values = list(range(trial_start * timestep, trial_stop * timestep, timestep))

    # Plot Each Timepoint
    for timepoint in tqdm(range(number_of_timepoints)):

        time_text = str(timepoint_list[timepoint]) + "ms"
        x_pos = 30
        y_pos = 75

        # Add Axes
        for condition_index in range(number_of_conditions):

            condition_activity_axis = figure_1.add_subplot(grid_spec_1[0, condition_index])

            # Select Images
            condition_activity_image = mean_activity_tensors[condition_index][timepoint]
            condition_activity_image = Group_Analysis_Utils.create_image_from_data(condition_activity_image, indicies, image_height, image_width)
            condition_activity_image = ndimage.gaussian_filter(condition_activity_image, sigma=1)


            # Gaussian Smoothing
            #condition_activity_image = ndimage.gaussian_filter(condition_activity_image, sigma=1)

            # Add Colourmap
            condition_activity_image = delta_f_colourmap.to_rgba(condition_activity_image)

            # Add Outlines
            condition_activity_image[atlas_indicies] = [1, 1, 1, 1]

            # Set Background To White
            condition_activity_image[background_indicies] = [1, 1, 1, 1]

            # Plot Image
            condition_activity_axis.imshow(condition_activity_image)

            # Remove Axis
            condition_activity_axis.axis('off')

            # Set Title
            condition_activity_axis.set_title(plot_titles[condition_index].replace("_", " "))

            # Add Time Text
            condition_activity_axis.text(x_pos, y_pos, time_text, color='Black')

        colourbar = figure_1.colorbar(cm.ScalarMappable(norm=Normalize(vmin=delta_f_vmin * 100, vmax=delta_f_vmax * 100), cmap=blue_black_red_cmap), ax=condition_activity_axis, fraction=0.046, pad=0.04)
        #colourbar.set_title("dF/F (%)")

        # Get Difference Image
        difference_axis = figure_1.add_subplot(grid_spec_1[0, -1])

        # Select Images
        difference_activity_image = np.subtract(mean_activity_tensors[0][timepoint], mean_activity_tensors[1][timepoint])

        # THreshold By P Value
        difference_activity_image = np.where(p_values[timepoint] < 0.5, difference_activity_image, 0)
        difference_activity_image = Group_Analysis_Utils.create_image_from_data(difference_activity_image, indicies, image_height, image_width)
        difference_activity_image = ndimage.gaussian_filter(difference_activity_image, sigma=1)

        # Add Colourmap
        difference_activity_image = difference_colourmap.to_rgba(difference_activity_image)

        # Add Outlines
        difference_activity_image[atlas_indicies] = [1, 1, 1, 1]

        # Set Background To White
        difference_activity_image[background_indicies] = [1, 1, 1, 1]

        # Plot Image
        difference_axis.imshow(difference_activity_image)

        # Remove Axis
        difference_axis.axis('off')

        # Set Title
        difference_axis.set_title("Difference")

        # Add Diference Colorbar
        diff_colourbar = figure_1.colorbar(cm.ScalarMappable(norm=Normalize(vmin=diff_v_min * 100, vmax=diff_v_max * 100), cmap=blue_black_red_cmap), ax=difference_axis, fraction=0.046, pad=0.04)

        # Add Time Text
        difference_axis.text(x_pos, y_pos, time_text, color='Black')


        plt.draw()
        figure_1.set_size_inches(15, 9)
        plt.savefig(os.path.join(save_directory, str(timepoint_count).zfill(3) + ".png"))
        plt.clf()


        timepoint_count += 1




