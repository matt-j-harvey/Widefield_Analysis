import os
import h5py
from tqdm import tqdm
import numpy as np
import statsmodels.api as sm
import pandas as pd
import tables
from datetime import datetime
from collections import Counter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from skimage.feature import canny
from scipy import ndimage
from skimage.transform import rescale

from Widefield_Utils import widefield_utils


# View For Each Session
# View For Each Mouse
# View Overall Average



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



def get_background_pixels(indicies, image_height, image_width ):
    template = np.ones(image_height * image_width)
    template[indicies] = 0
    template = np.reshape(template, (image_height, image_width))
    background_indicies = np.nonzero(template)
    return background_indicies


def get_mask_edge_pixels(indicies, image_height, image_width):

    template = np.zeros(image_height * image_width)
    template[indicies] = 3
    template = np.reshape(template, (image_height, image_width))

    template = ndimage.gaussian_filter(template, sigma=1)

    edges = canny(template, sigma=1)
    edges = np.where(edges > 0.5, 1, 0)
    edge_indicies = np.nonzero(edges)
    return edge_indicies


def view_average_difference(tensor_directory, condition_1_index, condition_2_index, comparison_name, delta_f_vmin=-0.05, delta_f_vmax=0.05, diff_scale_factor = 0.5):

    # Load Data
    condition_averages = np.load(os.path.join(tensor_directory, "Average_Coefs", "Condition_Average_Matrix.npy"))
    condition_1_data = condition_averages[condition_1_index]
    condition_2_data = condition_averages[condition_2_index]

    # Create Colourmaps
    diff_v_max = delta_f_vmax * diff_scale_factor
    diff_v_min = -1 * diff_v_max
    blue_black_red_cmap = widefield_utils.get_musall_cmap()
    delta_f_colourmap = cm.ScalarMappable(cmap=blue_black_red_cmap, norm=Normalize(vmin=delta_f_vmin, vmax=delta_f_vmax))
    difference_colourmap = cm.ScalarMappable(cmap=blue_black_red_cmap, norm=Normalize(vmin=diff_v_min, vmax=diff_v_max))

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

    # Get Background Pixels
    background_indicies = get_background_pixels(indicies, image_height, image_width)

    # Get Mask Edge Pixels
    #mask_edge_indicies = get_mask_edge_pixels(indicies, image_height, image_width)

    # Create Output Directory
    output_directory = os.path.join(tensor_directory, comparison_name)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # Iterate Through Each Timepoint
    number_of_timepoints = np.shape(condition_1_data)[0]
    for timepoint_index in tqdm(range(number_of_timepoints), position=0, desc="Timepoint"):

        figure_1 = plt.figure(figsize=(20,20))

        condition_1_axis = figure_1.add_subplot(1,3,1)
        condition_2_axis = figure_1.add_subplot(1, 3, 2)
        diff_axis = figure_1.add_subplot(1, 3, 3)

        # Recreate Images
        condition_1_image = widefield_utils.create_image_from_data(condition_1_data[timepoint_index],  indicies, image_height, image_width)
        condition_2_image = widefield_utils.create_image_from_data(condition_2_data[timepoint_index], indicies, image_height, image_width)
        difference_image = np.subtract(condition_1_image, condition_2_image)

        # Add Colour
        condition_1_image = delta_f_colourmap.to_rgba(condition_1_image)
        condition_2_image = delta_f_colourmap.to_rgba(condition_2_image)
        difference_image = difference_colourmap.to_rgba(difference_image)

        # Remove Background
        condition_1_image[background_indicies] = [1, 1, 1, 1]
        condition_2_image[background_indicies] = [1, 1, 1, 1]
        difference_image[background_indicies] = [1, 1, 1, 1]

        # Smooth Edges
        #condition_1_image[mask_edge_indicies] = [0, 0, 0, 1]

        # Remove Axis
        condition_1_axis.axis('off')
        condition_2_axis.axis('off')
        diff_axis.axis('off')

        # Plot These
        condition_1_axis.imshow(condition_1_image)
        condition_2_axis.imshow(condition_2_image)
        diff_axis.imshow(difference_image)

        # Add Colourbars
        condition_1_colourbar = figure_1.colorbar(cm.ScalarMappable(norm=Normalize(vmin=delta_f_vmin * 100, vmax=delta_f_vmax * 100), cmap=blue_black_red_cmap), ax=condition_1_axis, fraction=0.046, pad=0.04)
        condition_2_colourbar = figure_1.colorbar(cm.ScalarMappable(norm=Normalize(vmin=delta_f_vmin * 100, vmax=delta_f_vmax * 100), cmap=blue_black_red_cmap), ax=condition_2_axis, fraction=0.046, pad=0.04)
        diff_colourbar = figure_1.colorbar(cm.ScalarMappable(norm=Normalize(vmin=diff_v_min * 100, vmax=diff_v_max * 100), cmap=blue_black_red_cmap), ax=diff_axis, fraction=0.046, pad=0.04)

        plt.title(str(timepoint_index))
        plt.savefig(os.path.join(output_directory, str(timepoint_index).zfill(3) + ".png"))
        plt.close()




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

