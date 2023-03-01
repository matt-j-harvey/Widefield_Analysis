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
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from skimage.transform import resize

from Widefield_Utils import widefield_utils




def get_background_pixels(indicies, image_height, image_width ):
    template = np.ones(image_height * image_width)
    template[indicies] = 0
    template = np.reshape(template, (image_height, image_width))
    background_indicies = np.nonzero(template)
    return background_indicies


def get_atlas_indicies():

    # Load Atlas Regions
    atlas_outlines = np.load(r"/home/matthew/Documents/Github_Code_Clean/Widefield_Analysis/Files/Atlas_Outlines.npy")

    # Load Atlas Transformation Dict
    atlas_alignment_dict = np.load("/home/matthew/Documents/Github_Code_Clean/Widefield_Analysis/Files/Atlas_Alignment_Dictionary.npy", allow_pickle=True)[()]

    # Transform Atlas
    atlas_outlines = widefield_utils.transform_mask_or_atlas_300(atlas_outlines, atlas_alignment_dict)

    # Get Pixel Indicies
    outline_pixels = np.nonzero(atlas_outlines)

    return outline_pixels


def view_average_difference(tensor_directory, condition_1_index, condition_2_index, comparison_name, vmin=-0.05, vmax=0.05, diff_scale_factor=0.5):

    # Load Data
    activity_dataset = np.load(os.path.join(tensor_directory, "Average_Activity", "Condition_Average_Matrix.npy"), allow_pickle=True)
    print("Activity Dataset Shape", np.shape(activity_dataset))# (Conditions x Timepoint x Pixels)
    number_of_conditions, number_of_timepoints, number_of_pixels = np.shape(activity_dataset)

    condition_1_data = activity_dataset[condition_1_index]
    condition_2_data = activity_dataset[condition_2_index]

    # Create Colourmaps
    diff_v_max = vmax * diff_scale_factor
    diff_v_min = -1 * diff_v_max
    blue_black_red_cmap = widefield_utils.get_musall_cmap()
    delta_f_colourmap = cm.ScalarMappable(cmap=blue_black_red_cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    difference_colourmap = cm.ScalarMappable(cmap=blue_black_red_cmap, norm=Normalize(vmin=diff_v_min, vmax=diff_v_max))

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

    # Get Background Pixels
    background_indicies = get_background_pixels(indicies, image_height, image_width)

    # Get Atlas Outlines
    atlas_indicies = get_atlas_indicies()

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

        condition_1_image = resize(condition_1_image, (300, 300), anti_aliasing=True, preserve_range=True)
        condition_2_image = resize(condition_2_image, (300, 300), anti_aliasing=True, preserve_range=True)
        difference_image = resize(difference_image, (300, 300), anti_aliasing=True, preserve_range=True)

        condition_1_image[atlas_indicies] = [1, 1, 1, 1]
        condition_2_image[atlas_indicies] = [1, 1, 1, 1]
        difference_image[atlas_indicies] = [1, 1, 1, 1]

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
        condition_1_colourbar = figure_1.colorbar(cm.ScalarMappable(norm=Normalize(vmin=vmin * 100, vmax=vmax * 100), cmap=blue_black_red_cmap), ax=condition_1_axis, fraction=0.046, pad=0.04)
        condition_2_colourbar = figure_1.colorbar(cm.ScalarMappable(norm=Normalize(vmin=vmin * 100, vmax=vmax * 100), cmap=blue_black_red_cmap), ax=condition_2_axis, fraction=0.046, pad=0.04)
        diff_colourbar = figure_1.colorbar(cm.ScalarMappable(norm=Normalize(vmin=diff_v_min * 100, vmax=diff_v_max * 100), cmap=blue_black_red_cmap), ax=diff_axis, fraction=0.046, pad=0.04)

        plt.title(str(timepoint_index))
        plt.savefig(os.path.join(output_directory, str(timepoint_index).zfill(3) + ".png"))
        plt.close()


def view_average_difference_per_mouse(tensor_directory, condition_1_index, condition_2_index, comparison_name, vmin=-0.05, vmax=0.05, diff_scale_factor=0.5):

    # Load Data
    activity_dataset = np.load(os.path.join(tensor_directory, "Average_Activity", "Mouse_Condition_Average_Matrix.npy"), allow_pickle=True)  # (Mice x Conditions x Timepoint x Pixels)
    n_mice, number_of_conditions, number_of_timepoints, number_of_pixels = np.shape(activity_dataset)

    # Get Mouse Data
    condition_1_list_per_mouse = activity_dataset[:, condition_1_index]
    condition_2_list_per_mouse = activity_dataset[:, condition_2_index]

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)


    # Create Colourmaps
    diff_v_max = vmax * diff_scale_factor
    diff_v_min = -1 * diff_v_max
    blue_black_red_cmap = widefield_utils.get_musall_cmap()
    delta_f_colourmap = cm.ScalarMappable(cmap=blue_black_red_cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    difference_colourmap = cm.ScalarMappable(cmap=blue_black_red_cmap, norm=Normalize(vmin=diff_v_min, vmax=diff_v_max))

    # Get Background Pixels
    background_indicies = get_background_pixels(indicies, image_height, image_width)

    # Create Output Directory
    output_directory = os.path.join(tensor_directory, comparison_name + "_Individual_Mice")
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # Plot This
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

            # Add Colour
            condition_1_image = delta_f_colourmap.to_rgba(condition_1_image)
            condition_2_image = delta_f_colourmap.to_rgba(condition_2_image)
            difference_image = difference_colourmap.to_rgba(difference_image)

            # Remove Background
            condition_1_image[background_indicies] = [1, 1, 1, 1]
            condition_2_image[background_indicies] = [1, 1, 1, 1]
            difference_image[background_indicies] = [1, 1, 1, 1]


            # Plot These
            condition_1_axis.imshow(condition_1_image)
            condition_2_axis.imshow(condition_2_image)
            diff_axis.imshow(difference_image)

            condition_1_axis.axis('off')
            condition_2_axis.axis('off')
            diff_axis.axis('off')

        figure_1.suptitle(str(timepoint_index))
        plt.savefig(os.path.join(output_directory, str(timepoint_index).zfill(3) + ".png"))

