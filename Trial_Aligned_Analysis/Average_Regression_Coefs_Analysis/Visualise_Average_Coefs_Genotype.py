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
from skimage.transform import resize, rotate
from skimage.morphology import binary_dilation
from skimage.feature import canny
from skimage.measure import find_contours
from scipy import ndimage

from Widefield_Utils import widefield_utils


def get_smooth_outline(resize_shape=600):

    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

    # Reshape Into Template
    template = np.zeros(image_height * image_width)
    template[indicies] = 1
    template = np.reshape(template, (image_height, image_width))

    template = resize(template, (resize_shape, resize_shape), anti_aliasing=True)

    edges = canny(template, sigma=8)
    for x in range(5):
        edges = binary_dilation(edges)


    edge_indicies = np.nonzero(edges)

    return edge_indicies


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


def create_stimulus_grating(shift=0, angle=45):

    stimulus_grating_image = np.zeros((100,150))
    n_stripes = 5
    stimulus_width = 150
    x_values = np.linspace(start=0, stop=2*np.pi, num=stimulus_width)
    x_values = np.multiply(x_values, n_stripes)
    values = np.sin(x_values)
    stimulus_grating_image[:] = values
    stimulus_grating_image = np.roll(stimulus_grating_image, shift=-shift, axis=1)


    # Rotate
    stimulus_grating_image = rotate(stimulus_grating_image, angle=angle, resize=False, mode='wrap', preserve_range=True)

    return stimulus_grating_image


def add_stimulus_grating(axis, shift, angle):

    # Create Grating Image
    stimulus_grating = create_stimulus_grating(shift=shift, angle=angle)

    # Create Axis
    width = 100
    height = 100
    x0 = 0
    y0 = 0
    stimulus_axis = axis.inset_axes([x0, y0, width, height],transform=axis.transData)    # create
    stimulus_axis.imshow(stimulus_grating, cmap='Greys')
    stimulus_axis.axis('off')


def view_average_difference(tensor_directory, selected_condition_index, comparison_name, start_window, stop_window, title_list=["", "", ""], vmin=-0.05, vmax=0.05, diff_magnitude=0.5):

    # Load Data
    # (Genotypes x Conditions x Timepoint x Pixels)
    activity_dataset = np.load(os.path.join(tensor_directory, "Average_Coefs", "Genotype_Average_Matrix.npy"), allow_pickle=True)
    print("Activity Dataset Shape", np.shape(activity_dataset))
    number_of_genotypes, number_of_conditions, number_of_timepoints, number_of_pixels = np.shape(activity_dataset)

    condition_1_data = activity_dataset[1, selected_condition_index]
    condition_2_data = activity_dataset[0, selected_condition_index]

    # Create Colourmaps
    diff_v_max = diff_magnitude
    diff_v_min = -1 * diff_magnitude

    delta_f_cmap = cm.get_cmap("inferno")
    difference_cmap = widefield_utils.get_musall_cmap()

    delta_f_colourmap = cm.ScalarMappable(cmap=delta_f_cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    difference_colourmap = cm.ScalarMappable(cmap=difference_cmap, norm=Normalize(vmin=diff_v_min, vmax=diff_v_max))

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

    # Get Background Pixels
    background_indicies = get_background_pixels(indicies, image_height, image_width)

    # Get Atlas Outlines
    atlas_indicies = get_atlas_indicies()

    # Get Mask Edge Pixels
    edge_indicies = get_smooth_outline()
    print("Edge indicies", edge_indicies)

    # Create Output Directory
    output_directory = os.path.join(tensor_directory, comparison_name)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    x_values = list(range(int(start_window), int(stop_window)))
    x_values = np.multiply(x_values, 36)

    # Iterate Through Each Timepoint
    stimuli_roll_count = 0
    number_of_timepoints = np.shape(condition_1_data)[0]
    for timepoint_index in tqdm(range(number_of_timepoints), position=0, desc="Timepoint"):

        figure_1 = plt.figure(figsize=(20,7))

        condition_1_axis = figure_1.add_subplot(1, 3, 2)
        condition_2_axis = figure_1.add_subplot(1, 3, 1)
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

        condition_1_image = resize(condition_1_image, (600, 600), anti_aliasing=True, preserve_range=True)
        condition_2_image = resize(condition_2_image, (600, 600), anti_aliasing=True, preserve_range=True)
        difference_image = resize(difference_image, (600, 600), anti_aliasing=True, preserve_range=True)

        # Smooth Edges
        condition_1_image[edge_indicies] = [1, 1, 1, 1]
        condition_2_image[edge_indicies] = [1, 1, 1, 1]
        difference_image[edge_indicies] = [1, 1, 1, 1]

        # Remove Axis
        condition_1_axis.axis('off')
        condition_2_axis.axis('off')
        diff_axis.axis('off')

        condition_1_axis.set_title(title_list[0])
        condition_2_axis.set_title(title_list[1])
        diff_axis.set_title(title_list[2])

        # Plot These
        condition_1_axis.imshow(condition_1_image)
        condition_2_axis.imshow(condition_2_image)
        diff_axis.imshow(difference_image)

        # Add Colourbars
        figure_1.colorbar(cm.ScalarMappable(norm=Normalize(vmin=vmin * 100, vmax=vmax * 100), cmap=delta_f_cmap), ax=condition_1_axis, fraction=0.046, pad=0.04)
        figure_1.colorbar(cm.ScalarMappable(norm=Normalize(vmin=vmin * 100, vmax=vmax * 100), cmap=delta_f_cmap), ax=condition_2_axis, fraction=0.046, pad=0.04)
        figure_1.colorbar(cm.ScalarMappable(norm=Normalize(vmin=diff_v_min * 100, vmax=diff_v_max * 100), cmap=difference_cmap), ax=diff_axis, fraction=0.046, pad=0.04)

        # Add Stimulus Image
        if x_values[timepoint_index] >= 0:
            add_stimulus_grating(condition_1_axis, stimuli_roll_count, 45)
            add_stimulus_grating(condition_2_axis, stimuli_roll_count, 45)
            stimuli_roll_count += 2


        figure_1.suptitle(str(x_values[timepoint_index]) + "ms")
        plt.savefig(os.path.join(output_directory, str(timepoint_index).zfill(3) + ".png"))
        plt.close()







