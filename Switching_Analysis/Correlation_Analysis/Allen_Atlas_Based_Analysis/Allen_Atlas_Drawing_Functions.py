import numpy as np
from scipy import ndimage
from skimage.transform import resize
import sys
import os
import matplotlib.pyplot as plt
import cv2
from skimage import feature
sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions


def add_region_outline(base_directory):

    # Load Mask
    mask = np.load(base_directory + "mask.npy")

    # Get Outline
    edges = feature.canny(mask, sigma=3)

    # Binarise
    binary_edges = np.where(edges > 0, 1, 0)

    return binary_edges


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

    mask_indicies = np.nonzero(np.ndarray.flatten(masked_atlas))


    return masked_atlas, mask_indicies

"""
base_directory = "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/"
outline_array, outline_indicies = add_region_outline(base_directory)

plt.imshow(outline_array)
plt.show()
"""