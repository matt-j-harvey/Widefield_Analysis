import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import networkx as nx
import cv2
from matplotlib.pyplot import cm
from matplotlib.colors import to_rgb
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist
import os
import tables
import sys
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from skimage.transform import resize


sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions


def get_selected_pixels(selected_regions, pixel_assignments):

    selected_pixels = []
    for region in selected_regions:
        region_mask = np.where(pixel_assignments == region, 1, 0)
        region_indicies = np.nonzero(region_mask)[0]
        for index in region_indicies:
            selected_pixels.append(index)
    selected_pixels.sort()

    return selected_pixels


def get_session_allen_atlas(base_directory):

    # Load Atlas Regions
    atlas_region_mapping = np.load(r"/home/matthew/Documents/Allen_Atlas_Templates/Atlas_Template_V2.npy")

    # Load Atlas Transformation Details
    atlas_alignment_dictionary = np.load(os.path.join(base_directory, "Atlas_Alignment_Dictionary.npy"), allow_pickle=True)
    atlas_alignment_dictionary  = atlas_alignment_dictionary[()]
    atlas_rotation              = atlas_alignment_dictionary['rotation']
    atlas_x_scale_factor        = atlas_alignment_dictionary['x_scale_factor']
    atlas_y_scale_factor        = atlas_alignment_dictionary['y_scale_factor']
    atlas_x_shift               = atlas_alignment_dictionary['x_shift']
    atlas_y_shift               = atlas_alignment_dictionary['y_shift']

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

    return masked_atlas

base_directory = "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/"



# Load Region Assignments
pixel_assignments = np.load(os.path.join(base_directory, "Pixel_Assignmnets.npy"))
pixel_assignments = np.ndarray.astype(pixel_assignments, np.int)

# Get Number Of Regions
region_list = list(pixel_assignments)
region_list = set(region_list)
region_list = list(region_list)
number_of_regions = len(region_list)
print("Number of regions", number_of_regions)
print(region_list)

indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

region_centroids = []
excluded_list = [0, 1, 3, 4, 5, 10, 17, 18]
for region in region_list:
    if region not in excluded_list:

        selected_pixels = get_selected_pixels([region], pixel_assignments)
        selected_pixels_original_space = indicies[selected_pixels]
        blank_mask = np.zeros((image_height * image_width))
        blank_mask[selected_pixels_original_space] = 1
        blank_mask = np.ndarray.reshape(blank_mask, (image_height, image_width))

        pixel_coordinates = np.nonzero(blank_mask)
        mean_y_coord = np.mean(pixel_coordinates[0])
        mean_x_coord = np.mean(pixel_coordinates[1])
        region_centroids.append([mean_y_coord, mean_x_coord])

        #plt.imshow(blank_mask)
        #plt.scatter([mean_x_coord], [mean_y_coord])
        #plt.show()

masked_atlas = get_session_allen_atlas(base_directory)

region_centroids = np.array(region_centroids)
plt.imshow(masked_atlas)
plt.scatter(region_centroids[:, 1], region_centroids[:, 0])
plt.show()

np.save("/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/Allen_Region_Centroids.npy", region_centroids)


