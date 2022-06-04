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
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
from datetime import datetime

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions
import Draw_Brain_Network
import Allen_Atlas_Drawing_Functions


def get_selected_pixels(selected_regions, pixel_assignments):

    # Get Pixels Within Selected Regions
    selected_pixels = []
    for region in selected_regions:
        region_mask = np.where(pixel_assignments == region, 1, 0)
        region_indicies = np.nonzero(region_mask)[0]
        for index in region_indicies:
                selected_pixels.append(index)
        selected_pixels.sort()

    return selected_pixels


def view_seed_map(base_directory, map, selected_regions, name='Untitled'):

    # Load Mask
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    # Load Region Assigments
    pixel_assignments = np.load(os.path.join(base_directory, "Pixel_Assignmnets.npy"))

    # Get Selected Pixels
    selected_pixels = get_selected_pixels(selected_regions, pixel_assignments)

    # Divide by 2
    map = np.add(map, 1)
    map = np.divide(map, 2)

    # Create Colourmap
    colourmap = cm.get_cmap('seismic')
    map_rgba = colourmap(map)

    # Highlight Regions
    highlight_colour = (1, 1, 0, 1)
    map_rgba[selected_pixels] = highlight_colour

    map_image = np.zeros((image_width * image_height, 4))
    map_image[indicies] = map_rgba
    map_image = np.ndarray.reshape(map_image, (image_height, image_width, 4))

    plt.title(name)
    plt.imshow(map_image)
    plt.axis('off')
    plt.savefig("/home/matthew/Pictures/Lab_Meeting_30_11_2021/Seed_Modulation_Maps/" + name + ".png")
    plt.close()


def get_mean_maps(session_list, map_filename):

    map_list = []

    for base_directory in session_list:
        map = np.load(os.path.join(base_directory, map_filename))
        map_list.append(map)

    map_list = np.array(map_list)
    map_list = np.nan_to_num(map_list)




    mean_map = np.mean(map_list, axis=0)

    # Load Mask
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(session_list[0])

    map_image = np.zeros(image_width * image_height)
    map_image[indicies] = mean_map
    map_image = np.ndarray.reshape(map_image, (image_height, image_width))

    plt.axis('off')


    plt.imshow(map_image, cmap='bwr', vmin=-0.5, vmax=0.5)
    plt.show()



def view_all_seed_maps(session_list):

    for base_directory in session_list:

        # Load Mask
        indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

        # Get Visual Maps
        v1_visual_correlation_map = np.load(base_directory + "/V1_Visual_Correlation_Map.npy")
        v1_odour_correlation_map = np.load(base_directory + "/V1_Odour_Correlation_Map.npy")
        difference_map = np.subtract(v1_visual_correlation_map, v1_odour_correlation_map)

        # Convert To Images
        v1_visual_correlation_map = Widefield_General_Functions.create_image_from_data(v1_visual_correlation_map, indicies, image_height, image_width)
        v1_odour_correlation_map = Widefield_General_Functions.create_image_from_data(v1_odour_correlation_map, indicies, image_height, image_width)
        v1_difference_map = Widefield_General_Functions.create_image_from_data(difference_map, indicies, image_height, image_width)

        figure_1 = plt.figure()
        visual_context_axis = figure_1.add_subplot(1, 3, 1)
        odour_context_axis = figure_1.add_subplot(1, 3, 2)
        difference_axis = figure_1.add_subplot(1, 3, 3)

        visual_context_axis.imshow(v1_visual_correlation_map, cmap='bwr', vmin=-0.5, vmax=0.5)
        odour_context_axis.imshow(v1_odour_correlation_map, cmap='bwr', vmin=-0.5, vmax=0.5)
        difference_axis.imshow(v1_difference_map, cmap='bwr', vmin=-0.5, vmax=0.5)

        plt.show()
        print("V1 correlation map", np.shape(v1_visual_correlation_map))




def view_signifiance(session_list):

    context_1_map_list = []
    context_2_map_list = []
    difference_map_list = []

    for base_directory in session_list:

        # Load Mask
        indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

        # Get Visual Maps
        v1_visual_correlation_map = np.load(base_directory + "/V1_Visual_Correlation_Map.npy")
        v1_odour_correlation_map = np.load(base_directory + "/V1_Odour_Correlation_Map.npy")
        difference_map = np.subtract(v1_visual_correlation_map, v1_odour_correlation_map)

        context_1_map_list.append(v1_visual_correlation_map)
        context_2_map_list.append(v1_odour_correlation_map)
        difference_map_list.append(difference_map)


    t_stats, p_values = stats.ttest_rel(context_1_map_list, context_2_map_list)
    t_stats = np.abs(t_stats)

    image = Widefield_General_Functions.create_image_from_data(t_stats, indicies, image_height, image_width)
    image = ndimage.gaussian_filter(image, sigma=3)
    plt.imshow(image, cmap='jet')
    plt.show()


        # Convert To Images
        #v1_visual_correlation_map = Widefield_General_Functions.create_image_from_data(v1_visual_correlation_map, indicies, image_height, image_width)
        #v1_odour_correlation_map = Widefield_General_Functions.create_image_from_data(v1_odour_correlation_map, indicies, image_height, image_width)
        #v1_difference_map = Widefield_General_Functions.create_image_from_data(difference_map, indicies, image_height, image_width)




controls = ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_02_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_08_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_23_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_31_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_15_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_10_29_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_11_05_Transition_Imaging"]

view_all_seed_maps(controls)
view_signifiance(controls)



