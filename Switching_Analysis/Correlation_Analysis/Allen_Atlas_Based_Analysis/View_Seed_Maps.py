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




controls = ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging/"]
            #"/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN78.1A/2020_12_09_Switching_Imaging/",
            #"/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN78.1D/2020_11_29_Switching_Imaging/"]

mutants = [ "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK12.1F/2021_09_22_Transition_Imaging/"]
            #"/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN71.2A/2020_12_17_Switching_Imaging/"]


visual_onsets_file = "visual_context_stable_vis_2_frame_onsets.npy"
odour_onsets_file = "odour_context_stable_vis_2_frame_onsets.npy"
trial_start = 0
trial_stop = 40
v1  = [45, 46]
pmv = [47, 48]
amv = [39, 40]
rsc = [32, 28]
m2  = [8, 9]
s1 = [21, 24]

"""
get_mean_maps(controls, "V1_Visual_Correlation_Map.npy")
get_mean_maps(controls, "PMV_Visual_Correlation_Map.npy")
get_mean_maps(controls, "AMV_Visual_Correlation_Map.npy")
get_mean_maps(controls, "RSC_Visual_Correlation_Map.npy")
get_mean_maps(controls, "S1_Visual_Correlation_Map.npy")
get_mean_maps(controls, "M2_Visual_Correlation_Map.npy")


get_mean_maps(mutants, "V1_Visual_Correlation_Map.npy")
get_mean_maps(mutants, "PMV_Visual_Correlation_Map.npy")
get_mean_maps(mutants, "AMV_Visual_Correlation_Map.npy")
get_mean_maps(mutants, "RSC_Visual_Correlation_Map.npy")
"""

v1_diff_list = []
amv_diff_list = []
pmv_diff_list = []
rsc_diff_list = []
s1_diff_list = []
m2_diff_list = []

for base_directory in controls:

    print(base_directory)

    # Get Visual Maps
    v1_visual_correlation_map  = np.load(base_directory + "/V1_Visual_Correlation_Map.npy")
    pmv_visual_correlation_map = np.load(base_directory + "/PMV_Visual_Correlation_Map.npy")
    amv_visual_correlation_map = np.load(base_directory + "/AMV_Visual_Correlation_Map.npy")
    rsc_visual_correlation_map = np.load(base_directory + "/RSC_Visual_Correlation_Map.npy")
    s1_visual_correlation_map = np.load(base_directory + "/S1_Visual_Correlation_Map.npy")
    m2_visual_correlation_map = np.load(base_directory + "/M2_Visual_Correlation_Map.npy")

    # Get Odour Maps
    v1_odour_correlation_map  = np.load(base_directory + "/V1_Odour_Correlation_Map.npy")
    pmv_odour_correlation_map = np.load(base_directory + "/PMV_Odour_Correlation_Map.npy")
    amv_odour_correlation_map = np.load(base_directory + "/AMV_Odour_Correlation_Map.npy")
    rsc_odour_correlation_map = np.load(base_directory + "/RSC_Odour_Correlation_Map.npy")
    s1_odour_correlation_map = np.load(base_directory + "/S1_Odour_Correlation_Map.npy")
    m2_odour_correlation_map = np.load(base_directory + "/M2_Odour_Correlation_Map.npy")

    v1_diff = np.diff([v1_visual_correlation_map, v1_odour_correlation_map], axis=0)[0]
    amv_diff = np.diff([amv_visual_correlation_map, amv_odour_correlation_map], axis=0)[0]
    pmv_diff = np.diff([pmv_visual_correlation_map, pmv_odour_correlation_map], axis=0)[0]
    rsc_diff = np.diff([rsc_visual_correlation_map, rsc_odour_correlation_map], axis=0)[0]
    s1_diff = np.diff([s1_visual_correlation_map, s1_odour_correlation_map], axis=0)[0]
    m2_diff = np.diff([m2_visual_correlation_map, m2_odour_correlation_map], axis=0)[0]

    v1_diff_list.append(v1_diff)
    amv_diff_list.append(amv_diff)
    pmv_diff_list.append(pmv_diff)
    rsc_diff_list.append(rsc_diff)
    s1_diff_list.append(s1_diff)
    m2_diff_list.append(m2_diff)

v1_diff_list = np.array(v1_diff_list)
amv_diff_list = np.array(amv_diff_list)
pmv_diff_list = np.array(pmv_diff_list)
rsc_diff_list = np.array(rsc_diff_list)
s1_diff_list = np.array(s1_diff_list)
m2_diff_list = np.array(m2_diff_list)

v1_diff_list = np.nan_to_num(v1_diff_list)
amv_diff_list = np.nan_to_num(amv_diff_list)
pmv_diff_list = np.nan_to_num(pmv_diff_list)
rsc_diff_list = np.nan_to_num(rsc_diff_list)
s1_diff_list = np.nan_to_num(s1_diff_list)
m2_diff_list = np.nan_to_num(m2_diff_list)



view_seed_map(base_directory, np.mean(v1_diff_list,  axis=0), v1, name="V1 Modulation Controls")
view_seed_map(base_directory, np.mean(amv_diff_list, axis=0), amv, name="AMV Modulation Controls")
view_seed_map(base_directory, np.mean(pmv_diff_list, axis=0), pmv, name="PMV Modulation Controls")
view_seed_map(base_directory, np.mean(rsc_diff_list, axis=0), rsc, name="RSC Modulation Controls")
view_seed_map(base_directory, np.mean(s1_diff_list,  axis=0), s1, name="S1 Modulation Controls")
view_seed_map(base_directory, np.mean(m2_diff_list,  axis=0), m2, name='M2 Modulation Controls')





v1_diff_list = []
amv_diff_list = []
pmv_diff_list = []
rsc_diff_list = []
s1_diff_list = []
m2_diff_list = []

for base_directory in mutants:

    print(base_directory)

    # Get Visual Maps
    v1_visual_correlation_map  = np.load(base_directory + "/V1_Visual_Correlation_Map.npy")
    pmv_visual_correlation_map = np.load(base_directory + "/PMV_Visual_Correlation_Map.npy")
    amv_visual_correlation_map = np.load(base_directory + "/AMV_Visual_Correlation_Map.npy")
    rsc_visual_correlation_map = np.load(base_directory + "/RSC_Visual_Correlation_Map.npy")
    s1_visual_correlation_map = np.load(base_directory + "/S1_Visual_Correlation_Map.npy")
    m2_visual_correlation_map = np.load(base_directory + "/M2_Visual_Correlation_Map.npy")

    # Get Odour Maps
    v1_odour_correlation_map  = np.load(base_directory + "/V1_Odour_Correlation_Map.npy")
    pmv_odour_correlation_map = np.load(base_directory + "/PMV_Odour_Correlation_Map.npy")
    amv_odour_correlation_map = np.load(base_directory + "/AMV_Odour_Correlation_Map.npy")
    rsc_odour_correlation_map = np.load(base_directory + "/RSC_Odour_Correlation_Map.npy")
    s1_odour_correlation_map = np.load(base_directory + "/S1_Odour_Correlation_Map.npy")
    m2_odour_correlation_map = np.load(base_directory + "/M2_Odour_Correlation_Map.npy")

    v1_diff = np.diff([v1_visual_correlation_map, v1_odour_correlation_map], axis=0)[0]
    amv_diff = np.diff([amv_visual_correlation_map, amv_odour_correlation_map], axis=0)[0]
    pmv_diff = np.diff([pmv_visual_correlation_map, pmv_odour_correlation_map], axis=0)[0]
    rsc_diff = np.diff([rsc_visual_correlation_map, rsc_odour_correlation_map], axis=0)[0]
    s1_diff = np.diff([s1_visual_correlation_map, s1_odour_correlation_map], axis=0)[0]
    m2_diff = np.diff([m2_visual_correlation_map, m2_odour_correlation_map], axis=0)[0]

    v1_diff_list.append(v1_diff)
    amv_diff_list.append(amv_diff)
    pmv_diff_list.append(pmv_diff)
    rsc_diff_list.append(rsc_diff)
    s1_diff_list.append(s1_diff)
    m2_diff_list.append(m2_diff)

v1_diff_list = np.array(v1_diff_list)
amv_diff_list = np.array(amv_diff_list)
pmv_diff_list = np.array(pmv_diff_list)
rsc_diff_list = np.array(rsc_diff_list)
s1_diff_list = np.array(s1_diff_list)
m2_diff_list = np.array(m2_diff_list)

v1_diff_list = np.nan_to_num(v1_diff_list)
amv_diff_list = np.nan_to_num(amv_diff_list)
pmv_diff_list = np.nan_to_num(pmv_diff_list)
rsc_diff_list = np.nan_to_num(rsc_diff_list)
s1_diff_list = np.nan_to_num(s1_diff_list)
m2_diff_list = np.nan_to_num(m2_diff_list)



view_seed_map(base_directory, np.mean(v1_diff_list,  axis=0), v1, name="V1 Modulation Mutants")
view_seed_map(base_directory, np.mean(amv_diff_list, axis=0), amv, name="AMV Modulation Mutants")
view_seed_map(base_directory, np.mean(pmv_diff_list, axis=0), pmv, name="PMV Modulation Mutants")
view_seed_map(base_directory, np.mean(rsc_diff_list, axis=0), rsc, name="RSC Modulation Mutants")
view_seed_map(base_directory, np.mean(s1_diff_list,  axis=0), s1, name="S1 Modulation Mutants")
view_seed_map(base_directory, np.mean(m2_diff_list,  axis=0), m2, name='M2 Modulation Mutants')

