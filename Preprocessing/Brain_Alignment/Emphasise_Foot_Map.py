import numpy as np
import matplotlib.pyplot as plt
import tables
from scipy import ndimage
from skimage.feature import canny
from skimage.filters import sobel
import math
from pathlib import Path

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph
from skimage.transform import warp, resize, rescale

import os
import sys

import Registration_Utils

pyqtgraph.setConfigOptions(imageAxisOrder='row-major')

def combine_elements_into_path(list):
    current_path = list[0]
    for item in list[1:]:
        current_path = os.path.join(current_path, item)
    return current_path



def load_cpd_map(regression_dictionary, group_name, indicies, image_height, image_width, alignment_dict):

    # Load CPDs
    regression_group_names = regression_dictionary["Coef_Names"]
    regression_cpds = regression_dictionary["Coefficients_of_Partial_Determination"]
    regression_cpds = np.transpose(regression_cpds)

    # Create Image
    selected_cpd_index = regression_group_names.index(group_name)
    selected_cpd = regression_cpds[selected_cpd_index]
    selected_cpd = Registration_Utils.create_image_from_data(selected_cpd, indicies, image_height, image_width)

    # Align Image
    selected_cpd = Registration_Utils.transform_image(selected_cpd, alignment_dict)

    return selected_cpd

def get_foot_map(session_list):

    mouse_limbs_list = []

    for base_directory in session_list:

        # Load Regression Dict
        regression_dictionary = np.load(os.path.join(base_directory, "Regression_Coefs", "Regression_Dictionary_Simple.npy"), allow_pickle=True)[()]

        # Load Mask
        indicies, image_height, image_width = Registration_Utils.load_downsampled_mask(base_directory)

        # Load Within Mouse Alignment Dictionary
        within_mouse_alignment_dictionary = np.load(os.path.join(base_directory, "Within_Mouse_Alignment_Dictionary.npy"), allow_pickle=True)[()]

        # Load CPD Maps
        session_limbs_map   = load_cpd_map(regression_dictionary, 'Limbs',          indicies, image_height, image_width, within_mouse_alignment_dictionary)
        #session_limbs_map = ndimage.gaussian_filter(session_limbs_map, sigma=4)
        #limb_map = sobel(session_limbs_map)

        #plt.imshow(limb_map)
        #plt.show()

        # Add To List
        mouse_limbs_list.append(session_limbs_map)

    # Get Mean Maps
    mouse_limbs_map = np.mean(mouse_limbs_list, axis=0)
    epmhasised_limb_map = np.zeros(np.shape(mouse_limbs_map))
    epmhasised_limb_map[100:200, 50:250] = mouse_limbs_map[100:200, 50:250]
    plt.title("mean limbs")
    plt.imshow(epmhasised_limb_map)
    plt.show()

def view_limb_regressors(session_list):


    mouse_limbs_list = []

    for base_directory in session_list:

        # Load Regression Dict
        regression_dictionary = np.load(os.path.join(base_directory, "Regression_Coefs", "Regression_Dictionary_Simple.npy"), allow_pickle=True)[()]

        # Load Mask
        indicies, image_height, image_width = Registration_Utils.load_downsampled_mask(base_directory)

        # Load Within Mouse Alignment Dictionary
        within_mouse_alignment_dictionary = np.load(os.path.join(base_directory, "Within_Mouse_Alignment_Dictionary.npy"), allow_pickle=True)[()]

        # Load CPD Maps
        session_coefs = regression_dictionary["Coefs"]
        session_coefs = np.transpose(session_coefs)
        regressor_names = regression_dictionary["Coef_Names"]
        limb_index = regressor_names.index("Limbs")

        coef_group_starts = regression_dictionary["Coef_Group_Starts"]
        coef_group_stops = regression_dictionary["Coef_Group_Stops"]

        limb_start = coef_group_starts[limb_index]
        limb_stop = coef_group_stops[limb_index]

        limb_regressors = session_coefs[limb_start:limb_stop]
        print("Limb regressors", np.shape(limb_regressors))
        regressor_map_list = []
        for regressor in limb_regressors:
            regressor_map = Registration_Utils.create_image_from_data(regressor, indicies, image_height, image_width)
            regressor_map = np.abs(regressor_map)
            regressor_map = np.divide(regressor_map, np.percentile(regressor_map, q=90))
            regressor_map = Registration_Utils.transform_image(regressor_map, within_mouse_alignment_dictionary)
            regressor_map_list.append(regressor_map)

        mean_regressor = np.mean(regressor_map_list, axis=0)
        mouse_limbs_list.append(mean_regressor)

    mean_regressor = np.mean(mouse_limbs_list, axis=0)
    mean_regressor = ndimage.gaussian_filter(mean_regressor, sigma=1)

    # Save This
    root_directory = Path(session_list[0]).parts[0:-1]
    root_directory = combine_elements_into_path(root_directory)
    save_directory = os.path.join(root_directory, "Limb_Map")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    np.save(os.path.join(save_directory, "Foot_Map.npy"), mean_regressor)



    plt.title("MEan Regressor")
    plt.imshow(mean_regressor, cmap='jet')
    plt.show()


mouse_list = [

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_13_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_15_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_17_Switching_Imaging"],

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_31_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_02_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_04_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_08_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_10_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_12_Transition_Imaging"],

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_20_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_22_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_24_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_14_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_16_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_18_Transition_Imaging"],

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_17_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_19_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_23_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_30_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_06_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_08_Transition_Imaging"],

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_17_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_19_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_22_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_24_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_26_Transition_Imaging"],

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_14_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_20_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_26_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_05_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_08_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_10_Transition_Imaging"],

    [r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_21_Switching_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_23_Switching_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_11_Switching_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_13_Transition_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_15_Transition_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_17_Transition_Imaging"],

    [r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_14_Switching_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_20_Switching_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_22_Switching_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_29_Transition_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_03_Transition_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_05_Transition_Imaging"],


    [r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_05_Switching_Imaging",
     r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_09_Switching_Imaging"],

    [r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_29_Switching_Imaging",
     r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_05_Switching_Imaging"],

    [r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_02_Switching_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_04_Switching_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_06_Switching_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_02_Transition_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_08_Transition_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_10_Transition_Imaging"],

    [r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_26_Switching_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_28_Switching_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_02_Switching_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_23_Transition_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_31_Transition_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_04_02_Transition_Imaging"],

]

for session_list in mouse_list:
    view_limb_regressors(session_list)
#get_foot_map(session_list)
