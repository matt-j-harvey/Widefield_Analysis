import pandas as pd
import tables
import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec
import os
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
from sklearn.decomposition import TruncatedSVD, IncrementalPCA
from scipy import ndimage
from tqdm import tqdm
import pickle
import pybresenham
import cv2
from skimage.transform import rescale
from skimage.registration import optical_flow_tvl1, optical_flow_ilk

def get_deeplabcut_filename(base_directory):
    file_list = os.listdir(base_directory)
    for file_name in file_list:
        if "DLC_resnet101_Matt_Bodycam" in file_name:
            return file_name

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def extract_limb_positions(base_directory, deeplabcut_file_name, number_of_limbs):

    # Get Limb Positions
    deeplabcut_dataframe = pd.read_hdf(os.path.join(base_directory, deeplabcut_file_name))
    columns = deeplabcut_dataframe.columns

    # Extract Data
    x_coords_list = []
    y_coords_list = []
    probability_list = []
    for limb_index in range(number_of_limbs):
        limb_x_coords = deeplabcut_dataframe[columns[limb_index * 3 + 0]]
        limb_y_coords = deeplabcut_dataframe[columns[limb_index * 3 + 1]]
        limb_likelihood = deeplabcut_dataframe[columns[limb_index * 3 + 2]]

        x_coords_list.append(limb_x_coords)
        y_coords_list.append(limb_y_coords)
        probability_list.append(limb_likelihood)

    x_coords_list = np.array(x_coords_list)
    y_coords_list = np.array(y_coords_list)
    return x_coords_list, y_coords_list, probability_list


def match_limb_movements_to_widefield_motion(base_directory, transformed_limb_data):

    # Load Widefield To Mousecam Frame Dict
    widefield_to_mousecam_frame_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy"), allow_pickle=True)[()]
    widefield_frame_list = list(widefield_to_mousecam_frame_dict.keys())

    number_of_mousecam_frames = np.shape(transformed_limb_data)[0]

    # Match Whisker Activity To Widefield Frames
    matched_limb_data = []
    for widefield_frame in widefield_frame_list:
        corresponding_mousecam_frame = widefield_to_mousecam_frame_dict[widefield_frame]
        if corresponding_mousecam_frame < number_of_mousecam_frames:
            matched_limb_data.append(transformed_limb_data[corresponding_mousecam_frame])

    matched_limb_data = np.array(matched_limb_data)
    print("matched_limb_data", np.shape(matched_limb_data))

    matched_limb_data = np.squeeze(matched_limb_data)
    print("matched_limb_data", np.shape(matched_limb_data))

    return matched_limb_data



def plot_cumulative_explained_variance(explained_variance, save_directory):

    cumulative_variance = np.cumsum(explained_variance)
    x_values = list(range(1, len(cumulative_variance)+1))
    plt.title("Cumulative Explained Variance, Limb Movement PCA")
    plt.plot(x_values, cumulative_variance)
    plt.ylim([0, 1.1])
    plt.savefig(os.path.join(save_directory, "Limb_Cumulative_Explained_Variance.png"))
    plt.close()


def create_limb_movement_matrix(limb_x_coords, limb_y_coords, probability_list, probability_threshold=0.8):

    number_of_limbs, number_of_timepoints = np.shape(limb_x_coords)
    print("Number of limbs: ", number_of_limbs)
    print("Number of timepoints: ", number_of_timepoints)
    print("Probability List", np.shape(probability_list))

    x_diffs = np.diff(limb_x_coords, axis=1)
    y_diffs = np.diff(limb_y_coords, axis=1)
    print("x diff shape", np.shape(x_diffs))

    # Threshold By Probability
    for timepoint_index in range(1, number_of_timepoints):
        for limb_index in range(number_of_limbs):

            current_probability = probability_list[limb_index][timepoint_index]
            previous_probability = probability_list[limb_index][timepoint_index-1]

            if current_probability < probability_threshold or previous_probability < probability_threshold:
                x_diffs[limb_index, timepoint_index-1] = 0
                y_diffs[limb_index, timepoint_index-1] = 0


    # Stack Data Frames
    movement_matrix = np.vstack([x_diffs, y_diffs])

    # Decompose
    model = TruncatedSVD(n_components=8)
    transformed_data = model.fit_transform(np.transpose(movement_matrix))

    """
    print("Transformed Data Shape", np.shape(transformed_data))
    plt.imshow(movement_matrix[:, 0:100], vmin=-20, vmax=20, cmap='bwr')
    forceAspect(plt.gca())
    plt.show()

    plt.plot(transformed_data[:, 0])
    plt.plot(transformed_data[:, 1])
    plt.plot(transformed_data[:, 2])
    plt.plot(transformed_data[:, 3])
    plt.show()
    """

    return transformed_data, model




def extract_limb_movements(base_directory, probability_threshold=0.8, number_of_limbs=4, downscale_factor=8):

    # Create Save Directory
    save_directory = os.path.join(base_directory, "Mousecam_Analysis")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Get Deeplabcut FIle Name
    deeplabcut_file_name = get_deeplabcut_filename(base_directory)
    print("Using Deeplabcut File: ", deeplabcut_file_name)

    # Get Limb Positions
    limb_x_coords, limb_y_coords, probability_list = extract_limb_positions(base_directory, deeplabcut_file_name, number_of_limbs)

    # Create Limb Movement Matrix
    limb_movement_matrix, model = create_limb_movement_matrix(limb_x_coords, limb_y_coords, probability_list)

    # Save Model Details
    explained_variance = model.explained_variance_ratio_
    components = model.components_
    plot_cumulative_explained_variance(explained_variance, save_directory)

    # Match Limb Movements To Widefield Frames
    matched_limb_movements = match_limb_movements_to_widefield_motion(base_directory, limb_movement_matrix)
    print("Matched Limb Movement Shape", np.shape(matched_limb_movements))

    # Save Details
    np.save(os.path.join(save_directory, "Matched_Limb_Movements_Simple.npy"), matched_limb_movements)
    np.save(os.path.join(save_directory, "limb_components.npy"), components)
    np.save(os.path.join(save_directory, "limb_components_explained_variance.npy"), explained_variance)



session_list = [

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_13_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_15_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_17_Switching_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_31_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_02_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_04_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_08_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_10_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_12_Transition_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_20_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_22_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_24_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_14_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_16_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_18_Transition_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_17_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_19_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_23_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_30_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_06_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_08_Transition_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_15_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_17_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_19_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_22_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_24_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_26_Transition_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_14_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_20_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_26_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_05_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_08_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_10_Transition_Imaging",

        r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_28_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_05_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_09_Switching_Imaging",

        r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_29_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_05_Switching_Imaging",

        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_02_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_04_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_06_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_02_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_08_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_10_Transition_Imaging",

        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_26_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_28_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_02_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_23_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_31_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_04_02_Transition_Imaging",

        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_21_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_23_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_11_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_13_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_15_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_17_Transition_Imaging",

        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_14_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_20_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_22_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_29_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_03_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_05_Transition_Imaging",
    ]


for base_directory in session_list:
    extract_limb_movements(base_directory)
