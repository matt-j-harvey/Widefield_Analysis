import os

import matplotlib.pyplot as plt
import numpy as np
import tables
from skimage.transform import resize
from sklearn.decomposition import TruncatedSVD

import pickle
from scipy import ndimage, stats
from tqdm import tqdm
import pathlib

from Files import Session_List
from Widefield_Utils import widefield_utils


def check_save_directory(output_root_directory, session_directory):

    # Split Path
    session_directory_parts = pathlib.Path(session_directory)
    session_directory_parts = list(session_directory_parts.parts)
    mouse_name = session_directory_parts[0]
    session_name = session_directory_parts[1]

    # Check Mouse Directory Exists
    mouse_directory = os.path.join(output_root_directory, mouse_name)
    if not os.path.exists(mouse_directory):
        os.mkdir(mouse_directory)

    # Create Save Directory
    save_directory = os.path.join(mouse_directory, session_name)
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    return save_directory



def register_data(session_directory, delta_f_matrix, gaussian_filter=True, gaussian_filter_sd=1):

    # Load Alignment Dictionaries
    within_mouse_alignment_dictionary = np.load(os.path.join(session_directory, "Within_Mouse_Alignment_Dictionary.npy"), allow_pickle=True)[()]
    across_mouse_alignment_dictionary = widefield_utils.load_across_mice_alignment_dictionary(session_directory)

    # Load Masks
    common_indicies, common_height, common_width = widefield_utils.load_tight_mask()

    # Register Each Frame
    aligned_delta_f_matrix = []

    number_of_frames = np.shape(delta_f_matrix)[0]

    for frame_index in range(number_of_frames):

        frame = delta_f_matrix[frame_index]

        # Remove NaNs
        frame = np.nan_to_num(frame)

        # Apply Gaussian Filter
        if gaussian_filter == True:
            frame = ndimage.gaussian_filter(frame, sigma=gaussian_filter_sd)

        # Register Within Mouse
        frame = widefield_utils.transform_image(frame, within_mouse_alignment_dictionary)

        # Register Across Mice
        frame = widefield_utils.transform_image(frame, across_mouse_alignment_dictionary)

        # Apply Shared Tight Mask
        frame = np.ndarray.flatten(frame)
        frame = frame[common_indicies]

        # Add To Registered Data
        aligned_delta_f_matrix.append(frame)


    aligned_delta_f_matrix = np.array(aligned_delta_f_matrix)

    return aligned_delta_f_matrix




def downsample_data(delta_f_matrix):

    # Load Tight Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()

    # Get Downsampled Mask
    downsampled_indicies, downsampled_height, downsampled_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

    # Downsample Frames
    downsampled_data = []

    for frame in delta_f_matrix:

        # Reconstruct Into 2D
        reconstructed_image = widefield_utils.create_image_from_data(frame, indicies, image_height, image_width)

        # Downsample
        downsampled_image = resize(reconstructed_image, (downsampled_height, downsampled_width), order=1, anti_aliasing=True)

        # Re-Mask
        downsampled_image = np.reshape(downsampled_image, (downsampled_height * downsampled_width))
        downsampled_image = downsampled_image[downsampled_indicies]

        downsampled_data.append(downsampled_image)

    downsampled_data = np.array(downsampled_data)
    return downsampled_data


def moving_average(a, n=3) :

    number_of_frames, number_of_pixels = np.shape(a)

    smoothed_matrix = np.zeros(np.shape(a))
    for frame_index in range(number_of_frames):

        window_start = frame_index - n
        if window_start < 0: window_start = 0

        temporal_window = a[window_start:frame_index]
        smoothed_matrix[frame_index] = np.mean(temporal_window, axis=0)

    smoothed_matrix = np.nan_to_num(smoothed_matrix)
    return smoothed_matrix





def preprocess_session(session_directory):

    # Load Delta F Data
    corrected_svt = np.load(os.path.join(session_directory, "Churchland_Preprocessing", "Corrected_SVT.npy"))
    u = np.load(os.path.join(session_directory, "Churchland_Preprocessing", "U.npy"))

    # Get Chunk Structure
    number_of_components, number_of_frames = np.shape(corrected_svt)
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = widefield_utils.get_chunk_structure(chunk_size=1000, array_size=number_of_frames)

    delta_f_matrix = []
    for chunk_index in tqdm(range(number_of_chunks), desc="Preprocessing_Data"):
        chunk_start = chunk_starts[chunk_index]
        chunk_stop = chunk_stops[chunk_index]

        # Extract Chunk
        delta_f_chunk = corrected_svt[:, chunk_start:chunk_stop]

        # Transform To Pixel Space
        delta_f_chunk = np.dot(u, delta_f_chunk)

        # Reshape Array
        delta_f_chunk = np.moveaxis(delta_f_chunk, [2], [0])

        # Register Data and Apply Gaussian
        delta_f_chunk = register_data(session_directory, delta_f_chunk)

        # Downsample
        delta_f_chunk = downsample_data(delta_f_chunk)

        # Add To Matrix
        delta_f_matrix.append(delta_f_chunk)

    # Concatenate CHunks
    delta_f_matrix = np.vstack(delta_f_matrix)
    print("Delta F Matrix Shape", np.shape(delta_f_matrix))

    # Save This
    np.save(os.path.join(session_directory, "Delta_F_Matrix_100_by_100_SVD.npy"), delta_f_matrix)
    """

    delta_f_matrix = np.load(os.path.join(session_directory, "Delta_F_Matrix_100_by_100_SVD.npy"))
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width )
    colourmap = widefield_utils.get_musall_cmap()
    plt.ion()
    for frame in delta_f_matrix[3000:]:
        frame = widefield_utils.create_image_from_data(frame, indicies, image_height, image_width)
        plt.imshow(frame, vmin=-0.05, vmax=0.05, cmap=colourmap)
        plt.draw()
        plt.pause(0.1)
        plt.clf()
    
    """



"""
Preprocessing Steps:

1.) Takes Raw Delta F Data

2.) Registers It to A Common Space

3.) Applies A Gaussian Filter

4.) Applies A Shared Tight Mask

5.) Further Downsample To 100 x 100

"""


# Load Session List


control_switching_sessions = [

    r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_05_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_09_Switching_Imaging",

    r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_29_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_05_Switching_Imaging",

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
    #r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_04_02_Transition_Imaging",

]


selected_session_list = [

    #r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_05_Switching_Imaging",
    #r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_09_Switching_Imaging",

    #r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_29_Switching_Imaging",
    #r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_05_Switching_Imaging",

    #r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_21_Switching_Imaging",
    #r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_23_Switching_Imaging",
    #r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_11_Switching_Imaging",
    #r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_13_Transition_Imaging",
    #r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_15_Transition_Imaging",
    #r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_17_Transition_Imaging",

    #r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_14_Switching_Imaging",
    #r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_20_Switching_Imaging",
    #r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_22_Switching_Imaging",
    #r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_29_Transition_Imaging",
    #r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_03_Transition_Imaging",
    #r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_05_Transition_Imaging",

    #r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_02_Switching_Imaging",
    #r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_04_Switching_Imaging",
    #r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_06_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_02_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_08_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_10_Transition_Imaging",

    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_26_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_28_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_02_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_23_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_31_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_04_02_Transition_Imaging",

]

selected_session_list = [
                    #"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_14_Discrimination_Imaging",
                    #"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging",
                    #"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_16_Discrimination_Imaging",
                    #"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_17_Discrimination_Imaging",
                    #"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_19_Discrimination_Imaging",
                    #"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_21_Discrimination_Imaging",
                    #"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_24_Discrimination_Imaging",

                    #"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_14_Discrimination_Imaging",
                    #"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_15_Discrimination_Imaging",
                    #"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_16_Discrimination_Imaging",
                    #"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_17_Discrimination_Imaging",
                    #"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_19_Discrimination_Imaging",
                    #"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_21_Discrimination_Imaging",
                    #"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_23_Discrimination_Imaging",
                    #"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_25_Discrimination_Imaging",

                    #"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_04_Discrimination_Imaging",
                    #"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_06_Discrimination_Imaging",
                    #"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_08_Discrimination_Imaging",
                    #"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_10_Discrimination_Imaging",
                    "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_12_Discrimination_Imaging",
                    "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_14_Discrimination_Imaging",
                    "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_22_Discrimination_Imaging",

                    "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_01_Discrimination_Imaging",
                    "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_03_Discrimination_Imaging",
                    "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_05_Discrimination_Imaging",
                    "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_07_Discrimination_Imaging",
                    "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_09_Discrimination_Imaging",
                    "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_11_Discrimination_Imaging",
                    "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_13_Discrimination_Imaging",
                    "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_15_Discrimination_Imaging",
                    "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_17_Discrimination_Imaging",
                    "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_19_Discrimination_Imaging",
                    "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_22_Discrimination_Imaging",
                    "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_24_Discrimination_Imaging",

                    "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_04_29_Discrimination_Imaging",
                    "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_01_Discrimination_Imaging",
                    "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_03_Discrimination_Imaging",
                    "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_05_Discrimination_Imaging",
                    "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_07_Discrimination_Imaging",
                    "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_09_Discrimination_Imaging",

                    "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_25_Discrimination_Imaging",
                    "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_29_Discrimination_Imaging",
                    "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_01_Discrimination_Imaging",
                    "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_03_Discrimination_Imaging",
                    "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_05_Discrimination_Imaging",
                    "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_07_Discrimination_Imaging",
                    "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging",


                    ]


# Mutant Switching Sessions
selected_session_list = [

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
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_17_Switching_Imaging", ## here on 28/01
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
    ]



session_list = [

     #"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_13_Discrimination_Imaging", - Too big to do in 1
     #"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_14_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_15_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_16_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_17_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_19_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_21_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_23_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_25_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_27_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_29_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_01_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_03_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_05_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_07_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_09_Discrimination_Imaging",

     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_02_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_04_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_06_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_08_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_10_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_12_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_14_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_16_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_18_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_23_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_25_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_27_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_01_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_03_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging",

     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_04_30_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_02_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_04_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_06_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_08_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_10_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_12_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_14_Discrimination_Imaging",


     #"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_04_30_Discrimination_Imaging", # motion corected file not downloaded
     #"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_02_Discrimination_Imaging", # motion corected file not downloaded
     #"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_04_Discrimination_Imaging", # motion corected file not downloaded
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_06_Discrimination_Imaging",
    ]

session_list = [
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_08_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_10_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_12_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_14_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_16_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_18_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_20_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_22_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_24_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_26_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_04_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_15_Discrimination_Imaging",

     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_28_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_30_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_02_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_04_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_06_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_09_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_11_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_13_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_15_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_17_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging",

     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_20_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_22_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_24_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_26_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_28_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_30_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_02_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_04_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_06_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_08_Discrimination_Imaging",
]
session_list = ["/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_14_Discrimination_Imaging"]
# Iterate Through Session List
for session_directory in tqdm(session_list, desc="Session"):
    print(session_directory)
    preprocess_session(session_directory)

