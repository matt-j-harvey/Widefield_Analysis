import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

import Registration_Utils


def load_mask_details(base_directory):
    mask_details_dict = np.load(os.path.join(base_directory, "Downsampled_mask_dict.npy"), allow_pickle=True)[()]
    indicies = mask_details_dict["indicies"]
    image_height = mask_details_dict["image_height"]
    image_width = mask_details_dict["image_width"]
    return indicies, image_height, image_width


def extract_blue_example_images(session_list):

    for base_directory in tqdm(session_list):

        # Load Blue Data
        motion_corrected_data_file = os.path.join(base_directory, "Motion_Corrected_Downsampled_Data.hdf5")
        motion_corrected_data_container = h5py.File(motion_corrected_data_file, mode="r")
        motion_corrected_blue_data = motion_corrected_data_container["Blue_Data"]
        number_of_pixels, number_of_frames = np.shape(motion_corrected_blue_data)
        blue_frame = motion_corrected_blue_data[:, 0]

        # Load Mask Dictionary
        indicies, image_height, image_width = load_mask_details(base_directory)

        # Reconstruct_Blue_Frame
        blue_frame = Registration_Utils.create_image_from_data(blue_frame, indicies, image_height, image_width)

        # Save Blue Frame
        np.save(os.path.join(base_directory, "Blue_Example_Image.npy"), blue_frame)

        # Close Corrected Data File
        motion_corrected_data_container.close()




session_list = [
    #r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_23_Transition_Imaging",
    #r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_31_Transition_Imaging",
    #r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_04_02_Transition_Imaging",

    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_02_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_08_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_10_Transition_Imaging",

    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_13_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_15_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_17_Transition_Imaging",

    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_10_29_Transition_Imaging",
    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_11_03_Transition_Imaging",
    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_11_05_Transition_Imaging"

]

extract_blue_example_images(session_list)