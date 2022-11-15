import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

def get_matched_bodycam_data(base_directory):

    bodycam_transformed_data = np.load(os.path.join(base_directory, "Mousecam_Analysis", "Transformed_Mousecam_Data.npy"))
    print("Transformed Bodycam Data Shape", np.shape(bodycam_transformed_data))

    # Load Widefied To Mousecam Frame Dict
    widefield_to_mousecam_frame_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy"), allow_pickle=True)[()]
    widefield_frame_keys = list(widefield_to_mousecam_frame_dict.keys())

    matched_bodycam_data = []
    for widefield_index in widefield_frame_keys:
        mousecam_index = widefield_to_mousecam_frame_dict[widefield_index]
        matched_bodycam_data.append(bodycam_transformed_data[mousecam_index])

    matched_bodycam_data = np.array(matched_bodycam_data)
    np.save(os.path.join(base_directory, "Mousecam_Analysis", "Matched_Bodycam_Transformed_Data.npy"), matched_bodycam_data)

base_directory = r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_10_29_Transition_Imaging"
get_matched_bodycam_data(base_directory)