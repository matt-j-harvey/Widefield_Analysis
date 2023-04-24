import numpy as np
import matplotlib.pyplot as plt
import tables
from tqdm import tqdm
import os

from Widefield_Utils import widefield_utils


def get_mean_intesity_trace(base_directory):

    # Load Data
    opto_data_file = os.path.join(base_directory, "Downsampled_Delta_F.h5")
    data_container = tables.open_file(opto_data_file, mode="r")
    data = data_container.root["Data"]

    # Get Mean Frames
    mean_frame_list = []
    for frame in tqdm(data):
        mean_frame_list.append(np.mean(frame))

    # Save
    mean_file = os.path.join(base_directory, "Mean_Frame_intensities.npy")
    np.save(mean_file, mean_frame_list)



base_directory = r"/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/Calibration/Opto_Test_2023_04_11"
get_mean_intesity_trace(base_directory)
#plot_mean_intensity_trace(base_directory)
#plot_mean_intensity_trace_with_visual_stim(base_directory)