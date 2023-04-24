import os

import cv2
import tables
import h5py
from tqdm import tqdm
from skimage.feature import canny

import numpy as np
import matplotlib.pyplot as plt

from Widefield_Utils import widefield_utils
from Behaviour_Analysis import Behaviour_Utils



def get_opto_detection_frame_times(frame_time_list, opto_frame_onsets):

    opto_frame_times = []
    for frame_index in opto_frame_onsets:
        frame_time = frame_time_list[frame_index]
        opto_frame_times.append(frame_time)

    return opto_frame_times


def check_projector_delay(base_directory, mean_intensity_threshold=18000):

    """
    Characterises the difference beween:
    Onset of projector trigger from teensy
    Time of camera frame when Stimulus Detected

    """

    # Load Mean Frame Intesity Files
    mean_frame_intensities = np.load(os.path.join(base_directory, "Mean_Frame_intensities.npy"))
    opto_frame_onsets = Behaviour_Utils.get_step_onsets(mean_frame_intensities, threshold=mean_intensity_threshold)

    # Load AI File
    ai_data = widefield_utils.load_ai_recorder_file(base_directory)
    stimuli_dictionary = widefield_utils.create_stimuli_dictionary()

    # Get Opto Onset Times
    opto_trace = ai_data[stimuli_dictionary["Optogenetics"]]
    opto_ai_onsets = Behaviour_Utils.get_step_onsets(opto_trace)
    np.save(os.path.join(base_directory, "Opto_Ai_Onsets.npy"), opto_ai_onsets)

    # Get Frame Times in Ms
    blue_led_trace = ai_data[stimuli_dictionary["LED 1"]]
    frame_onsets = Behaviour_Utils.get_step_onsets(blue_led_trace)
    np.save(os.path.join(base_directory, "Frame_Onsets.npy"), frame_onsets)

    # Get Time in Ms of Opto Detection Frames
    opto_frame_times = get_opto_detection_frame_times(frame_onsets, opto_frame_onsets)

    # Get Differecne Between Each AI Onset and Each Frame Onset
    delay_list = np.subtract(opto_frame_times, opto_ai_onsets)

    return delay_list


