import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import Regression_Utils


def smooth_trace(trace, N=5):
    trace = np.convolve(trace, np.ones(N) / N, mode='same')
    return trace

def get_running_onsets(base_directory):

    # Load Downsampled AI
    downsampled_ai_matrix = np.load(os.path.join(base_directory, "Downsampled_AI_Matrix_Framewise.npy"))
    number_of_frames = np.shape(downsampled_ai_matrix)[1]

    # Extract Running Trace
    stimuli_dict = Regression_Utils.create_stimuli_dictionary()
    running_trace = downsampled_ai_matrix[stimuli_dict["Running"]]

    # Load Running Baseline
    running_baseline = np.load(os.path.join(base_directory, "Running_Baseline.npy"))

    # Remove Baseline
    running_trace = np.subtract(running_trace, running_baseline)
    running_trace = np.clip(running_trace, a_min=0, a_max=None)

    # Smooth With Moving Average
    smooth_running_trace = smooth_trace(running_trace)

    # Get Running Onsets
    running_onsets = Regression_Utils.get_step_onsets(smooth_running_trace,  threshold=0.1, window=20)

    # Get Running Events
    running_events = np.zeros(number_of_frames)
    for onset in running_onsets:
        running_events[onset] = 1

    # Save These
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "Running_Onsets.npy"), running_onsets)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "Running_Events.npy"), running_events)

    """
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.plot(running_trace)
    axis_1.plot(smooth_running_trace, c='g')
    axis_1.axhline(0)
    for onset in running_onsets:
        axis_1.axvline(onset, c='k')
    plt.show()
    """

"""
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

session_list = ["/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Switching_Opto/KPGC2.2G/2022_12_02_Switching_Opto"]

for base_directory in session_list:
    get_running_onsets(base_directory)
"""