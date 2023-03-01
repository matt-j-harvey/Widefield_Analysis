import os

import numpy as np
import matplotlib.pyplot as plt

from Widefield_Utils import widefield_utils
from Files import Session_List


def check_lick_aligned_hits(base_directory):

    lick_aligned_hits = np.load(os.path.join(base_directory,  "Stimuli_Onsets", "Lick_Aligned_Hits.npy"))
    downsampled_ai_matrix = np.load(os.path.join(base_directory, "Downsampled_AI_Matrix_Framewise.npy"))

    stimuli_dict = widefield_utils.create_stimuli_dictionary()

    lick_trace = downsampled_ai_matrix[stimuli_dict["Lick"]]
    vis_1_trace = downsampled_ai_matrix[stimuli_dict["Visual 1"]]

    for onset in lick_aligned_hits:
        trial_start = onset - 100
        trial_stop = onset + 100

        trial_lick_trace = lick_trace[trial_start:trial_stop]
        trial_vis_trace = vis_1_trace[trial_start:trial_stop]

        plt.plot(trial_lick_trace)
        plt.plot(trial_vis_trace)

    plt.show()


def get_lick_aligned_hits(base_directory):

    # Vis 1
    # Correct
    # Lick IS atleast 1 Second Post Visual Onset and Less Than 3

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)
    print("Behaviour matrix shape", np.shape(behaviour_matrix))

    # Load Frame Onsets
    frame_time_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    frame_times = list(frame_time_dict.keys())

    lick_aligned_hits = []
    for trial in behaviour_matrix:
        trial_type = trial[1]
        correct = trial[3]
        lick_onset = trial[22]
        stimuli_onset = trial[11]


        if trial_type == 1 and correct == 1:
            nearest_frame_onset = widefield_utils.take_closest(frame_times, lick_onset)
            nearest_frame = frame_time_dict[nearest_frame_onset]
            lick_aligned_hits.append(nearest_frame)

    np.save(os.path.join(base_directory,  "Stimuli_Onsets", "Lick_Aligned_Hits.npy"), lick_aligned_hits)

"""
control_pre_learning = Session_List.control_pre_learning_session_list
for base_directory in control_pre_learning:
    get_lick_aligned_hits(base_directory)


control_intermediate_learning = Session_List.control_intermediate_learning_session_list
for base_directory in control_intermediate_learning:
    get_lick_aligned_hits(base_directory)


control_post_learning = Session_List.control_post_learning_session_list
for base_directory in control_post_learning:
    get_lick_aligned_hits(base_directory)
"""

nxak_7_1_b = [

    # Pre
    ["NXAK7.1B/2021_02_03_Discrimination_Imaging",
    "NXAK7.1B/2021_02_05_Discrimination_Imaging",
    "NXAK7.1B/2021_02_07_Discrimination_Imaging",
    "NXAK7.1B/2021_02_09_Discrimination_Imaging"],

    # Int
    ["NXAK7.1B/2021_02_15_Discrimination_Imaging",
    "NXAK7.1B/2021_02_17_Discrimination_Imaging",
    "NXAK7.1B/2021_02_19_Discrimination_Imaging",
    "NXAK7.1B/2021_02_22_Discrimination_Imaging"],

    # Post
    ["NXAK7.1B/2021_02_24_Discrimination_Imaging"]

]

session_list = [

    [r"NRXN71.2A/2020_11_14_Discrimination_Imaging",
     r"NRXN71.2A/2020_12_09_Discrimination_Imaging"],

    [r"NXAK4.1A/2021_02_04_Discrimination_Imaging",
     r"NXAK4.1A/2021_03_05_Discrimination_Imaging"],

    [r"NXAK10.1A/2021_05_02_Discrimination_Imaging",
     r"NXAK10.1A/2021_05_14_Discrimination_Imaging"],

    [r"NXAK16.1B/2021_05_02_Discrimination_Imaging",
     r"NXAK16.1B/2021_06_15_Discrimination_Imaging"],

    [r"NXAK20.1B/2021_09_30_Discrimination_Imaging",
     r"NXAK20.1B/2021_10_19_Discrimination_Imaging"],

    [r"NXAK24.1C/2021_09_22_Discrimination_Imaging",
     r"NXAK24.1C/2021_10_08_Discrimination_Imaging"],
]

data_root_diretory = r"/media/matthew/External_Harddrive_1/Neurexin_Data"

for mouse in session_list:
    for session in mouse:
        get_lick_aligned_hits(os.path.join(data_root_diretory, session))