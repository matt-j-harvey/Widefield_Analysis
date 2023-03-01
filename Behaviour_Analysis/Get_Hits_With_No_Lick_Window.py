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


def visualise_traces(base_directory):

    # Load Onsets
    onsets_list = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Vis_Aligned_Hits_atleast_500.npy"))
    print("ONsets", onsets_list)

    # Load AI Matrix
    downsampled_ai_matrix = np.load(os.path.join(base_directory, "Downsampled_AI_Matrix_Framewise.npy"))

    # Create Simuli Dict
    stimuli_dict = widefield_utils.create_stimuli_dictionary()
    vis_1_trace = downsampled_ai_matrix[stimuli_dict["Visual 1"]]
    lick_trace = downsampled_ai_matrix[stimuli_dict["Lick"]]

    for onset in onsets_list:
        trial_start = onset - 100
        trial_stop = onset + 100
        plt.plot(vis_1_trace[trial_start:trial_stop], c='b', alpha=0.5)
        plt.plot(lick_trace[trial_start:trial_stop], c='tab:orange', alpha=0.5)

    plt.show()

def get_lick_aligned_hits(base_directory):

    # Vis 1
    # Correct
    # Lick IS atleast 1 Second Post Visual Onset and Less Than 3

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

    # Load AI Matrix
    downsampled_ai_matrix = np.load(os.path.join(base_directory, "Downsampled_AI_Matrix_Framewise.npy"))

    # Load Lick Threshold
    lick_threshold = np.load(os.path.join(base_directory, "Lick_Threshold.npy"))

    # Create Simuli Dict
    stimuli_dict = widefield_utils.create_stimuli_dictionary()
    vis_1_trace = downsampled_ai_matrix[stimuli_dict["Visual 1"]]
    lick_trace = downsampled_ai_matrix[stimuli_dict["Lick"]]

    # Load Frame Onsets
    frame_time_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    frame_times = list(frame_time_dict.keys())

    lick_aligned_hits = []
    for trial in behaviour_matrix:
        trial_type = trial[1]
        correct = trial[3]
        lick_onset = trial[22]
        stimuli_onset = trial[11]
        closest_onset_frame = trial[18]
        reaction_time = lick_onset - stimuli_onset

        if closest_onset_frame != None:

            # Get Lick Pre Window
            trial_window_start = closest_onset_frame - 14
            preceeding_lick_window = lick_trace[trial_window_start:closest_onset_frame]
            print("Trial Window Start", trial_window_start)
            if trial_window_start > 0:
                print("PReceeding Lick Window", np.shape(preceeding_lick_window), "Max", np.max(preceeding_lick_window), "Lick Threshold", lick_threshold)


                if np.max(preceeding_lick_window) < lick_threshold:
                    if trial_type == 1 and correct == 1 and reaction_time > 500:
                        nearest_frame_onset = widefield_utils.take_closest(frame_times, stimuli_onset)
                        nearest_frame = frame_time_dict[nearest_frame_onset]
                        lick_aligned_hits.append(nearest_frame)

    np.save(os.path.join(base_directory,  "Stimuli_Onsets", "Vis_Aligned_Hits_atleast_500.npy"), lick_aligned_hits)


session_list = [
"NRXN78.1A/2020_11_15_Discrimination_Imaging",
"NRXN78.1A/2020_11_17_Discrimination_Imaging",
"NRXN78.1A/2020_11_19_Discrimination_Imaging",
"NRXN78.1A/2020_11_21_Discrimination_Imaging",
"NRXN78.1D/2020_11_15_Discrimination_Imaging",
"NRXN78.1D/2020_11_21_Discrimination_Imaging",
"NRXN78.1D/2020_11_23_Discrimination_Imaging",
"NRXN78.1D/2020_11_25_Discrimination_Imaging",
"NXAK4.1B/2021_02_04_Discrimination_Imaging",
"NXAK4.1B/2021_02_06_Discrimination_Imaging",
"NXAK4.1B/2021_02_08_Discrimination_Imaging",
"NXAK4.1B/2021_02_10_Discrimination_Imaging",
"NXAK4.1B/2021_02_14_Discrimination_Imaging",
"NXAK4.1B/2021_02_22_Discrimination_Imaging",
"NXAK7.1B/2021_02_01_Discrimination_Imaging",
"NXAK7.1B/2021_02_03_Discrimination_Imaging",
"NXAK7.1B/2021_02_05_Discrimination_Imaging",
"NXAK7.1B/2021_02_07_Discrimination_Imaging",
"NXAK7.1B/2021_02_09_Discrimination_Imaging",
"NXAK7.1B/2021_02_24_Discrimination_Imaging",
"NXAK14.1A/2021_04_29_Discrimination_Imaging",
"NXAK14.1A/2021_05_01_Discrimination_Imaging",
"NXAK14.1A/2021_05_03_Discrimination_Imaging",
"NXAK14.1A/2021_05_05_Discrimination_Imaging",
"NXAK14.1A/2021_05_07_Discrimination_Imaging",
"NXAK14.1A/2021_05_09_Discrimination_Imaging",
"NXAK22.1A/2021_09_25_Discrimination_Imaging",
"NXAK22.1A/2021_10_07_Discrimination_Imaging",
"NXAK22.1A/2021_10_08_Discrimination_Imaging",
]

data_root_diretory = r"/media/matthew/Expansion/Control_Data"



Neurexin_Learning_Tuples = [
"NRXN71.2A/2020_11_14_Discrimination_Imaging",
"NRXN71.2A/2020_11_16_Discrimination_Imaging",
"NRXN71.2A/2020_11_17_Discrimination_Imaging",
"NRXN71.2A/2020_11_19_Discrimination_Imaging",
"NRXN71.2A/2020_11_21_Discrimination_Imaging",
"NRXN71.2A/2020_11_23_Discrimination_Imaging",
"NRXN71.2A/2020_11_25_Discrimination_Imaging",
"NRXN71.2A/2020_11_27_Discrimination_Imaging",
"NRXN71.2A/2020_11_29_Discrimination_Imaging",
"NRXN71.2A/2020_12_01_Discrimination_Imaging",
"NRXN71.2A/2020_12_03_Discrimination_Imaging",
"NRXN71.2A/2020_12_05_Discrimination_Imaging"
"NXAK4.1A/2021_02_02_Discrimination_Imaging",
"NXAK4.1A/2021_02_04_Discrimination_Imaging",
"NXAK4.1A/2021_02_06_Discrimination_Imaging",
"NXAK4.1A/2021_03_03_Discrimination_Imaging",
"NXAK4.1A/2021_03_05_Discrimination_Imaging",
"NXAK10.1A/2021_04_30_Discrimination_Imaging",
"NXAK10.1A/2021_05_02_Discrimination_Imaging",
"NXAK10.1A/2021_05_12_Discrimination_Imaging",
"NXAK10.1A/2021_05_14_Discrimination_Imaging",
"NXAK16.1B/2021_05_02_Discrimination_Imaging",
"NXAK16.1B/2021_05_06_Discrimination_Imaging",
"NXAK16.1B/2021_05_08_Discrimination_Imaging",
"NXAK16.1B/2021_05_10_Discrimination_Imaging",
"NXAK16.1B/2021_05_12_Discrimination_Imaging",
"NXAK16.1B/2021_05_14_Discrimination_Imaging",
"NXAK16.1B/2021_05_16_Discrimination_Imaging",
"NXAK16.1B/2021_05_18_Discrimination_Imaging",
"NXAK16.1B/2021_06_15_Discrimination_Imaging",
"NXAK20.1B/2021_09_28_Discrimination_Imaging",
"NXAK20.1B/2021_09_30_Discrimination_Imaging",
"NXAK20.1B/2021_10_02_Discrimination_Imaging",
"NXAK20.1B/2021_10_11_Discrimination_Imaging",
"NXAK20.1B/2021_10_13_Discrimination_Imaging",
"NXAK20.1B/2021_10_15_Discrimination_Imaging",
"NXAK20.1B/2021_10_17_Discrimination_Imaging",
"NXAK20.1B/2021_10_19_Discrimination_Imaging",
"NXAK24.1C/2021_09_20_Discrimination_Imaging",
"NXAK24.1C/2021_09_22_Discrimination_Imaging",
"NXAK24.1C/2021_09_24_Discrimination_Imaging",
"NXAK24.1C/2021_09_26_Discrimination_Imaging",
"NXAK24.1C/2021_10_02_Discrimination_Imaging",
"NXAK24.1C/2021_10_04_Discrimination_Imaging",
"NXAK24.1C/2021_10_06_Discrimination_Imaging",
]

session_list = Neurexin_Learning_Tuples
data_root_diretory = "/media/matthew/External_Harddrive_1/Neurexin_Data"

for session in session_list:
    base_directory = os.path.join(data_root_diretory, session)
    print(base_directory)
    get_lick_aligned_hits(base_directory)
    visualise_traces(base_directory)
