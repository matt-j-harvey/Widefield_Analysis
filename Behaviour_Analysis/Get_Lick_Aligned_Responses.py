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
    onsets_list = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Lick_Aligned_Hits_atleast_500.npy"))

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
        reaction_time = lick_onset - stimuli_onset

        if trial_type == 1 and correct == 1 and reaction_time > 500:
            nearest_frame_onset = widefield_utils.take_closest(frame_times, lick_onset)
            nearest_frame = frame_time_dict[nearest_frame_onset]
            lick_aligned_hits.append(nearest_frame)

    np.save(os.path.join(base_directory,  "Stimuli_Onsets", "Lick_Aligned_Hits_atleast_500.npy"), lick_aligned_hits)


mutant_session_tuples = [
    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_14_Discrimination_Imaging",
     r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_09_Discrimination_Imaging"],

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_04_Discrimination_Imaging",
     r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging"],

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_02_Discrimination_Imaging",
     r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_14_Discrimination_Imaging"],

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_02_Discrimination_Imaging",
     r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_15_Discrimination_Imaging"],

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_30_Discrimination_Imaging",
     r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging"],

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_22_Discrimination_Imaging",
     r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_08_Discrimination_Imaging"],

]




data_root_diretory = r"/media/matthew/External_Harddrive_1/Neurexin_Data"

for mouse in mutant_session_tuples:
    for session in mouse:
        base_directory = os.path.join(data_root_diretory, session)
        get_lick_aligned_hits(base_directory)
        visualise_traces(base_directory)
