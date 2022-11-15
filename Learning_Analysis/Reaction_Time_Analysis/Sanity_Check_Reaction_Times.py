import matplotlib.pyplot as plt
import numpy as np
import os

import Reaction_Utils

def visualise_reaction_times(base_directory):

    # Load AI Matrix
    ai_data = Reaction_Utils.load_ai_recorder_file(base_directory)

    # Create Stimuli Dictionary
    ai_trace_dict = Reaction_Utils.create_stimuli_dictionary()

    # Extract Stimuli Lick and Reward Traces
    vis_1_trace = ai_data[ai_trace_dict["Visual 1"]]
    lick_trace = ai_data[ai_trace_dict["Lick"]]
    reward_trace = ai_data[ai_trace_dict["Reward"]]
    trial_end_trace = ai_data[ai_trace_dict["Trial End"]]

    # Load Lick Threshold
    lick_threshold = np.load(os.path.join(base_directory, "Lick_Threshold.npy"))
    print("Lick Threshold", lick_threshold)

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

    # Iterate Through Trial
    for trial in behaviour_matrix:

        # Get Trial Behavioural Characteristics
        trial_type = trial[1]
        correct = trial[3]
        reaction_time = trial[23]
        trial_end = trial[15]
        stimuli_onset = trial[11]

        if trial_type == 1 and correct == 1:
            print("Reaction Time", reaction_time)

            start_window = 500

            plt.plot(vis_1_trace[stimuli_onset-start_window:trial_end], c='b')
            plt.plot(lick_trace[stimuli_onset-start_window:trial_end], c='orange')
            plt.plot(reward_trace[stimuli_onset-start_window:trial_end], c='gold')
            plt.plot(trial_end_trace[stimuli_onset-start_window:trial_end], c='cyan')
            plt.scatter([reaction_time + start_window], [1], c='k')
            plt.show()


session_list = [r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_14_Discrimination_Imaging",
                r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_15_Discrimination_Imaging"]


visualise_reaction_times(session_list[0])