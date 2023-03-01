import os
import numpy as np
import matplotlib.pyplot as plt

from Widefield_Utils import widefield_utils

def check_visual_stimuli_onsets(base_directory, stimuli_onsets_file, preceeding_window=-200, following_window=200):

    # Load Stimuli Onsets
    stimuli_onset_frames = np.load(os.path.join(base_directory, "Stimuli_Onsets", stimuli_onsets_file))

    # Load Widefield Frame Dict
    widefield_frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    widefield_frame_times = widefield_utils.invert_dictionary(widefield_frame_times)

    # Load AI Matrix
    ai_data = widefield_utils.load_ai_recorder_file(base_directory)

    # Create Stimuli Dict
    stimuli_dictionary = widefield_utils.create_stimuli_dictionary()
    vis_1_trace = ai_data[stimuli_dictionary["Visual 1"]]
    vis_2_trace = ai_data[stimuli_dictionary["Visual 2"]]

    number_of_trials = len(stimuli_onset_frames)

    figure_1 = plt.figure(figsize=(10, number_of_trials * 0.9))
    columns = 4
    rows = np.ceil(float(number_of_trials)/4)

    for trial_index in range(number_of_trials):

        # Create Axis
        axis = figure_1.add_subplot(rows, columns, trial_index + 1)

        # Get Trial Start
        stimuli_frame = stimuli_onset_frames[trial_index]
        stimuli_time = widefield_frame_times[stimuli_frame]
        trial_start = stimuli_time + preceeding_window
        trial_stop = stimuli_time + following_window

        # Plot Vis Traces
        x_values = list(range(preceeding_window, following_window))
        axis.plot(x_values, vis_1_trace[trial_start:trial_stop], c='b')
        axis.plot(x_values, vis_2_trace[trial_start:trial_stop], c='r')
        axis.axvline(x=0, color='k', linestyle='dashed')
        axis.set_title(str(trial_index))


    stim_name = stimuli_onsets_file.replace(".npy","")
    save_directory = os.path.join(base_directory, "Stimuli_Onsets", str(stim_name) + ".png")
    plt.savefig(save_directory)
    plt.close()

    print("AI Data", np.shape(ai_data))





session_list = [

        r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_28_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_05_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_09_Switching_Imaging",

        r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging",
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
        #r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_02_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_23_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_31_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_04_02_Transition_Imaging"
]


vis_context_vis_1_onset_file = "visual_context_stable_vis_1_onsets.npy"
vis_context_vis_2_onset_file = "visual_context_stable_vis_2_onsets.npy"

odour_context_vis_1_onset_file = "odour_context_stable_vis_1_onsets.npy"
odour_context_vis_2_onset_file = "odour_context_stable_vis_2_onsets.npy"

for session in session_list:

    base_directory = session #os.path.join(r"/media/matthew/Expansion/Control_Data", session)

    check_visual_stimuli_onsets(base_directory, vis_context_vis_1_onset_file)
    check_visual_stimuli_onsets(base_directory, vis_context_vis_2_onset_file)
    check_visual_stimuli_onsets(base_directory, odour_context_vis_1_onset_file)
    check_visual_stimuli_onsets(base_directory, odour_context_vis_2_onset_file)


