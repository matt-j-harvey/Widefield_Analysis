import numpy as np
import os
import matplotlib.pyplot as plt

import Split_AI_Channels_By_Context
from Widefield_Utils import widefield_utils



def split_stream_by_context(stimuli_onsets, context_onsets, context_window):
    context_negative_onsets = []
    context_positive_onsets = []

    # Iterate Through Visual 1 Onsets
    for stimuli_onset in stimuli_onsets:
        context = False
        window_start = stimuli_onset
        window_end = stimuli_onset + context_window

        for context_onset in context_onsets:
            if context_onset >= window_start and context_onset <= window_end:
                context = True

        if context == True:
            context_positive_onsets.append(stimuli_onset)
        else:
            context_negative_onsets.append(stimuli_onset)

    return context_negative_onsets, context_positive_onsets




def get_length_of_longest_stim(binary_list):
    max_size = 0
    prev = None

    for i in binary_list:

        if i == 1 and i == prev:
            size += 1
            if size > max_size:
                max_size = size
        else:
            size = 0

        prev = i

    return result


def create_regressor_matrix(number_of_timepoints, longest_stimuli, stimuli_trace):

    regressor_matrix = np.zeros((number_of_timepoints, longest_stimuli))

    current_depth = 0
    for timepoint_index in range(number_of_timepoints):

        if stimuli_trace[timepoint_index] == 1:
            regressor_matrix[timepoint_index, current_depth] = 1
            current_depth += 1

        else:
            current_depth = 0

    return regressor_matrix



def create_stimuli_regressor(regressor_trace, threshold=0.3):

    # Binarise
    regressor_trace = np.where(regressor_trace > threshold, 1, 0)

    # Get Longest Continuous Stimuli
    longest_stimuli = get_length_of_longest_stim(regressor_trace)

    # Populate Full Regressor
    number_of_timepoints = len(regressor_trace)
    regressor_matrix = create_regressor_matrix(number_of_timepoints, longest_stimuli, stimuli_trace)

    return regressor_matrix


def view_irrel_channel(downsampled_ai_matrix, stimuli_dictionary):

    irrel_trace = downsampled_ai_matrix[stimuli_dictionary["Irrelevance"]]

    odour_1_trace = downsampled_ai_matrix[stimuli_dictionary["Odour 1"]]
    odour_2_trace = downsampled_ai_matrix[stimuli_dictionary["Odour 2"]]
    combined_odour_trace = np.add(odour_2_trace, odour_1_trace)

    vis_1_trace = downsampled_ai_matrix[stimuli_dictionary["Visual 1"]]
    vis_2_trace = downsampled_ai_matrix[stimuli_dictionary["Visual 2"]]
    combined_vis_trace = np.add(vis_1_trace, vis_2_trace)

    plt.plot(combined_odour_trace, c='g')
    plt.plot(combined_vis_trace, c='b', alpha=0.5)
    plt.plot(irrel_trace, c='k', alpha=0.5)
    plt.show()


def create_stimuli_regressors(downsampled_ai_matrix):

    # Create Stimuli Dict
    stimuli_dictionary = widefield_utils.create_stimuli_dictionary()

    # Extract Traces
    trace_list = []
    vis_context_vis_1_trace, vis_context_vis_2_trace, odour_context_vis_1_trace, odour_context_vis_2_trace = Split_AI_Channels_By_Context.split_ai_channels_by_context(downsampled_ai_matrix)
    trace_list.append(vis_context_vis_1_trace)
    trace_list.append(vis_context_vis_2_trace)
    trace_list.append(odour_context_vis_1_trace)
    trace_list.append(odour_context_vis_2_trace)
    trace_list.append(downsampled_ai_matrix["Odour_1"])
    trace_list.append(downsampled_ai_matrix["Odour_2"])

    # Create Regressors For Each Stimuli
    stimuli_regressor_list = []
    for stimulus in selected_stimuli:
        stimuli_trace = downsampled_ai_matrix[stimuli_dictionary[stimulus]]
        stimuli_regressor = create_stimuli_regressor(stimuli_trace)
        stimuli_regressor_list.append(stimuli_regressor)

    return stimuli_regressor_list
    """


selected_session_list = [

    [r"NRXN78.1A/2020_12_05_Switching_Imaging",
    r"NRXN78.1A/2020_12_09_Switching_Imaging"],

    [r"NRXN78.1D/2020_11_29_Switching_Imaging",
    r"NRXN78.1D/2020_12_05_Switching_Imaging"],

    [r"NXAK14.1A/2021_05_21_Switching_Imaging",
    r"NXAK14.1A/2021_05_23_Switching_Imaging",
    r"NXAK14.1A/2021_06_11_Switching_Imaging",
    r"NXAK14.1A/2021_06_13_Transition_Imaging",
    r"NXAK14.1A/2021_06_15_Transition_Imaging",
    r"NXAK14.1A/2021_06_17_Transition_Imaging"],

    [r"NXAK22.1A/2021_10_14_Switching_Imaging",
    r"NXAK22.1A/2021_10_20_Switching_Imaging",
    r"NXAK22.1A/2021_10_22_Switching_Imaging",
    r"NXAK22.1A/2021_10_29_Transition_Imaging",
    r"NXAK22.1A/2021_11_03_Transition_Imaging",
    r"NXAK22.1A/2021_11_05_Transition_Imaging"],

    [r"NXAK4.1B/2021_03_02_Switching_Imaging",
    r"NXAK4.1B/2021_03_04_Switching_Imaging",
    r"NXAK4.1B/2021_03_06_Switching_Imaging",
    r"NXAK4.1B/2021_04_02_Transition_Imaging",
    r"NXAK4.1B/2021_04_08_Transition_Imaging",
    r"NXAK4.1B/2021_04_10_Transition_Imaging"],

    [r"NXAK7.1B/2021_02_26_Switching_Imaging",
    r"NXAK7.1B/2021_02_28_Switching_Imaging",
    r"NXAK7.1B/2021_03_02_Switching_Imaging",
    r"NXAK7.1B/2021_03_23_Transition_Imaging",
    r"NXAK7.1B/2021_03_31_Transition_Imaging",
    r"NXAK7.1B/2021_04_02_Transition_Imaging"
    ],

]

for mouse in selected_session_list:
    for session in mouse:
        downsampled_ai_matrix = np.load(os.path.join("/media/matthew/Expansion/Control_Data", session, "Downsampled_AI_Matrix_Framewise.npy"))
        create_stimuli_regressors(downsampled_ai_matrix)