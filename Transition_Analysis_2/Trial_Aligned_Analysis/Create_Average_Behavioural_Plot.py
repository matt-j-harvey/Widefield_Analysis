import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec
import tables

import Trial_Aligned_Utils


def scatter_onsets(base_directory, onset_file):

    # Load Frame Times
    frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    frame_times = Trial_Aligned_Utils.invert_dictionary(frame_times)
    print("Frame Times Keys", frame_times.keys())

    # Load Onsets
    onsets_list = np.load(os.path.join(base_directory, "Stimuli_Onsets", onset_file))

    stimuli_dictionary = Trial_Aligned_Utils.create_stimuli_dictionary()

    # Load AI Matrix
    ai_matrix = Trial_Aligned_Utils.load_ai_data(base_directory)

    vis_1_trace = ai_matrix[stimuli_dictionary["Visual 1"]]
    vis_2_trace = ai_matrix[stimuli_dictionary["Visual 2"]]
    odour_1_trace = ai_matrix[stimuli_dictionary["Odour 1"]]
    odour_2_trace = ai_matrix[stimuli_dictionary["Odour 2"]]

    for onset in onsets_list:
        if onset > 3000:
            onset_time = frame_times[onset]
            plt.scatter([onset_time], [3])

    plt.plot(vis_1_trace, c='b')
    plt.plot(vis_2_trace, c='r')
    plt.plot(odour_1_trace, c='g')
    plt.plot(odour_2_trace, c='m')
    plt.show()


def get_trace_tensor(ai_trace, onsets_list, start_window, stop_window):

    trace_tensor = []
    for onset in onsets_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window
        trial_data = ai_trace[trial_start:trial_stop]
        trace_tensor.append(trial_data)

    trace_tensor = np.array(trace_tensor)
    trace_mean = np.mean(trace_tensor, axis=0)
    trace_std = np.std(trace_tensor, axis=0)
    return trace_mean, trace_std


def view_condition_behaviour(base_directory, start_window, stop_window, onset_file_list, tensor_names, selected_behavioural_traces, difference_conditions):

    number_of_conditions = len(tensor_names)

    figure_1 = plt.figure()
    girdspec_1 = GridSpec(figure=figure_1, nrows=1, ncols=number_of_conditions)

    # Create Stimuli Dictionary
    stimuli_dictionary = Trial_Aligned_Utils.create_stimuli_dictionary()

    # Load AI Matrix
    ai_matrix = np.load(os.path.join(base_directory, "Downsampled_AI_Matrix_Framewise.npy"))

    trace_colours = {
        'Running':'k',
        'Visual 1':"b",
        'Visual 2':'r',
        'Odour 1':'g',
        'Odour 2':'m',
        'Lick':'y'}

    number_of_traces = len(selected_behavioural_traces)
    for condition_index in range(number_of_conditions):

        # Get Condition Name
        condition_name = tensor_names[condition_index]

        # Load Condition Onsets
        condition_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", onset_file_list[condition_index]))

        # Create Axis
        axis = figure_1.add_subplot(girdspec_1[0, condition_index])

        for trace_index in range(number_of_traces):
            trace_name = selected_behavioural_traces[trace_index]
            trace = ai_matrix[stimuli_dictionary[trace_name]]
            trace_mean, trace_std = get_trace_tensor(trace, condition_onsets, start_window, stop_window)
            trace_colour = trace_colours[trace_name]

            axis.plot(trace_mean, c=trace_colour)

        axis.set_title(condition_name)

    plt.show()


# Get Analysis Details
analysis_name = "Absence Of Expected Odour"
#analysis_name = "Vis_1_v_Vis_2"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Trial_Aligned_Utils.load_analysis_container(analysis_name)

session_list = [
    r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_23_Transition_Imaging",
    r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_31_Transition_Imaging",
    r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_04_02_Transition_Imaging",
]

onset_file = r"visual_context_stable_vis_1_onsets.npy"
for base_directory in session_list:
    #stop_window = 150
    view_condition_behaviour(base_directory, start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions)
    #scatter_onsets(base_directory, onset_file)