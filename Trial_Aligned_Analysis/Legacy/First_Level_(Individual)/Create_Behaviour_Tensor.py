import numpy as np
import sys
import os
import Single_Session_Analysis_Utils

def get_trial_data(ai_data, stimuli_dictionary, selected_trace, onsets_time, start_window_time, stop_window_time):

    selected_trace_tensor = []
    number_of_trials = len(onsets_time)

    # Get Ai Data Trace
    selected_trace = ai_data[stimuli_dictionary[selected_trace]]
    numbe_of_timepoints = len(selected_trace)

    for trial_index in range(number_of_trials):

        trial_onset = onsets_time[trial_index]
        trial_start = trial_onset + start_window_time
        trial_stop = trial_onset + stop_window_time

        if trial_start >= 0 and trial_stop < numbe_of_timepoints:

            trial_data = selected_trace[trial_start:trial_stop]

            selected_trace_tensor.append(trial_data)

    selected_trace_tensor = np.array(selected_trace_tensor)
    return selected_trace_tensor


def create_behaviour_tensor(base_directory, onsets_file, start_window, stop_window, selected_traces):

    # Load Downsampled AI
    ai_data = np.load(os.path.join(base_directory, "Downsampled_AI_Matrix_Framewise.npy"), allow_pickle=True)

    # Create Stimuli Dictionary
    stimuli_dictionary = Single_Session_Analysis_Utils.create_stimuli_dictionary()

    # Load Onsets
    stimuli_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", onsets_file), allow_pickle=True)

    # Get Create Behaviour Tensor Dict
    behaviour_tensor_dict = {}
    for trace_name in selected_traces:
        selected_trace_tensor = get_trial_data(ai_data, stimuli_dictionary, trace_name, stimuli_onsets, start_window, stop_window)
        behaviour_tensor_dict[trace_name] = selected_trace_tensor

    return behaviour_tensor_dict

