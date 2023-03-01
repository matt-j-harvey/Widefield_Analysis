import numpy as np
import os
from Widefield_Utils import widefield_utils

"""

A Special Case Of Getting A Trial Tensor

This Will Return A Tensor - All Be The Length Of The Longest Trial, But Padded With NaNs After They Have Terminated

"""


def get_ragged_tensor(data_matrix, start_stop_tuples):

    # Create Empty Tensor To Hold Data
    data_tensor = []

    # Get Correlation Matrix For Each Trial
    number_of_trials = len(start_stop_tuples)
    for trial_index in range(number_of_trials):

        # Get Trial Activity
        trial_start = start_stop_tuples[trial_index][0]
        trial_stop = start_stop_tuples[trial_index][1]


        trial_activity = data_matrix[trial_start:trial_stop]
        trial_activity = np.nan_to_num(trial_activity)
        data_tensor.append(trial_activity)


    return data_tensor


def create_combined_stimulus_trace(behaviour_matrix, trace_name_list):

    # Create AI Channel Dict
    ai_channel_dict = widefield_utils.create_stimuli_dictionary()

    if len(trace_name_list) > 1:
        trace_list = []
        for trace_name in trace_name_list:
            trace = behaviour_matrix[ai_channel_dict[trace_name]]
            trace_list.append(trace)

        combined_stimuli_trace = np.vstack(trace_list)
        combined_stimuli_trace = np.max(combined_stimuli_trace, axis=0)
    else:
        combined_stimuli_trace = behaviour_matrix[ai_channel_dict[trace_name]]

    return combined_stimuli_trace

def get_trial_stop(trial_start, stop_trace, trace_threshold, minimum_length, max_length=140):

    number_of_timepoints = len(stop_trace)
    trial_ongoing = True
    trial_length = 0
    while trial_ongoing:

        if trial_start + trial_length >= number_of_timepoints:
            trial_ongoing = False
            return None
        else:

            if trial_length > max_length:
                trial_ongoing = False
                return trial_start + trial_length

            elif trial_length > minimum_length and stop_trace[trial_start + trial_length] > trace_threshold:
                trial_ongoing = False
                return trial_start + trial_length

            else:
                trial_length += 1


def get_trial_stats_and_stops(onsets_list, number_of_timepoints, start_window, stop_trace, minimum_length, trace_threshold=0.5, start_cutoff=3000):

    number_of_trials = np.shape(onsets_list)[0]
    trial_start_stop_tuple_list = []

    for trial_index in range(number_of_trials):

        trial_start = onsets_list[trial_index] + start_window
        trial_stop = get_trial_stop(onsets_list[trial_index], stop_trace, trace_threshold, minimum_length)

        if trial_stop != None:
            if trial_start > start_cutoff and trial_stop < number_of_timepoints:
                trial_start_stop_tuple_list.append([trial_start, trial_stop])

                print("Trial: ", trial_index, "Start ", trial_start, "Stop ", trial_stop, "Length: ", trial_stop - trial_start)

    return trial_start_stop_tuple_list



def get_extended_tensor(base_directory, activity_matrix, onsets_list, start_window, stop_stimuli):

    # Load Downsampled Behaviour Matrix
    downsampled_ai_matrix = np.load(os.path.join(base_directory, "Downsampled_AI_Matrix_Framewise.npy"))

    # Create Combined Stimulus Trace
    combined_stimulus_trace = create_combined_stimulus_trace(downsampled_ai_matrix, stop_stimuli)

    # Get Trial Stars and Stops
    number_of_timepoints = np.shape(activity_matrix)[0]
    start_stop_tuple_list = get_trial_stats_and_stops(onsets_list, number_of_timepoints, start_window, combined_stimulus_trace, minimum_length=20)

    # Get Ragged Tensor
    extended_tensor = get_ragged_tensor(activity_matrix, start_stop_tuple_list)

    return extended_tensor
