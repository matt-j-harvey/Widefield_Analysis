import numpy as np
import sys
import os

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions



def convert_frame_indexes_to_onsets(frame_indexes, frame_times):

    frame_times_list = []

    for index in frame_indexes:
        frame_times_list.append(frame_times[index])

    return frame_times_list


def get_widow_size(frame_index_list, window, frame_times):

    # Adjust_Onsets
    adjusted_index_list = []
    for frame_index in frame_index_list:
        adjusted_index_list.append(frame_index + window)

    # Get Frame Times Of Onsets
    stimuli_onsets = convert_frame_indexes_to_onsets(frame_index_list, frame_times)
    adjusted_onsets = convert_frame_indexes_to_onsets(adjusted_index_list, frame_times)

    # Get Largest Window
    number_of_trials = len(frame_index_list)
    window_size_list = []

    for trial_index in range(number_of_trials):
        onset = stimuli_onsets[trial_index]
        adjusted_onset = adjusted_onsets[trial_index]

        window_size = np.abs(onset - adjusted_onset)
        window_size_list.append(window_size)

    mean_window_size = np.mean(window_size_list)
    mean_window_size = np.int(mean_window_size)

    return mean_window_size




def get_trial_data(ai_data, stimuli_dictionary, selected_trace, onsets_time, start_window_time, stop_window_time):

    selected_trace_tensor = []
    number_of_trials = len(onsets_time)

    # Get Ai Data Trace
    selected_trace = ai_data[stimuli_dictionary[selected_trace]]

    for trial_index in range(number_of_trials):

        trial_onset = onsets_time[trial_index]
        trial_start = trial_onset + start_window_time
        trial_stop = trial_onset + stop_window_time

        trial_data = selected_trace[trial_start:trial_stop]

        selected_trace_tensor.append(trial_data)

    selected_trace_tensor = np.array(selected_trace_tensor)
    return selected_trace_tensor





def create_behaviour_tensor(base_directory, onsets_file, start_window, stop_window, selected_traces, timestep=36):

    # Load AI Data
    ai_filename = Widefield_General_Functions.get_ai_filename(base_directory)
    full_ai_filepath = base_directory + "/" + ai_filename
    ai_data = Widefield_General_Functions.load_ai_recorder_file(full_ai_filepath)

    # Create Stimuli Dictionary
    stimuli_dictionary = Widefield_General_Functions.create_stimuli_dictionary()

    # Load Stimuli Onsets
    stimuli_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", onsets_file))

    # Load Frame Times
    frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    frame_times = Widefield_General_Functions.invert_dictionary(frame_times)

    # Get Largest Window
    #start_window_time = get_widow_size(stimuli_onsets, start_window, frame_times)
    #stop_window_time = get_widow_size(stimuli_onsets, stop_window, frame_times)

    start_window_time = start_window * timestep
    stop_window_time = stop_window * timestep

    # Convert Onsets To Times
    stimuli_onsets_time = convert_frame_indexes_to_onsets(stimuli_onsets, frame_times)

    # Get Trensors
    behaviour_tensor_dict = {}
    for trace_name in selected_traces:
        selected_trace_tensor = get_trial_data(ai_data, stimuli_dictionary, trace_name, stimuli_onsets_time, start_window_time, stop_window_time)
        behaviour_tensor_dict[trace_name] = selected_trace_tensor

    return behaviour_tensor_dict