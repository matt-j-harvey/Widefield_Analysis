import os
import matplotlib.pyplot as plt
import numpy as np
import tables
import random

def ResampleLinear1D(original, targetLen):
    original = np.array(original, dtype=float)
    index_arr = np.linspace(0, len(original) - 1, num=targetLen, dtype=float)
    index_floor = np.array(index_arr, dtype=int)  # Round down
    index_ceil = index_floor + 1
    index_rem = index_arr - index_floor  # Remain
    val1 = original[index_floor]
    val2 = original[index_ceil % len(original)]
    interp = val1 * (1.0 - index_rem) + val2 * index_rem
    assert (len(interp) == targetLen)
    return interp


def invert_dictionary(dictionary):
    inv_map = {v: k for k, v in dictionary.items()}
    return inv_map


def get_ai_filename(base_directory):

    ai_filename = None

    # Get List of all files
    file_list = os.listdir(base_directory)

    # Get .h5 files
    h5_file_list = []
    for file in file_list:
        if file[-3:] == ".h5":
            h5_file_list.append(file)

    # File the H5 file which is two dates seperated by a dash
    for h5_file in h5_file_list:

        original_filename = h5_file

        # Remove Ending
        h5_file = h5_file[0:-3]

        # Split By Dashes
        h5_file = h5_file.split("-")

        if len(h5_file) == 2 and h5_file[0].isnumeric() and h5_file[1].isnumeric():
            return original_filename



def load_ai_recorder_file(ai_recorder_file_location):

    table = tables.open_file(ai_recorder_file_location, mode='r')
    data = table.root.Data
    number_of_seconds = np.shape(data)[0]
    number_of_channels = np.shape(data)[1]
    sampling_rate = np.shape(data)[2]
    data_matrix = np.zeros((number_of_channels, number_of_seconds * sampling_rate))

    for second in range(number_of_seconds):
        data_window = data[second]
        start_point = second * sampling_rate
        for channel in range(number_of_channels):
            data_matrix[channel, start_point:start_point + sampling_rate] = data_window[channel]

    data_matrix = np.clip(data_matrix, a_min=0, a_max=None)

    return data_matrix


def downsample_ai_recorder_data(base_directory):

    # Load Frame Times
    frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    frame_times = invert_dictionary(frame_times)
    print("Number of camera pulses: ", len(frame_times.keys()))

    # Load AI Recorder File
    ai_filename = get_ai_filename(base_directory)
    ai_data = load_ai_recorder_file(os.path.join(base_directory, ai_filename))
    number_of_traces = np.shape(ai_data)[0]
    print("AI Data Shape", np.shape(ai_data))

    # Load Delta F Matrix
    delta_f_matrix = np.load(os.path.join(base_directory, "Cluster_Activity_Matrix.npy"))
    print("Delta F Matrix Shape", np.shape(delta_f_matrix))

    # Get Data Structure
    number_of_timepoints, number_of_regions = np.shape(delta_f_matrix)
    imaging_start = frame_times[0]
    imaging_stop = frame_times[number_of_timepoints - 1]
    print("Imaging Start", imaging_start)
    print("Imaging Stop", imaging_stop)

    # Get Traces Only While Imaging
    ai_data = ai_data[:, imaging_start:imaging_stop]

    downsampled_ai_data = []
    for trace_index in range(number_of_traces):
        trace = ai_data[trace_index]
        downsampled_trace = ResampleLinear1D(trace, number_of_timepoints)
        downsampled_ai_data.append(downsampled_trace)

    downsampled_ai_data = np.array(downsampled_ai_data)
    print("Downsampled Data", np.shape(downsampled_ai_data))

    # Save Trace
    np.save(os.path.join(base_directory, "Downsample_AI_Data.npy"), downsampled_ai_data)



session_list = [
     "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_01_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_03_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_05_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_07_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_09_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_11_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_13_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_15_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_17_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_19_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_22_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_24_Discrimination_Imaging"]

for session in session_list:
    downsample_ai_recorder_data(session)
