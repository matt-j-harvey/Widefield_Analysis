import numpy as np

import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
import tables
import random
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist, cdist

from sklearn.linear_model import LinearRegression

def invert_dictionary(dictionary):
    inv_map = {v: k for k, v in dictionary.items()}
    return inv_map

def get_ai_filename(base_directory):
    # Get List of all files
    file_list = os.listdir(base_directory)
    ai_filename = None

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



def create_stimuli_dictionary():
    channel_index_dictionary = {
        "Photodiode": 0,
        "Reward": 1,
        "Lick": 2,
        "Visual 1": 3,
        "Visual 2": 4,
        "Odour 1": 5,
        "Odour 2": 6,
        "Irrelevance": 7,
        "Running": 8,
        "Trial End": 9,
        "Camera Trigger": 10,
        "Camera Frames": 11,
        "LED 1": 12,
        "LED 2": 13,
        "Mousecam": 14,
        "Optogenetics": 15,
    }

    return channel_index_dictionary


def forceAspect(ax, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)



def normalise_trace(trace):
    trace = np.subtract(trace, np.min(trace))
    trace = np.divide(trace, np.max(trace))
    return trace



def downsammple_trace_framewise(trace, frame_times):

    # Get Average Frame Duration
    frame_duration_list = []
    for frame_index in range(1000):
        frame_duaration = frame_times[frame_index + 1] - frame_times[frame_index]
        frame_duration_list.append(frame_duaration)
    average_duration = int(np.mean(frame_duration_list))

    downsampled_trace = []
    number_of_frames = len(frame_times.keys())
    for frame_index in range(number_of_frames-1):
        frame_start = frame_times[frame_index]
        frame_end = frame_times[frame_index + 1]
        frame_data = trace[frame_start:frame_end]
        frame_data_mean = np.mean(frame_data)
        downsampled_trace.append(frame_data_mean)

    # Add Final Frame
    final_frame_start = frame_times[number_of_frames-1]
    final_frame_end = final_frame_start + average_duration
    final_frame_data = trace[final_frame_start:final_frame_end]
    final_frame_mean = np.mean(final_frame_data)
    downsampled_trace.append(final_frame_mean)

    return downsampled_trace



def visualise_downsampling(original_trace, downsampled_trace):

    figure_1 = plt.figure()
    rows = 2
    columns = 1
    original_axis = figure_1.add_subplot(rows, columns, 1)
    downsample_axis = figure_1.add_subplot(rows, columns, 2)

    original_axis.plot(original_trace)
    downsample_axis.plot(downsampled_trace)

    plt.show()



def downsample_ai_matrix(base_directory):

    # Load Frame Times
    frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    frame_times = invert_dictionary(frame_times)
    print("Number of Frames", len(frame_times.keys()))

    # Load AI Recorder File
    ai_filename = get_ai_filename(base_directory)
    ai_data = load_ai_recorder_file(os.path.join(base_directory, ai_filename))

    # Extract Relevant Traces
    number_of_traces = np.shape(ai_data)[0]
    print("Number of traces", number_of_traces)

    # Create Downsampled AI Matrix
    downsampled_ai_matrix = []
    for trace_index in range(number_of_traces):
        full_trace = ai_data[trace_index]
        downsampled_trace = downsammple_trace_framewise(full_trace, frame_times)
        normalised_trace = normalise_trace(downsampled_trace)
        downsampled_ai_matrix.append(normalised_trace)
    downsampled_ai_matrix = np.array(downsampled_ai_matrix)
    print("DOwnsampled AI Matrix Shape", np.shape(downsampled_ai_matrix))

    # Save Downsampled AI Matrix
    np.save(os.path.join(base_directory, "Downsampled_AI_Matrix.npy"), downsampled_ai_matrix)





session_list = [
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging",
]


number_of_sessions = len(session_list)
for session in session_list:
    downsample_ai_matrix(session)

