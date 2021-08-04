import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
import os
import math
import scipy
import tables


def get_ai_filename(base_directory):

    #Get List of all files
    file_list = os.listdir(base_directory)
    ai_filename = None

    #Get .h5 files
    h5_file_list = []
    for file in file_list:
        if file[-3:] == ".h5":
            h5_file_list.append(file)

    #File the H5 file which is two dates seperated by a dash
    for h5_file in h5_file_list:
        original_filename = h5_file

        #Remove Ending
        h5_file = h5_file[0:-3]

        #Split By Dashes
        h5_file = h5_file.split("-")

        if len(h5_file) == 2 and h5_file[0].isnumeric() and h5_file[1].isnumeric():
            ai_filename = "/" + original_filename
            print("Ai filename is: ", ai_filename)
            return ai_filename




def load_ai_recorder_file(ai_recorder_file_location):
    table = tables.open_file(ai_recorder_file_location, mode='r')
    data = table.root.Data

    number_of_seconds = np.shape(data)[0]
    number_of_channels = np.shape(data)[1]
    sampling_rate = np.shape(data)[2]

    print("Number of seconds", number_of_seconds)

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
        "Photodiode"        :0,
        "Reward"            :1,
        "Lick"              :2,
        "Visual 1"          :3,
        "Visual 2"          :4,
        "Odour 1"           :5,
        "Odour 2"           :6,
        "Irrelevance"       :7,
        "Running"           :8,
        "Trial End"         :9,
        "Camera Trigger"    :10,
        "Camera Frames"     :11,
        "LED 1"             :12,
        "LED 2"             :13,
        "Mousecam"          :14,
        "Optogenetics"      :15,
        }

    return channel_index_dictionary


def invert_dictionary(dictionary):
    inv_map = {v: k for k, v in dictionary.items()}
    return inv_map



def ResampleLinear1D(original, targetLen):
    original = np.array(original, dtype=np.float)
    index_arr = np.linspace(0, len(original)-1, num=targetLen, dtype=np.float)
    index_floor = np.array(index_arr, dtype=np.int) #Round down
    index_ceil = index_floor + 1
    index_rem = index_arr - index_floor #Remain

    val1 = original[index_floor]
    val2 = original[index_ceil % len(original)]
    interp = val1 * (1.0-index_rem) + val2 * index_rem
    assert(len(interp) == targetLen)
    return interp


def get_ai_recorder_components(base_directory, onsets_list, start_window, stop_window):

    # Get Window Size
    number_of_trials = len(onsets_list)
    window_size = stop_window - start_window
    print("window size", window_size)

    # Load AI Data
    ai_file_location = get_ai_filename(base_directory)
    ai_data = load_ai_recorder_file(base_directory + ai_file_location)

    # Extract Traces
    stimuli_dict = create_stimuli_dictionary()
    running_trace = ai_data[stimuli_dict["Running"]]
    lick_trace = ai_data[stimuli_dict["Lick"]]

    # Load Frame Times
    time_frame_dict = np.load(base_directory + "/Stimuli_Onsets/Frame_Times.npy", allow_pickle=True)
    time_frame_dict = time_frame_dict[()]
    frame_time_dict = invert_dictionary(time_frame_dict)

    ai_data = np.zeros((number_of_trials, window_size, 2))
    for trial_index in range(number_of_trials):
        onset = onsets_list[trial_index]
        start_frame = onset + start_window
        stop_frame = onset + stop_window

        start_time = frame_time_dict[start_frame]
        stop_time = frame_time_dict[stop_frame]

        trial_lick_data = lick_trace[start_time:stop_time]
        trial_running_data = running_trace[start_time:stop_time]

        trial_lick_data = ResampleLinear1D(trial_lick_data,       window_size)
        trial_running_data = ResampleLinear1D(trial_running_data, window_size)

    ai_data[trial_index, :, 0] = trial_lick_data
    ai_data[trial_index, :, 1] = trial_running_data

    return ai_data


def get_offset(onset, stream, threshold=0.5):

    count = 0
    on = True
    while on:
        if stream[onset + count] < threshold:
            on = False
            return onset + count
        else:
            count += 1



def create_visual_stimuli_design_matrix(onsets, start_window, stop_window, base_directory):

    # Load AI Data
    ai_file_location = get_ai_filename(base_directory)
    ai_data = load_ai_recorder_file(base_directory + ai_file_location)

    for visual onset in
    visual_frame_onset =


base_directory = r"/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Switching_Analysis/Selected_sessions/NXAK16.1B/2021_06_23_Switching_Imaging"
condition = "visual_context_stable_vis_2"

start_window = -10
stop_window = 100

# Load Frame Onsets and Frame Times
frame_onsets_file = base_directory + "/Stimuli_Onsets/" + condition + "_frame_onsets.npy"
frame_onsets = np.load(frame_onsets_file)

# Load All Trials
ai_regressors = get_ai_recorder_components(base_directory, frame_onsets, start_window, stop_window)
print("Ai Regressor Matrix Shape", np.shape(ai_regressors))




# Create Design Matrix
# One Model For Each Context
# Running speed
# Lick Trace
# Behaviour Video SVDs



