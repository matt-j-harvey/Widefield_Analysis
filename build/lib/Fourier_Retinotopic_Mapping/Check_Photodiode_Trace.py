import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
import os
import mat73
import scipy.io
import tables
from scipy import signal, ndimage, stats
from sklearn.neighbors import KernelDensity
import cv2
from matplotlib import gridspec


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


def get_step_onsets(trace, threshold=1, window=10):
    state = 0
    number_of_timepoints = len(trace)
    onset_times = []
    time_below_threshold = 0

    onset_line = []

    for timepoint in range(number_of_timepoints):
        if state == 0:
            if trace[timepoint] > threshold:
                state = 1
                onset_times.append(timepoint)
                time_below_threshold = 0
            else:
                pass
        elif state == 1:
            if trace[timepoint] > threshold:
                time_below_threshold = 0
            else:
                time_below_threshold += 1
                if time_below_threshold > window:
                    state = 0
                    time_below_threshold = 0
        onset_line.append(state)

    return onset_times, onset_line






def get_frame_indexes(frame_stream):

    frame_indexes = {}
    state = 1
    threshold = 2
    count = 0

    for timepoint in range(0, len(frame_stream)):

        if frame_stream[timepoint] > threshold:
            if state == 0:
                state = 1
                frame_indexes[timepoint] = count
                count += 1

        else:
            if state == 1:
                state = 0
            else:
                pass

    return frame_indexes


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


def split_visual_onsets_by_context(visual_1_onsets, visual_2_onsets, odour_1_onsets, odour_2_onsets, following_window_size=7000):

    combined_odour_onsets = odour_1_onsets + odour_2_onsets
    visual_block_stimuli_1, odour_block_stimuli_1 = split_stream_by_context(visual_1_onsets, combined_odour_onsets, following_window_size)
    visual_block_stimuli_2, odour_block_stimuli_2 = split_stream_by_context(visual_2_onsets, combined_odour_onsets, following_window_size)

    onsets_list = [visual_block_stimuli_1, visual_block_stimuli_2, odour_block_stimuli_1, odour_block_stimuli_2]

    return onsets_list



def get_nearest_frame(stimuli_onsets, frame_onsets):


    frame_times = frame_onsets.keys()
    nearest_frames = []
    window_size = 50

    for onset in stimuli_onsets:
        smallest_distance = 1000
        closest_frame = None

        window_start = onset - window_size
        window_stop  = onset + window_size

        for timepoint in range(window_start, window_stop):

            #There is a frame at this time
            if timepoint in frame_times:
                distance = abs(onset - timepoint)

                if distance < smallest_distance:
                    smallest_distance = distance
                    closest_frame = frame_onsets[timepoint]

        if closest_frame != None:
            if closest_frame > 11:
                nearest_frames.append(closest_frame)

    nearest_frames = np.array(nearest_frames)
    return nearest_frames



def get_visual_onsets_in_stable_odour_trials(visual_1_onsets, visual_2_onsets, stable_odour_1_onsets, stable_odour_2_onsets):

    following_window_size = 5000

    combined_stable_odour_onsets = stable_odour_1_onsets + stable_odour_2_onsets
    vis_1_onsets_in_stable_odour_trials = []
    vis_2_onsets_in_stable_odour_trials = []

    #Get Vis 1 onsets in stable odour trials
    for visual_onset in visual_1_onsets:
        following_window = visual_onset + following_window_size

        for odour_onset in combined_stable_odour_onsets:
            if odour_onset > visual_onset and odour_onset <= following_window:
                vis_1_onsets_in_stable_odour_trials.append(visual_onset)

    # Get Vis 2 onsets in stable odour trials
    for visual_onset in visual_2_onsets:
        following_window = visual_onset + following_window_size

        for odour_onset in combined_stable_odour_onsets:
            if odour_onset > visual_onset and odour_onset <= following_window:
                vis_2_onsets_in_stable_odour_trials.append(visual_onset)

    return vis_1_onsets_in_stable_odour_trials, vis_2_onsets_in_stable_odour_trials

def normalise_trace(trace):
    trace = np.divide(trace, np.max(trace))
    return trace



def visualise_onsets(onsets_list, traces_list, colour_list=['y', 'b', 'r', 'g', 'm']):

    for onset_type in onsets_list:
        onsets     = onset_type[0]
        onset_name = onset_type[1]

        plt.title(onset_name)

        for trace_index in range(len(traces_list)):
            trace = traces_list[trace_index]
            colour = colour_list[trace_index]
            plt.plot(trace, c=colour)

        plt.scatter(onsets, np.ones(len(onsets))*np.max(traces_list))
        plt.show()


def visualise_raw_traces(ai_recorder_data):
    number_of_traces = np.shape(ai_recorder_data)[0]

    for trace in range(number_of_traces):
        plt.title(trace)
        plt.plot(ai_recorder_data[trace])
        plt.show()



def exclude_trial_outside_imaging_window(onsets_list, first_frame_time, last_frame_time, buffer_window=5000):
    included_onsets = []

    for onset in onsets_list:
        if onset > (first_frame_time + buffer_window) and onset < (last_frame_time - buffer_window):
            included_onsets.append(onset)

    return included_onsets


def get_closest(list, value):
    return min(list, key=lambda x: abs(x - value))


def turn_onsets_to_offsets(onsets, trace):
    step_size = 1

    offsets = []
    for onset in onsets:
        searching = True
        timepoint = onset
        initial_value = trace[onset]
        while searching:
            value = trace[timepoint]
            difference = initial_value - value
            print(difference)
            if difference > step_size:
                offsets.append(timepoint)
                searching = False
            else:
                timepoint += 1

    return offsets


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


def get_intervals(trial_list):

    intervals = []
    for trial in trial_list:
        number_of_onsets = len(trial)
        for onset_index in range(1, number_of_onsets):
            interval = trial[onset_index] - trial[onset_index-1]
            intervals.append(interval)

    return intervals


def organise_sweep_onsets(sweeps_per_trial, trial_order, number_of_trials, sweep_onsets):

    horiontal_onsets = []
    vertical_onsets = []

    count = 0
    for trial in range(number_of_trials):

        trial_type = trial_order[trial]
        trial_onsets = sweep_onsets[count:count + sweeps_per_trial]
        count += sweeps_per_trial

        if trial_type == 1:
            horiontal_onsets.append(trial_onsets)
        else:
            vertical_onsets.append(trial_onsets)

    return horiontal_onsets, vertical_onsets


def get_matlab_filename(base_directory):

    file_list = os.listdir(base_directory)
    for file in file_list:
        file_split = file.split(".")
        if file_split[-1] == 'mat':
            return "/" + file



def check_photodiode_times(base_directory):

    # Set Filenames
    base_directory = base_directory + "/"
    ai_filename = get_ai_filename(base_directory)
    matlab_filename = get_matlab_filename(base_directory)

    # Load Matlab Data
    matlab_data = mat73.loadmat(base_directory + matlab_filename)
    matlab_data = matlab_data['presentationData']
    sweeps_per_trial = int(matlab_data['sweeps'])
    number_of_trials = int(np.max(matlab_data['trialNumber']))
    trial_order = matlab_data['trialType']
    trials_per_direction = int(matlab_data['trialsPerDirection'])
    display_period = matlab_data['period']

    print("Getting Photodiode Times")
    print("Sweeps Per Trial:", sweeps_per_trial)
    print("Number Of Trials: ", number_of_trials)
    print("Trial Order: ", trial_order)
    print("Trials Per Direction", trials_per_direction)
    print("Display Period", display_period)

    # For that one session without the Mat File
    """
    sweeps_per_trial = 10
    number_of_trials = 20
    trial_order = [0,0,0,1,0,1,1,0,0,0,1,1,1,1,1,0,1,0,0,1]
    """

    # Get Stimuli Dictionary
    stimuli_dictionary = create_stimuli_dictionary()

    # Load Photodiode Data
    ai_data = load_ai_recorder_file(base_directory + ai_filename)
    photodiode_trace = ai_data[0]
    photodiode_trace = np.subtract(np.ones(len(photodiode_trace)), photodiode_trace)
    photodiode_trace = np.subtract(photodiode_trace, np.min(photodiode_trace))
    photodiode_trace = np.divide(photodiode_trace, np.max(photodiode_trace))

    sweep_onsets, sweep_line = get_step_onsets(photodiode_trace, threshold=0.9, window=500)

    plt.title("Photodiode Trace")
    plt.plot(photodiode_trace)
    plt.scatter(sweep_onsets, np.ones(len(sweep_onsets)), c='g')
    plt.show()

    # Orgnaise Sweep Onsets
    horizontal_onsets, vertical_onsets = organise_sweep_onsets(sweeps_per_trial, trial_order, number_of_trials, sweep_onsets)
    horizontal_onsets_flattened = np.ndarray.flatten(np.array(horizontal_onsets))
    vertical_onsets_flattened = np.ndarray.flatten(np.array(vertical_onsets))

    plt.title("Photodiode Trace")
    plt.plot(photodiode_trace)
    plt.scatter(horizontal_onsets_flattened, np.ones(len(horizontal_onsets_flattened)), c='g')
    plt.scatter(vertical_onsets_flattened, np.ones(len(vertical_onsets_flattened)), c='tab:orange')
    plt.show()

    # Get Frame Times
    frame_stream = ai_data[stimuli_dictionary["LED 1"]]
    frame_onsets = get_frame_indexes(frame_stream)

    # Get Nearest Frame For Stimuli Onsets
    horizontal_intervals = get_intervals(horizontal_onsets)
    vertical_intervals = get_intervals(vertical_onsets)

    plt.title("Horizontal Distribution")
    plt.hist(horizontal_intervals)
    plt.show()

    plt.title("Vertical Distribution")
    plt.hist(vertical_intervals)
    plt.show()


    horizontal_frame_onsets = []
    for trial in range(int(number_of_trials/2)):
        trial_onsets = get_nearest_frame(horizontal_onsets[trial], frame_onsets)
        horizontal_frame_onsets.append(trial_onsets)

    vertical_frame_onsets = []
    for trial in range(int(number_of_trials/2)):
        trial_onsets = get_nearest_frame(vertical_onsets[trial], frame_onsets)
        vertical_frame_onsets.append(trial_onsets)

    # Save Onsets
    save_directory = base_directory + "/Stimuli_Onsets"
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    np.save(save_directory + "/Horizontal_Onsets.npy", horizontal_onsets)
    np.save(save_directory + "/Vertical_Onsets.npy", vertical_onsets)
    np.save(save_directory + "/Frame_Onsets.npy", frame_onsets)
    np.save(save_directory + "/Horizontal_Frame_Onsets.npy", horizontal_frame_onsets)
    np.save(save_directory + "/Vertical_Frame_Onsets.npy", vertical_frame_onsets)



