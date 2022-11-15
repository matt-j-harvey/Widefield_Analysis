import pandas as pd
from bisect import bisect_left
import os
import tables
import numpy as np

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
            return original_filename

def load_ai_recorder_file(base_directory):

    ai_filename = get_ai_filename(base_directory)
    ai_recorder_file_location = os.path.join(base_directory, ai_filename)

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


def flatten(l):
    return [item for sublist in l for item in sublist]


def load_mouse_sessions(mouse_name, session_type):

    # This Is The Location of My Experimental Logbook
    logbook_file_location = r"/home/matthew/Documents/Experiment_Logbook.ods"

    #  Read Logbook As A Dataframe
    logbook_dataframe = pd.read_excel(logbook_file_location, engine="odf")

    # Return A List Of the File Directories Of Sessions Matching The Mouse Name and Session Type
    selected_sessions = logbook_dataframe.loc[(logbook_dataframe["Mouse"] == mouse_name) & (logbook_dataframe["Session Type"] == session_type), ["Filepath"]].values.tolist()

    # Flatten The Subsequent Nested List
    selected_sessions = flatten(selected_sessions)

    return selected_sessions



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

def load_all_sessions_of_type(session_type):

    # This Is The Location of My Experimental Logbook
    logbook_file_location = r"/home/matthew/Documents/Experiment_Logbook.ods"

    #  Read Logbook As A Dataframe
    logbook_dataframe = pd.read_excel(logbook_file_location, engine="odf")

    # Return A List Of the File Directories Of Sessions Matching The Mouse Name and Session Type
    selected_sessions = logbook_dataframe.loc[(logbook_dataframe["Session Type"] == session_type), ["Filepath"]].values.tolist()

    # Flatten The Subsequent Nested List
    selected_sessions = flatten(selected_sessions)

    return selected_sessions


def take_closest(myList, myNumber):

    """
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.
    """

    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before

def get_step_onsets(trace, threshold=1, window=3):
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

    return onset_times
