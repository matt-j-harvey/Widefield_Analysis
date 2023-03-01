import math

import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
import os
import tables
from scipy import signal, ndimage, stats
from scipy.io import loadmat
from sklearn.neighbors import KernelDensity
import cv2
from matplotlib import gridspec, patches

from Widefield_Utils import widefield_utils
import Create_Behaviour_Matrix_Discrimination


def get_fsm_file(base_directory):

    file_list = os.listdir(base_directory)

    for file in file_list:
        spit_file = file.split(".")
        if spit_file[-1] =='mat':
            return file

    return None


def get_opto_trials(base_directory):

    # Get Matlab Filename
    matlab_file = get_fsm_file(base_directory)
    print(matlab_file)

    # Load Laser Powers
    laser_powers = loadmat(os.path.join(base_directory, "Stimuli_Onsets", "laser_powers.mat"))['laser_powers'][0]
    print(laser_powers)

    # Load Opto IDs
    opto_ids = loadmat(os.path.join(base_directory, "Stimuli_Onsets", "opto_stim_ids.mat"))['image_ids']
    opto_ids = np.ndarray.flatten(opto_ids)
    print(opto_ids)

    number_of_unique_opto_stimuli = len(list(set(opto_ids)))
    opto_id_trial_list = []
    for unique_opto_index in range(number_of_unique_opto_stimuli):
       opto_id_trial_list.append([])

    opto_index = 0
    for trial_index in range(len(laser_powers)):
        if laser_powers[trial_index] != 0:
            opto_stim_index = opto_ids[opto_index]
            opto_id_trial_list[opto_stim_index-1].append(trial_index)
            opto_index += 1

    print("oopto index", opto_index, "presentations", len(opto_ids))
    print("Opto id trial list", opto_id_trial_list)

    return opto_id_trial_list


def get_widefield_onset_frames_for_opto_trials(base_directory, opto_id_trial_list):

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

    # Load Widefield Frame Times
    widefield_frame_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    widefield_frame_times = list(widefield_frame_dict.keys())
    print("widefield Frame Dict", "Keys:", list(widefield_frame_dict.keys())[0:5], "Values: ",  list(widefield_frame_dict.values())[0:5])

    # Create Onset Tensor
    onset_tensor = []
    for stimulus in opto_id_trial_list:
        stimulus_onsets = []
        for trial in stimulus:
            trial_onset = behaviour_matrix[trial][11]
            print("Trial Onset", trial_onset)

            closest_frame_onset = widefield_utils.take_closest(widefield_frame_times, trial_onset)
            print("closest_frame_onset", closest_frame_onset)

            closest_widefield_frame = widefield_frame_dict[closest_frame_onset]
            print("closest_widefield_frame", closest_widefield_frame)
            stimulus_onsets.append(closest_widefield_frame)
        onset_tensor.append(stimulus_onsets)

    return onset_tensor


def get_widefield_name(base_directory):
    file_list = os.listdir(base_directory)
    for file_name in file_list:
        if "widefield.h5" in file_name:
            return file_name
    return None


def get_opto_tensor(base_directory, opto_onsets_tensor, start_window=-10, stop_window=10):

    # Load Blue Data
    widefield_file_name = get_widefield_name(base_directory)
    widefield_data_container = tables.open_file(os.path.join(base_directory, widefield_file_name), mode='r')
    widefield_data = widefield_data_container.root["blue"]
    print("Blue data shape", np.shape(widefield_data))

    # Create Data Tensor
    data_tensor = []
    for stimulus in opto_onsets_tensor:
        stimulus_tensor = []
        for onset in stimulus:
            trial_start = onset + start_window
            trial_stop = onset + stop_window
            print("Trial Start: ", trial_start, "Trial Stop: ", trial_stop)
            trial_data = widefield_data[trial_start:trial_stop]
            stimulus_tensor.append(trial_data)

        # Get Stimulus Mean
        stimulus_tensor = np.array(stimulus_tensor)
        print("stimulus tensor", np.shape(stimulus_tensor))
        stimulus_mean = np.mean(stimulus_tensor, axis=0)
        data_tensor.append(stimulus_mean)

    # Get Stimulus Average
    data_tensor = np.array(data_tensor)
    print("Data Tensor Shape", np.shape(data_tensor))

    # Close Data Container
    widefield_data_container.close()

    return data_tensor


def view_opto_tensor(opto_tensor):

    number_of_stimuli, trial_length, image_height, image_width = np.shape(opto_tensor)

    plt.ion()
    figure_1 = plt.figure()
    rows = 1
    columns = number_of_stimuli
    for timepoint_index in range(trial_length):

        for stimuli_index in range(number_of_stimuli):
            stimuli_axis = figure_1.add_subplot(rows, columns, stimuli_index + 1)
            stimuli_axis.imshow(opto_tensor[stimuli_index, timepoint_index])

        figure_1.suptitle(str(timepoint_index))
        plt.draw()
        plt.pause(0.1)
        plt.clf()


session_list = ["/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Switching_Opto/Calibration_Session/Opto_Test_2022_11_17"]




# Create Behaviour Matricies
for session in session_list:

    # Create Behaviour Matrix For Session
    #Create_Behaviour_Matrix_Discrimination.create_behaviour_matrix(session)

    # Extract Opto Trials
    opto_id_trial_list = get_opto_trials(session)

    # Get Widefield Onset Frames For Each Opto Stimulus
    onset_tensor = get_widefield_onset_frames_for_opto_trials(session, opto_id_trial_list)
    print("Onset Tensor", onset_tensor)

    # Get Tensor Of Opto Data
    opto_tensor = get_opto_tensor(session, onset_tensor)

    # view This
    view_opto_tensor(opto_tensor)