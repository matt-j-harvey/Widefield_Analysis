import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
import os
import tables
from scipy import signal, ndimage, stats
from sklearn.neighbors import KernelDensity
import cv2
from matplotlib import gridspec

import Retinotopy_Utils

def reconstruct_video(base_directory, data):

    # Load Mask
    indicies, image_height, image_width = Retinotopy_Utils.load_downsampled_mask(base_directory)

    print("Data Shape", np.shape(data))
    number_of_frames = np.shape(data)[0]

    # Set Colourmap
    vmin = -0.05
    vmax = 0.05
    colourmap = Retinotopy_Utils.get_musall_cmap()

    plt.ion()
    for frame in range(number_of_frames):
        frame_data = data[frame]
        frame_data = Retinotopy_Utils.create_image_from_data(frame_data, indicies, image_height, image_width)
        plt.title(frame)
        plt.imshow(template, cmap='jet', vmin=0, vmax=1)
        plt.draw()
        plt.pause(0.001)
        plt.clf()

    plt.close()
    plt.ioff()



def get_stimuli_average(preprocessed_data, stimuli_onsets, trial_details):

    #Get_Trial_Details
    trial_start = trial_details[0]
    #use_baseline = trial_details[3]

    use_baseline = False

    #Get Data From All Trials
    all_trials = []
    for onset in stimuli_onsets:
        if onset != None:
            trial_data = get_single_trial_trace(onset, preprocessed_data, trial_details)
            all_trials.append(trial_data)
    all_trials = np.array(all_trials)
    all_trials = np.nan_to_num(all_trials)

    if use_baseline == False:
        trial_average = np.mean(all_trials, axis=0)

    elif use_baseline == True:
        baseline_window = all_trials[:, 0:-1 * trial_start]

        baseline_mean = np.mean(baseline_window, axis=0)
        baseline_mean = np.mean(baseline_mean, axis=0)

        trial_average = np.subtract(all_trials, baseline_mean)
        trial_average = np.divide(trial_average, baseline_mean)

        trial_average = np.mean(trial_average, axis=0)

    trial_average = np.nan_to_num(trial_average)

    return trial_average, all_trials




def get_single_trial_trace(onset, preprocessed_data, trial_details):

    #Get Trial Details
    trial_start     = trial_details[0]
    trial_end       = trial_details[1]
    window_size     = trial_details[2]

    window_start = onset + trial_start
    window_stop = onset + trial_end

    trial_data = []
    for timepoint in range(window_start, window_stop):
        window_data = preprocessed_data[timepoint - window_size : timepoint + window_size]
        window_mean = np.mean(window_data, axis=0)
        trial_data.append(window_mean)

    trial_data = np.array(trial_data)
    return trial_data



def save_evoked_responses(home_directory, name, matrix, type="Average"):

    # Check Stimuli Evoked Responses Directory Exists
    responses_save_location = home_directory + "/Stimuli_Evoked_Responses"
    if not os.path.exists(responses_save_location):
        os.mkdir(responses_save_location)

    #Create Save Directory
    save_directory = responses_save_location + "/" + name
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    #Get File Name
    if type == "Average":
        filename = save_directory + "/" + name + "_Activity_Matrix_Average.npy"
    elif type == "All_Trials":
        filename = save_directory + "/" + name + "_Activity_Matrix_All_Trials.npy"
    np.save(filename, matrix)



def save_trial_details(home_directory, name, stimuli_onsets, trial_details):

    #Get Trial Details
    trial_start     = trial_details[0]
    trial_end       = trial_details[1]
    window_size     = trial_details[2]
    use_baseline    = trial_details[3]

    number_of_trials = np.shape(stimuli_onsets)[0]
    print("Number of trials", number_of_trials)

    current_trial_details = np.zeros((number_of_trials, 4), dtype=int)

    for trial in range(number_of_trials):
        onset = stimuli_onsets[trial]
        window_start = onset + trial_start
        window_stop = onset + trial_end

        current_trial_details[trial, 0] = int(window_start)
        current_trial_details[trial, 1] = int(window_stop)
        current_trial_details[trial, 2] = int(window_size)
        current_trial_details[trial, 3] = int(use_baseline)

    # Create Save Directory
    responses_save_location = home_directory + "/Stimuli_Evoked_Responses"
    save_directory = responses_save_location + "/" + name
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Get File Name
    filename = save_directory + "/" + name + "_Trial_Details.npy"
    np.save(filename, current_trial_details)





def extract_trial_aligned_activity(home_directory):

    # Trial Average Details
    trial_onset_filenames = ["Horizontal_Frame_Onsets.npy", "Vertical_Frame_Onsets.npy"]
    condition_names = ["Horizontal_Sweep", "Vertical_Sweep"]

    trial_start = 0
    trial_end = 422
    window_size = 2
    use_baseline = False
    trial_details = [trial_start, trial_end, window_size, use_baseline]

    #Setup File Structure
    preprocessed_data_file_location = os.path.join(home_directory, "Downsampled_Delta_F.h5")
    responses_save_location         = home_directory

    if not os.path.exists(responses_save_location):
        os.mkdir(responses_save_location)

    #Load Processed Data
    preprocessed_data_file = tables.open_file(preprocessed_data_file_location, mode='r')
    preprocessed_data = preprocessed_data_file.root['Data']

    #Load Trial Onsets
    condition_1_onset_file = os.path.join(home_directory, "Stimuli_Onsets", trial_onset_filenames[0])
    condition_2_onset_file = os.path.join(home_directory, "Stimuli_Onsets", trial_onset_filenames[1])

    condition_1_onsets = np.load(condition_1_onset_file, allow_pickle=True)
    condition_2_onsets = np.load(condition_2_onset_file, allow_pickle=True)

    condition_1_onsets = np.ndarray.flatten(condition_1_onsets)
    condition_2_onsets = np.ndarray.flatten(condition_2_onsets)


    #Remove Nones
    condition_1_onsets = list(filter(None, condition_1_onsets))
    condition_2_onsets = list(filter(None, condition_2_onsets))

    # Extract Average Activity
    condition_1_average, condition_1_all_trials = get_stimuli_average(preprocessed_data, condition_1_onsets, trial_details)

    # Save Average Activity Matricies
    save_evoked_responses(home_directory, condition_names[0], condition_1_average, type="Average")

    # Save All Trials Activity Matricies
    save_evoked_responses(home_directory, condition_names[0], condition_1_all_trials, type="All_Trials")

    # Save Trial Details
    save_trial_details(home_directory, condition_names[0], condition_1_onsets, trial_details)

    # Free Memory
    condition_1_all_trials = None
    condition_1_average = None

    # Repeat for condition 2
    condition_2_average, condition_2_all_trials = get_stimuli_average(preprocessed_data, condition_2_onsets, trial_details)
    save_evoked_responses(home_directory, condition_names[1], condition_2_average, type="Average")
    save_evoked_responses(home_directory, condition_names[1], condition_2_all_trials, type="All_Trials")
    save_trial_details(home_directory, condition_names[1], condition_2_onsets, trial_details)



