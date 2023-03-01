import os
import tables
import h5py
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from Widefield_Utils import widefield_utils
from Behaviour_Analysis import Behaviour_Utils


def get_opto_stim_log_file(base_directory):
    file_list = os.listdir(base_directory)
    for file_name in file_list:
        if "Opto_Stim_Log.h5" in file_name:
            return file_name

def get_blue_data_file(base_directory):
    file_list = os.listdir(base_directory)
    for file_name in file_list:
        if "_Blue_Data.hdf5" in file_name:
            return file_name




def draw_stimuli_onsets(blue_data, opto_onset_list, output_directory, preceeding_frames=-5, following_frames=5):

    number_of_stim_trials = len(opto_onset_list)
    number_of_trial_timepoints = following_frames - preceeding_frames

    # Create Figure
    """
    figure_1 = plt.figure()
    n_rows = number_of_stim_trials
    n_cols = number_of_trial_timepoints
    axis_count = 1
    """

    print("Opto onset frames", opto_onset_list)


    for trial_index in tqdm(range(number_of_stim_trials)):
        trial_onset = opto_onset_list[trial_index]
        trial_start = trial_onset + preceeding_frames
        trial_stop = trial_onset + following_frames
        trial_data = blue_data[:, trial_start:trial_stop]
        print("Trial Onset", trial_onset, "trial start", trial_start, "trial stop", trial_stop, "Trial Data", np.shape(trial_data))

        trial_data = np.reshape(trial_data, (600, 608, number_of_trial_timepoints))

        figure_1 = plt.figure()
        n_rows = 1
        n_cols = number_of_trial_timepoints
        for frame in range(number_of_trial_timepoints):
            axis = figure_1.add_subplot(n_rows, n_cols, frame + 1)
            axis.imshow(trial_data[:, :, frame])
            axis.axis('off')
            plt.title(frame)

        plt.savefig(os.path.join(output_directory, str(trial_index).zfill(3)))



def sort_onsets_by_patterns(onset_frame_list, stim_patterns_list, unique_stim_patterns):

    print("unique stim patterns", np.shape(unique_stim_patterns))

    # Get Number Of Unique Stim Patterns
    number_of_unique_stim_patterns = np.shape(unique_stim_patterns)[0]
    nested_onset_list = []
    for x in range(number_of_unique_stim_patterns):
        nested_onset_list.append([])

    number_of_stimuli = len(stim_patterns_list)
    for stim_index in range(number_of_stimuli):
        stim_onset = onset_frame_list[stim_index]
        stim_pattern = stim_patterns_list[stim_index]

        for unique_pattern_index in range(number_of_unique_stim_patterns):
            if np.array_equal(stim_pattern, unique_stim_patterns[unique_pattern_index]):
                print("Stim", stim_index, "is type", unique_pattern_index)
                nested_onset_list[unique_pattern_index].append(stim_onset)
            else:
                print("nope")

    return nested_onset_list



def check_opto_stim_machine_calibration(base_directory):

    # Create output Folder
    output_folder = os.path.join(base_directory, "Stimuli_Calibration_Checks")
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Load AI File
    ai_data = widefield_utils.load_ai_recorder_file(base_directory)
    stimuli_dictionary = widefield_utils.create_stimuli_dictionary()

    # Get Frame Times
    blue_led_trace = ai_data[stimuli_dictionary["LED 1"]]
    blue_onsets = Behaviour_Utils.get_step_onsets(blue_led_trace)
    np.save(os.path.join(output_folder, "Frame_Onsets.npy"), blue_onsets)
    plt.plot(blue_led_trace)
    plt.scatter(blue_onsets, np.ones(len(blue_onsets)))
    plt.show()

    # Get Opto Onsets
    opto_trace = ai_data[stimuli_dictionary["Optogenetics"]]
    opto_onset_list = Behaviour_Utils.get_step_onsets(opto_trace)
    np.save(os.path.join(output_folder, "Opto_Onset_Times.npy"), opto_onset_list)
    plt.plot(opto_trace)
    plt.scatter(opto_onset_list, np.ones(len(opto_onset_list)))
    plt.show()

    # Match Opto Onsets To Frames
    opto_onset_frames = []
    for opto_onset in opto_onset_list:
        closet_frame_onset = Behaviour_Utils.take_closest(blue_onsets, opto_onset)
        closet_frame = blue_onsets.index(closet_frame_onset)
        opto_onset_frames.append(closet_frame)
    np.save(os.path.join(output_folder, "opto_onset_frames.npy"), opto_onset_frames)

    # Load Opto ROI Log
    opto_log_filename = get_opto_stim_log_file(base_directory)
    opto_log_file = tables.open_file(os.path.join(base_directory, opto_log_filename), mode="r")
    opto_log_stim_images = opto_log_file.root["Stim_Images"]
    opto_log_timestamps = opto_log_file.root["Timestamps"]
    print("Opto Stimages", np.shape(opto_log_stim_images))

    # Match Onsets to ROI Log
    number_of_opto_triggers = len(opto_onset_list)
    number_of_opto_timestamps = np.shape(opto_log_timestamps)[0]
    print("Number Of Opto Triggers", number_of_opto_triggers)
    print("Number Of Opto Timestamps", number_of_opto_timestamps)
    if number_of_opto_triggers == number_of_opto_timestamps:
        print("Numbers Match :) Hiperdepiep Hoera")

    # Classify Opto Stims
    unique_stim_patterns = np.unique(opto_log_stim_images, axis=0)
    number_of_unique_stim_patterns = np.shape(unique_stim_patterns)[0]
    print("number_of_unique_stim_patterns", number_of_unique_stim_patterns)
    sorted_opto_stim_onsets = sort_onsets_by_patterns(opto_onset_frames, opto_log_stim_images, unique_stim_patterns)
    for group in sorted_opto_stim_onsets:
        print("sorted Opto Stim Onsts", group)

    # Plot Opto Stim Patterns
    figure_1 = plt.figure()
    for stim_pattern_index in tqdm(range(number_of_unique_stim_patterns)):
        stim_axis = figure_1.add_subplot(1, number_of_unique_stim_patterns, stim_pattern_index + 1)
        stim_axis.imshow(250 - unique_stim_patterns[stim_pattern_index], vmin=0, vmax=255)
        stim_axis.set_title("Stim_pattern_" + str(stim_pattern_index).zfill(3))
    plt.savefig(os.path.join(output_folder, "Stimuli_Patterns"))
    plt.close()





    # Load Blue Data
    blue_data_file = get_blue_data_file(base_directory)
    file_container = h5py.File(os.path.join(base_directory, blue_data_file), mode="r")
    blue_data = file_container["Data"]
    print("Blue Data Shape", np.shape(blue_data))

    """
    blue_data_sample = np.array(blue_data[:, 0:20000])
    print("BLue Data Sample", np.shape(blue_data_sample))
    blue_data_max_projection = np.max(blue_data_sample, axis=1)
    blue_data_max_projection = np.reshape(blue_data_max_projection, (600, 608))
    plt.imshow(blue_data_max_projection)
    plt.show()
    print("Unique Stim Patterns", np.shape(unique_stim_patterns))
    """

    # Extract Camera Data For These Frames
    for pattern_index in range(number_of_unique_stim_patterns):

        pattern_image_save_directory = os.path.join(output_folder, "Stim_" + str(pattern_index))
        if not os.path.exists(pattern_image_save_directory):
            os.mkdir(pattern_image_save_directory)

        draw_stimuli_onsets(blue_data, sorted_opto_stim_onsets[pattern_index], pattern_image_save_directory, preceeding_frames=-1, following_frames=7)



    # Plot Camera ROI Trace For Each Time


base_directory = r"/media/matthew/External_Harddrive_3/Opto_Test/Opto_Stim_Machine_Calibration/2023_02_14_Opto_Stim_Machine_Calibration"
check_opto_stim_machine_calibration(base_directory)