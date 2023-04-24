import os

import cv2
import tables
import h5py
from tqdm import tqdm
from skimage.feature import canny

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




def draw_stimuli_onsets(blue_data, opto_onset_list, opto_roi_mask, output_directory, preceeding_frames=-5, following_frames=5):

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

    # Get Mask Edges
    mask_edges = canny(opto_roi_mask)
    mask_indicies = np.nonzero(mask_edges)
    plt.imshow(mask_edges)
    plt.show()

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
            frame_image = trial_data[:, :, frame]
            frame_image[mask_indicies] = 65535
            axis.imshow(frame_image)
            axis.axis('off')
            plt.title(frame)

        plt.savefig(os.path.join(output_directory, str(trial_index).zfill(3)))
        plt.close()



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
                nested_onset_list[unique_pattern_index].append(stim_onset)

    return nested_onset_list



def get_opto_onset_frames(base_directory):

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



def check_opto_onset_frames_equal_saved_stim_data(base_directory):

    # Set Output Folder
    output_folder = os.path.join(base_directory, "Stimuli_Calibration_Checks")

    # Load Opto Onsets
    opto_onset_list = np.load(os.path.join(output_folder, "Opto_Onset_Times.npy"))

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
    else:
        print("Mismatch :(")



def get_roi_masks(base_directory):

    # Load Homography Matrix
    homography_matrix = np.load(r"/media/matthew/External_Harddrive_3/Opto_Test/Opto_Stim_Machine_Calibration/2023_02_14_Opto_Stim_Machine_Calibration/Homograhpy_Matrix.npy")

    # Load Opto Patterns
    opto_log_filename = get_opto_stim_log_file(base_directory)
    opto_log_file = tables.open_file(os.path.join(base_directory, opto_log_filename), mode="r")
    opto_log_stim_images = opto_log_file.root["Stim_Images"]
    unique_stim_patterns = np.unique(opto_log_stim_images, axis=0)

    # Get Inverse Maps
    stim_roi_masks = []
    for image in unique_stim_patterns:
        image = 250 - image
        image_height, image_width = np.shape(image)
        inverse_image = cv2.warpPerspective(image, homography_matrix, (image_width, image_height), flags=cv2.WARP_INVERSE_MAP)
        inverse_image = np.transpose(inverse_image)

        print("Image Shape", np.shape(inverse_image))
        plt.imshow(inverse_image)
        plt.show()

        stim_roi_masks.append(inverse_image)

    # Save These
    np.save(os.path.join(base_directory, "Stimuli_Calibration_Checks", "Stim_ROI_Masks.npy"), np.array(stim_roi_masks))


def get_sorted_opto_onsets(base_directory):

    # Set Output Folder
    output_folder = os.path.join(base_directory, "Stimuli_Calibration_Checks")

    # Load Opto Onset Frames
    opto_onset_frames = np.load(os.path.join(output_folder, "opto_onset_frames.npy"))

    # Load Opto ROI Log
    opto_log_filename = get_opto_stim_log_file(base_directory)
    opto_log_file = tables.open_file(os.path.join(base_directory, opto_log_filename), mode="r")
    opto_log_stim_images = opto_log_file.root["Stim_Images"]

    # Classify Opto Stims
    unique_stim_patterns = np.unique(opto_log_stim_images, axis=0)
    sorted_opto_stim_onsets = sort_onsets_by_patterns(opto_onset_frames, opto_log_stim_images, unique_stim_patterns)
    np.save(os.path.join(output_folder, "Sorted_Opto_Stim_Onsets.npy"), sorted_opto_stim_onsets)



def view_opto_stim_presentations(base_directory):

    # Set Output Folder
    output_folder = os.path.join(base_directory, "Stimuli_Calibration_Checks")

    # Load Opto Onset Frames
    opto_onset_frames = np.load(os.path.join(output_folder, "opto_onset_frames.npy"))

    # Load Opto ROI Log
    opto_log_filename = get_opto_stim_log_file(base_directory)
    opto_log_file = tables.open_file(os.path.join(base_directory, opto_log_filename), mode="r")
    opto_log_stim_images = opto_log_file.root["Stim_Images"]
    opto_log_timestamps = opto_log_file.root["Timestamps"]

    # Classify Opto Stims
    unique_stim_patterns = np.unique(opto_log_stim_images, axis=0)
    number_of_unique_stim_patterns = np.shape(unique_stim_patterns)[0]
    sorted_opto_stim_onsets = sort_onsets_by_patterns(opto_onset_frames, opto_log_stim_images, unique_stim_patterns)
    np.save(os.path.join(output_folder, "Sorted_Opto_Stim_Onsets.npy"), sorted_opto_stim_onsets)

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

    # Load ROI Masks
    opto_roi_masks = np.load(os.path.join(output_folder, "Stim_ROI_Masks.npy"))

    # Extract Camera Data For These Frames
    for pattern_index in range(number_of_unique_stim_patterns):

        pattern_image_save_directory = os.path.join(output_folder, "Stim_" + str(pattern_index))
        if not os.path.exists(pattern_image_save_directory):
            os.mkdir(pattern_image_save_directory)

        draw_stimuli_onsets(blue_data, sorted_opto_stim_onsets[pattern_index], opto_roi_masks[pattern_index], pattern_image_save_directory, preceeding_frames=-1, following_frames=7)




def get_example_opto_stims(base_directory):

    # Set Output Folder
    output_folder = os.path.join(base_directory, "Stimuli_Calibration_Checks")

    # Load Opto ROI Log
    opto_log_filename = get_opto_stim_log_file(base_directory)
    opto_log_file = tables.open_file(os.path.join(base_directory, opto_log_filename), mode="r")
    opto_log_stim_images = opto_log_file.root["Stim_Images"]

    # Classify Opto Stims
    unique_stim_patterns = np.unique(opto_log_stim_images, axis=0)
    np.save(os.path.join(output_folder, "Unique_Sim_Patterns.npy"), unique_stim_patterns)



def get_opto_stim_camera_examples(base_directory):

    # Load Sorted Opto Frame Onsets
    sorted_onsets = np.load(os.path.join(base_directory, "Stimuli_Calibration_Checks", "Sorted_Opto_Stim_Onsets.npy"), allow_pickle=True)

    print("Sorted Onsets", sorted_onsets)

    # Load Blue Data
    blue_data_file = get_blue_data_file(base_directory)
    file_container = h5py.File(os.path.join(base_directory, blue_data_file), mode="r")
    blue_data = file_container["Data"]
    print("Blue Data Shape", np.shape(blue_data))

    for stim_type_onsets in sorted_onsets:
        first_onset = stim_type_onsets[0]
        example_data = blue_data[:, first_onset:first_onset + 10]
        print("Example Data Shape", np.shape(example_data))
        example_data = np.max(example_data, axis=1)
        example_data = np.reshape(example_data, (600, 608))
        plt.imshow(example_data)
        plt.show()



def get_stim_traces(blue_data, stim_onsets, stim_roi):

    stim_traces = []

    stim_window_size = 20

    for onset in tqdm(stim_onsets):
        trial_data = blue_data[:, onset:onset + stim_window_size]
        trial_data = np.reshape(trial_data, (600, 608, stim_window_size))
        trial_data = trial_data[stim_roi[0][0]:stim_roi[0][1], stim_roi[1][0]:stim_roi[1][1]]
        roi_height, roi_width, stim_window_size = np.shape(trial_data)
        trial_data = np.reshape(trial_data, (roi_height * roi_width, stim_window_size))
        trial_data = np.mean(trial_data, axis=0)
        stim_traces.append(trial_data)

    return stim_traces

def get_roi_timecourses(base_directory):

    # Load Sorted Opto Frame Onsets
    sorted_onsets = np.load(os.path.join(base_directory, "Stimuli_Calibration_Checks", "Sorted_Opto_Stim_Onsets.npy"), allow_pickle=True)

    print("Sorted Onsets", sorted_onsets)

    # Load Blue Data
    blue_data_file = get_blue_data_file(base_directory)
    file_container = h5py.File(os.path.join(base_directory, blue_data_file), mode="r")
    blue_data = file_container["Data"]
    print("Blue Data Shape", np.shape(blue_data))

    # Set ROI Coords
    roi_1 = [[388, 425], [357, 404]]
    roi_2 = [[375, 430], [154, 217]]
    roi_3 = [[192, 249], [355, 409]]

    # Unpack
    roi_1_onsets = sorted_onsets[0]
    roi_2_onsets = sorted_onsets[1]
    roi_3_onsets = sorted_onsets[2]

    roi_1_stim_traces = get_stim_traces(blue_data, roi_1_onsets, roi_1)
    roi_2_stim_traces = get_stim_traces(blue_data, roi_2_onsets, roi_2)
    roi_3_stim_traces = get_stim_traces(blue_data, roi_3_onsets, roi_3)

    np.save(os.path.join(base_directory, "Stimuli_Calibration_Checks", "ROI_1_Traces.npy"), roi_1_stim_traces)
    np.save(os.path.join(base_directory, "Stimuli_Calibration_Checks", "ROI_2_Traces.npy"), roi_2_stim_traces)
    np.save(os.path.join(base_directory, "Stimuli_Calibration_Checks", "ROI_3_Traces.npy"), roi_3_stim_traces)


    for trace in roi_1_stim_traces:
        plt.plot(trace, c='b', alpha=0.3)

    for trace in roi_2_stim_traces:
        plt.plot(trace, c='r', alpha=0.3)

    for trace in roi_3_stim_traces:
        plt.plot(trace, c='g', alpha=0.3)

    plt.show()



def count_steps_untill_threshold(trace, threshold):
    count = 0
    for value in trace:
        if value > threshold:
            return count

        count += 1



def get_trace_distribution(trace_list, threshold):

    distribution = []
    for trace in trace_list:
        delay = count_steps_untill_threshold(trace, threshold)
        distribution.append(delay)

    return distribution


def plot_histogram(base_directory, threshold=10000):


    roi_1_stim_traces = np.load(os.path.join(base_directory, "Stimuli_Calibration_Checks", "ROI_1_Traces.npy"))
    roi_2_stim_traces = np.load(os.path.join(base_directory, "Stimuli_Calibration_Checks", "ROI_2_Traces.npy"))
    roi_3_stim_traces = np.load(os.path.join(base_directory, "Stimuli_Calibration_Checks", "ROI_3_Traces.npy"))

    roi_1_distribution = get_trace_distribution(roi_1_stim_traces, threshold)
    roi_2_distribution = get_trace_distribution(roi_2_stim_traces, threshold)
    roi_3_distribution = get_trace_distribution(roi_3_stim_traces, threshold)

    plt.hist(roi_1_distribution, color='b', alpha=0.3)
    plt.hist(roi_2_distribution, color='r', alpha=0.3)
    plt.hist(roi_3_distribution, color='g', alpha=0.3)
    plt.show()

base_directory = r"/media/matthew/External_Harddrive_3/Opto_Test/Opto_Stim_Machine_Calibration/2023_02_14_Opto_Stim_Machine_Calibration"


# Get Opto Onsets
#get_opto_onset_frames(base_directory)

# Check Opto Onsets Equal Saved Timestamps
#check_opto_onset_frames_equal_saved_stim_data(base_directory)

# Get Unique Stim Patterns
#get_unique_sim_patterns(base_directory)

# Get Sorted Opto Onsets
#get_sorted_opto_onsets(base_directory)

"""
# Get ROI Masks
get_roi_masks(base_directory)


# Plot Camera Images Around Stim Onsets
view_opto_stim_presentations(base_directory)
"""

#get_opto_stim_camera_examples(base_directory)

#get_roi_timecourses(base_directory)

plot_histogram(base_directory)