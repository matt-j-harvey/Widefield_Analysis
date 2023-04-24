import numpy as np
import matplotlib.pyplot as plt
import tables
import os
from tqdm import tqdm

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


def get_opto_stim_log_file(base_directory):
    file_list = os.listdir(base_directory)
    for file_name in file_list:
        if "Opto_Stim_Log.h5" in file_name:
            return file_name


def classify_pattern(pattern, unique_patterns):

    n_unique_patterns = np.shape(unique_patterns)[0]
    for unique_pattern_index in range(n_unique_patterns):
        if np.array_equal(pattern, unique_patterns[unique_pattern_index]):
            return unique_pattern_index


def check_stim_classification(base_directory):

    # Load Opto Onset Frames
    opto_onset_frames = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Opto_Onset_Frames.npy"))

    # Get Means
    mean_frame_intesities = np.load(os.path.join(base_directory, "Mean_Frame_intensities_Violet.npy"))

    # Load Camera Data
    camera_data_file = os.path.join(base_directory, "Downsampled_Delta_F.h5")
    camera_data_file_container = tables.open_file(camera_data_file, mode='r')
    camera_data = camera_data_file_container.root["Data"]

    # Load Opto Patterns
    opto_log_filename = get_opto_stim_log_file(base_directory)
    opto_log_file = tables.open_file(os.path.join(base_directory, opto_log_filename), mode="r")
    opto_stim_patterns = opto_log_file.root["Stim_Images"]

    unique_patterns = np.unique(opto_stim_patterns, axis=0)
    n_unique_patterns = len(unique_patterns)

    # Classify Opto Patterns From Log
    opto_pattern_lables = []
    for perturbation in opto_stim_patterns:
        perturbation_label = classify_pattern(perturbation, unique_patterns)
        opto_pattern_lables.append(perturbation_label)

    # Create Save Direcotires
    save_directory_list = []
    classified_peturbation_list = []
    for pattern_index in range(n_unique_patterns):
        pattern_save_directory = os.path.join(base_directory, "Pattern_" + str(pattern_index).zfill(3))
        save_directory_list.append(pattern_save_directory)
        classified_peturbation_list.append([])
        if not os.path.exists(pattern_save_directory):
            os.makedirs(pattern_save_directory)

    # Get Camera Image For Each Onset
    n_opto_perturbs = len(opto_onset_frames)
    for petrub_index in tqdm(range(n_opto_perturbs)):

        # Get Pattern Label
        pattern_label = opto_pattern_lables[petrub_index]

        # Get Pattern Save Directory
        pattern_save_directory = save_directory_list[pattern_label]

        # Get Pattern Image
        onset = opto_onset_frames[petrub_index]
        trial_data = camera_data[onset:onset+10]
        trial_data = np.max(trial_data, axis=0)
        trial_data = np.reshape(trial_data, (600, 608))

        classified_peturbation_list[pattern_label].append(trial_data)

        # Plot Pattern Image
        plt.title(str(pattern_label))
        plt.imshow(trial_data, vmin=0, vmax=50000)
        plt.savefig(os.path.join(pattern_save_directory, str(petrub_index).zfill(3) + ".png"))
        plt.close()

    # Save Max Projections
    max_projection_list = []
    for pattern_group in classified_peturbation_list:
        pattern_max_projection = np.max(pattern_group, axis=0)
        max_projection_list.append(pattern_max_projection)
    np.save(os.path.join(base_directory, "max_projection_list.npy"), max_projection_list)

    return max_projection_list


