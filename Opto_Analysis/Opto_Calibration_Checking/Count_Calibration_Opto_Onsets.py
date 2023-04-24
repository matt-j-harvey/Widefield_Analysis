import numpy as np
import matplotlib.pyplot as plt
import tables
from tqdm import tqdm
import os

from Widefield_Utils import widefield_utils


def get_opto_stim_log_file(base_directory):
    file_list = os.listdir(base_directory)
    for file_name in file_list:
        if "Opto_Stim_Log.h5" in file_name:
            return file_name


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



def plot_mean_intensity_trace(base_directory):

    mean_blue_file = os.path.join(base_directory, "Mean_Frame_intensities.npy")
    mean_violet_file = os.path.join(base_directory, "Mean_Frame_intensities_Violet.npy")

    mean_blue_data = np.load(mean_blue_file)
    mean_violet_data = np.load(mean_violet_file)

    plt.plot(mean_blue_data, c='b')
    plt.plot(mean_violet_data, c='m')
    plt.show()

    # Get Step Onsets
    opto_onsets = get_step_onsets(mean_data, threshold=18000)
    number_of_onsets = len(opto_onsets)

    plt.plot(mean_data)
    plt.scatter(opto_onsets, np.multiply(np.ones(len(opto_onsets)), 18000))
    plt.title(str(number_of_onsets))
    plt.show()


def normalise_trace(trace):
    min_value = np.min(trace)
    trace = np.subtract(trace, min_value)
    max_value = np.max(trace)
    trace = np.divide(trace, max_value)
    return trace


def plot_mean_intensity_trace_with_visual_stim(base_directory):


    mean_blue_file = os.path.join(base_directory, "Mean_Frame_intensities.npy")
    mean_violet_file = os.path.join(base_directory, "Mean_Frame_intensities_Violet.npy")

    mean_blue_data = np.load(mean_blue_file)
    mean_violet_data = np.load(mean_violet_file)

    # Load Vis Trace
    downsampled_ai_matrix = np.load(os.path.join(base_directory, "Downsampled_AI_Matrix_Framewise.npy"))
    stimuli_dictionary = widefield_utils.create_stimuli_dictionary()
    vis_1_trace = downsampled_ai_matrix[stimuli_dictionary["Visual 1"]]
    vis_2_trace = downsampled_ai_matrix[stimuli_dictionary["Visual 2"]]
    combined_matrix = np.vstack([vis_1_trace, vis_2_trace])
    combined_trace = np.max(combined_matrix, axis=0)
    print("Combined Matrix Shape", np.shape(combined_matrix))


    # Normalise
    mean_blue_data = normalise_trace(mean_blue_data)
    mean_violet_data = normalise_trace(mean_violet_data)
    combined_trace = normalise_trace(combined_trace)

    plt.plot(mean_blue_data, c='b')
    plt.plot(mean_violet_data, c='m')
    plt.plot(combined_trace, c='g')
    plt.show()



def get_camera_onsets(base_directory):
    mean_blue_file = os.path.join(base_directory, "Mean_Frame_intensities.npy")
    mean_blue_data = np.load(mean_blue_file)
    camera_onsets = get_step_onsets(mean_blue_data, threshold=18000)
    return camera_onsets


def get_ai_onsets(base_directory):
    downsampled_ai_matrix = np.load(os.path.join(base_directory, "Downsampled_AI_Matrix_Framewise.npy"))
    stimuli_dictionary = widefield_utils.create_stimuli_dictionary()
    opto_trace = downsampled_ai_matrix[stimuli_dictionary['Optogenetics']]
    ai_onsets = get_step_onsets(opto_trace, threshold=1.5)
    return ai_onsets


def get_log_file_onsets(base_directory):
    opto_log_filename = get_opto_stim_log_file(base_directory)
    opto_log_file = tables.open_file(os.path.join(base_directory, opto_log_filename), mode="r")
    opto_log_stim_images = opto_log_file.root["Stim_Images"]
    return opto_log_stim_images

def check_opto_onsets(base_directory):


    # Create Save Directory
    save_directory = os.path.join(base_directory, "Stimuli_Onsets")
    if not os.path.exists:
        os.mkdir(save_directory)

    # Get Camera Recorded Onsets
    camera_onsets = get_camera_onsets(base_directory)
    n_camera_onsets = len(camera_onsets)
    print("n_camera_onsets", n_camera_onsets)

    # Get AI Recorded Onsets
    ai_onsets = get_ai_onsets(base_directory)
    n_ai_onsets = len(ai_onsets)
    print("n_ai_onsets", n_ai_onsets)

    # Get Opto Log Stimuli
    opto_log_images = get_log_file_onsets(base_directory)
    n_opto_log_images = np.shape(opto_log_images)[0]
    print("n_opto_log_images", n_opto_log_images)

    frame_numbers_match = False
    if n_camera_onsets == n_ai_onsets and n_ai_onsets == n_opto_log_images:
        print("Stimuli Match Hiperdipip Horrea! :D")
        frame_numbers_match = True


    # Save onsets
    np.save(os.path.join(save_directory, "Opto_Onset_Frames.npy"), ai_onsets)

    return frame_numbers_match


