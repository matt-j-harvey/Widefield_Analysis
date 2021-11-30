import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
import os
import math
import scipy
import tables
from bisect import bisect_left
import cv2
from sklearn.decomposition import TruncatedSVD
from pathlib import Path
import joblib
from scipy import signal, ndimage, stats
from skimage.transform import resize
from scipy.interpolate import interp1d
import sys
from sklearn.linear_model import LinearRegression


sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions




def get_onsets(base_directory, onsets_file_list):

    # Load Onsets
    onsets = []
    for onsets_file in onsets_file_list:
        onsets_file_contents = np.load(os.path.join(base_directory, "Stimuli_Onsets", onsets_file))
        for onset in onsets_file_contents:
            onsets.append(onset)

    return onsets


def get_selected_pixels(selected_regions, pixel_assignments):

    # Get Pixels Within Selected Regions
    selected_pixels = []
    for region in selected_regions:
        region_mask = np.where(pixel_assignments == region, 1, 0)
        region_indicies = np.nonzero(region_mask)[0]
        for index in region_indicies:
                selected_pixels.append(index)
        selected_pixels.sort()

    return selected_pixels


def get_condition_running_traces(running_trace, onets, frame_times, trial_start_window, trial_stop_window):

    condition_running_traces = []

    # Get Realtime Start and Stops
    realtime_onsets = []
    realtime_durations = []

    for onset in onets:
        trial_start = onset + trial_start_window
        trial_stop = onset + trial_stop_window

        trial_start_realtime = frame_times[trial_start]
        trial_stop_realtime = frame_times[trial_stop]

        realtime_onsets.append(trial_start_realtime)
        realtime_durations.append(trial_stop_realtime - trial_start_realtime)

    minimum_duration = np.min(realtime_durations)

    for realtime_onset in realtime_onsets:
        trial_running_trace = running_trace[realtime_onset:realtime_onset + minimum_duration]
        condition_running_traces.append(trial_running_trace)

    condition_running_traces = np.array(condition_running_traces)
    return condition_running_traces


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


def downsample_running_traces(running_tensor, number_of_timepoints):

    downsampled_traces = []
    for trace in running_tensor:
        downsampled_trace = ResampleLinear1D(trace, number_of_timepoints)
        downsampled_traces.append(downsampled_trace)

    downsampled_traces = np.array(downsampled_traces)
    return downsampled_traces


def view_cortical_vector(base_directory, vector):

    # Load Mask
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    map_image = np.zeros(image_width * image_height)
    map_image[indicies] = vector
    map_image = np.ndarray.reshape(map_image, (image_height, image_width))

    map_magnitude = np.max(np.abs(vector))

    plt.imshow(map_image, cmap='bwr', vmin=-1 * map_magnitude, vmax=map_magnitude)
    plt.show()


def perform_reression_pixelwise(visual_activity_tensor, odour_activity_tensor, visual_running_traces, odour_running_traces):

    print("Visual Activity tensor shape", np.shape(visual_activity_tensor))
    print("Odour Activity Tensor Shape", np.shape(odour_activity_tensor))
    print("Visual Running Traces", np.shape(visual_running_traces))
    print("Odour Running Traces", np.shape(odour_running_traces))

    number_of_visual_trials = np.shape(visual_activity_tensor)[0]
    number_of_odour_trials = np.shape(odour_activity_tensor)[0]
    number_of_timepoints = np.shape(odour_activity_tensor)[1]
    number_of_pixels = np.shape(visual_activity_tensor)[2]

    concatenated_activity_tensor = np.vstack([visual_activity_tensor, odour_activity_tensor])
    concatenated_activity_tensor = np.reshape(concatenated_activity_tensor, ((number_of_odour_trials+number_of_visual_trials) * number_of_timepoints, number_of_pixels))

    concatenated_running_traces = np.vstack([visual_running_traces, odour_running_traces])
    concatenated_running_traces = np.reshape(concatenated_running_traces, ((number_of_odour_trials+number_of_visual_trials) * number_of_timepoints))

    coefficient_vector = []
    intercept_vector = []

    for pixel_index in range(number_of_pixels):
        model = LinearRegression(fit_intercept=True)
        model.fit(X=concatenated_running_traces.reshape(-1, 1), y=concatenated_activity_tensor[:, pixel_index].reshape(-1, 1))
        coef = model.coef_[0][0]
        intercept = model.intercept_[0]
        coefficient_vector.append(coef)
        intercept_vector.append(intercept)
        print(int((float(pixel_index) / number_of_pixels * 100)), "%", " Pixel: ", pixel_index, " of ", number_of_pixels, " Coef ", coef, " Intercept ", intercept)

    view_cortical_vector(base_directory, coefficient_vector)
    view_cortical_vector(base_directory, intercept_vector)

    return coefficient_vector, intercept_vector



def perform_reression(visual_activity_tensor, odour_activity_tensor, visual_running_traces, odour_running_traces):

    number_of_visual_trials = np.shape(visual_activity_tensor)[0]
    number_of_odour_trials = np.shape(odour_activity_tensor)[0]
    number_of_timepoints = np.shape(odour_activity_tensor)[1]

    concatenated_activity_tensor = np.vstack([visual_activity_tensor, odour_activity_tensor])
    concatenated_activity_tensor = np.reshape(concatenated_activity_tensor, ((number_of_odour_trials+number_of_visual_trials) * number_of_timepoints))

    concatenated_running_traces = np.vstack([visual_running_traces, odour_running_traces])
    concatenated_running_traces = np.reshape(concatenated_running_traces, ((number_of_odour_trials+number_of_visual_trials) * number_of_timepoints))

    """
    print("Concatenated Activity Tensor Shape", np.shape(concatenated_activity_tensor))
    print("Visual Activity tensor shape", np.shape(visual_activity_tensor))
    print("Odour Activity Tensor Shape", np.shape(odour_activity_tensor))
    print("Visual Running Traces", np.shape(visual_running_traces))
    print("Odour Running Traces", np.shape(odour_running_traces))
    print("Concatenated running traces", np.shape(concatenated_running_traces))
    print("Concatenated Activity Tensor", np.shape(concatenated_activity_tensor))
    print("Concateanted Running Trace", np.shape(concatenated_running_traces))
    """

    model = LinearRegression(fit_intercept=True)
    model.fit(X=concatenated_running_traces.reshape(-1, 1), y=concatenated_activity_tensor.reshape(-1, 1))

    return model



def get_activity_tensor(activity_matrix, onsets, start_window, stop_window):

    number_of_pixels = np.shape(activity_matrix)[1]
    number_of_trials = np.shape(onsets)[0]
    number_of_timepoints = stop_window - start_window

    # Create Empty Tensor To Hold Data
    activity_tensor = np.zeros((number_of_trials, number_of_timepoints, number_of_pixels))

    # Get Correlation Matrix For Each Trial
    for trial_index in range(number_of_trials):

        # Get Trial Activity
        trial_start = onsets[trial_index] + start_window
        trial_stop = onsets[trial_index] + stop_window
        trial_activity = activity_matrix[trial_start:trial_stop]
        activity_tensor[trial_index] = trial_activity

    return activity_tensor


def get_mean_region_trace(activity_tensor, selected_pixels):

    # Get Mean Region Traces
    region_pixel_responses = activity_tensor[:, :, selected_pixels]
    region_pixel_responses = np.nan_to_num(region_pixel_responses)
    region_responses = np.mean(region_pixel_responses, axis=2)
    region_responses = np.nan_to_num(region_responses)
    return region_responses


def get_corrected_traces(activity_tensor, running_traces, model):


    number_of_trials = np.shape(activity_tensor)[0]
    number_of_timepoints = np.shape(activity_tensor)[1]
    residual_activity_tensor = []
    for trial in range(number_of_trials):
        raw_activity = activity_tensor[trial]
        running_trace = running_traces[trial]
        predicted_activity = model.predict(running_trace.reshape(-1, 1))
        predicted_activity = np.ndarray.flatten(predicted_activity)
        residual_activity = np.subtract(raw_activity, predicted_activity)
        residual_activity_tensor.append(residual_activity)
        """
        figure_1 = plt.figure()
        raw_axis        = figure_1.add_subplot(1, 4, 1)
        running_axis    = figure_1.add_subplot(1, 4, 2)
        predicted_axis  = figure_1.add_subplot(1, 4, 3)
        corrected_axis  = figure_1.add_subplot(1, 4, 4)

  
        raw_axis.plot(raw_activity)
        running_axis.plot(running_trace)
        predicted_axis.plot(predicted_activity)

        corrected_axis.plot(raw_activity, c='b')
        corrected_axis.plot(residual_activity, c='g')
        plt.show()
        """
    residual_activity_tensor = np.array(residual_activity_tensor)
    return residual_activity_tensor


def plot_individual_response(base_directory, trial_start, trial_stop, visual_responses, odour_responses, save_directory, plot_name=None, colour='blue'):


    # Get X Values
    x_values = list(range(trial_start, trial_stop))
    x_values = np.multiply(x_values, 36)

    # Get Average Trace
    visual_average = np.mean(visual_responses, axis=0)
    odour_average = np.mean(odour_responses, axis=0)
    visual_average = np.nan_to_num(visual_average)
    odour_average = np.nan_to_num(odour_average)

    # Get STD
    visual_sd = scipy.stats.sem(visual_responses, axis=0)
    odour_sd = scipy.stats.sem(odour_responses, axis=0)
    visual_sd = np.nan_to_num(visual_sd)
    odour_sd = np.nan_to_num(odour_sd)

    # Check For Significance
    t_stats, p_values = stats.ttest_ind(visual_responses, odour_responses)
    significant_points = []
    count = 0
    for timepoint in x_values:
        if p_values[count] < 0.05:
            significant_points.append(timepoint)
        count += 1

    max_value = np.max([np.add(visual_average, visual_sd), np.add(odour_average, odour_sd)])

    plt.plot(x_values, visual_average, c=colour, marker='o')
    plt.plot(x_values, odour_average, c=colour, linestyle='dotted', marker='^')
    plt.fill_between(x_values, np.add(visual_average, visual_sd), np.subtract(visual_average, visual_sd), alpha=0.1, color=colour)
    plt.fill_between(x_values, np.add(odour_average, odour_sd), np.subtract(odour_average, odour_sd), alpha=0.1, color=colour)
    plt.scatter(significant_points, np.ones(len(significant_points)) * max_value, c='k')

    # Add Onset Line
    plt.axvline([0], c='k')

    # Set Plot Title
    if plot_name != None:
        plt.title(plot_name)

    # Ensure Save Directory Exists
    """
    full_save_path = os.path.join(base_directory, save_directory)
    if not os.path.isdir(full_save_path):
        os.mkdir(full_save_path)
    """
    #plt.savefig(full_save_path + ".png")
    #plt.close()
    plt.show()




def get_running_regression_coefficients(base_directory, visual_onsets_file_list, odour_onsets_file_list, trial_start, trial_stop, coefficient_save_name):

    # Load Frame Times
    frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    frame_times = Widefield_General_Functions.invert_dictionary(frame_times)

    # Load AI Recorder File
    ai_filename = Widefield_General_Functions.get_ai_filename(base_directory)
    ai_data = Widefield_General_Functions.load_ai_recorder_file(base_directory + "/" + ai_filename)
    running_trace = ai_data[8]

    # Load Delta F Matrix
    delta_f_matrix_filepath = os.path.join(base_directory, "Delta_F.h5")
    delta_f_matrix_container = tables.open_file(delta_f_matrix_filepath, mode='r')
    delta_f_matrix = delta_f_matrix_container.root['Data']

    # Load Onsets
    visual_onsets = get_onsets(base_directory, visual_onsets_file_list)
    odour_onsets = get_onsets(base_directory, odour_onsets_file_list)

    # Create Trial Tensor
    visual_activity_tensor = get_activity_tensor(delta_f_matrix, visual_onsets, trial_start, trial_stop)
    odour_activity_tensor  = get_activity_tensor(delta_f_matrix, odour_onsets, trial_start, trial_stop)

    # Load Running Traces
    visual_running_traces = get_condition_running_traces(running_trace, visual_onsets, frame_times, trial_start, trial_stop)
    odour_running_traces = get_condition_running_traces(running_trace, odour_onsets, frame_times, trial_start, trial_stop)

    # Downsample Running Traces
    number_of_timepoints = np.abs(trial_stop - trial_start)
    visual_running_traces = downsample_running_traces(visual_running_traces, number_of_timepoints)
    odour_running_traces = downsample_running_traces(odour_running_traces, number_of_timepoints)

    # Get Regression Coefficients
    coefficient_vector, intercept_vector = perform_reression_pixelwise(visual_activity_tensor, odour_activity_tensor, visual_running_traces, odour_running_traces)

    # Save These Coefficients
    coefficient_save_directory = os.path.join(base_directory, "Running_Regression_Coefficients")
    if not os.path.exists(coefficient_save_directory):
        os.mkdir(coefficient_save_directory)

    view_cortical_vector(base_directory, coefficient_vector)
    view_cortical_vector(base_directory, intercept_vector)

    np.save(os.path.join(coefficient_save_directory, coefficient_save_name, "_Coefficient_Vector.npy"), coefficient_vector)
    np.save(os.path.join(coefficient_save_directory, coefficient_save_name, "_Intercept_Vector.npy"),   intercept_vector)



def get_running_regression_coefficients_region_mean(base_directory, visual_onsets_file_list, odour_onsets_file_list, trial_start, trial_stop, selected_regions):

    # Load Frame Times
    frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    frame_times = Widefield_General_Functions.invert_dictionary(frame_times)

    # Load AI Recorder File
    ai_filename = Widefield_General_Functions.get_ai_filename(base_directory)
    ai_data = Widefield_General_Functions.load_ai_recorder_file(base_directory + "/" + ai_filename)
    running_trace = ai_data[8]

    # Open Atlas Labels
    atlas_labels = np.recfromcsv(r"/home/matthew/Documents/Allen_Atlas_Templates/Atlas_Labels.csv")

    # Load Region Assignments
    pixel_assignments = np.load(os.path.join(base_directory, "Pixel_Assignmnets.npy"))

    # Get Pixels Within Selected Regions
    selected_pixels = get_selected_pixels(selected_regions, pixel_assignments)

    # Load Delta F Matrix
    delta_f_matrix_filepath = os.path.join(base_directory, "Delta_F.h5")
    delta_f_matrix_container = tables.open_file(delta_f_matrix_filepath, mode='r')
    delta_f_matrix = delta_f_matrix_container.root['Data']

    # Load Onsets
    visual_onsets = get_onsets(base_directory, visual_onsets_file_list)
    odour_onsets = get_onsets(base_directory, odour_onsets_file_list)

    # Create Trial Tensor
    visual_activity_tensor = get_activity_tensor(delta_f_matrix, visual_onsets, trial_start, trial_stop)
    odour_activity_tensor  = get_activity_tensor(delta_f_matrix, odour_onsets, trial_start, trial_stop)

    visual_activity_tensor = get_mean_region_trace(visual_activity_tensor, selected_pixels)
    odour_activity_tensor = get_mean_region_trace(odour_activity_tensor, selected_pixels)

    # Load Running Traces
    visual_running_traces = get_condition_running_traces(running_trace, visual_onsets, frame_times, trial_start, trial_stop)
    odour_running_traces = get_condition_running_traces(running_trace, odour_onsets, frame_times, trial_start, trial_stop)

    # Downsample Running Traces
    number_of_timepoints = np.abs(trial_stop - trial_start)
    visual_running_traces = downsample_running_traces(visual_running_traces, number_of_timepoints)
    odour_running_traces = downsample_running_traces(odour_running_traces, number_of_timepoints)
    model = perform_reression(visual_activity_tensor, odour_activity_tensor, visual_running_traces, odour_running_traces)

    corrected_visual_traces = get_corrected_traces(visual_activity_tensor, visual_running_traces, model)
    corrected_odour_traces = get_corrected_traces(odour_activity_tensor, odour_running_traces, model)

    plot_individual_response(base_directory, trial_start, trial_stop, corrected_visual_traces, corrected_odour_traces, None, None)



controls = ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN78.1A/2020_12_09_Switching_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN78.1D/2020_11_29_Switching_Imaging"]

mutants =  ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK12.1F/2021_09_22_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN71.2A/2020_12_17_Switching_Imaging"]


v1 = [45, 46]
pmv = [47, 48]
amv = [39, 40]
rsc = [32, 28]
s1 = [21, 24]
m2 = [8, 9]
visual_cortex = v1 + pmv + amv

"""
trial_start = -65
trial_stop = -4
visual_onsets_file_list = ["visual_context_stable_vis_1_frame_onsets.npy", "visual_context_stable_vis_2_frame_onsets.npy"]
odour_onsets_file_list = ["odour_context_stable_vis_1_frame_onsets.npy", "odour_context_stable_vis_2_frame_onsets.npy"]
"""

trial_start = -10
trial_stop = 40
visual_onsets_file_list = ["visual_context_stable_vis_2_frame_onsets.npy"]
odour_onsets_file_list = ["odour_context_stable_vis_2_frame_onsets.npy"]

x_values = list(range(trial_start, trial_stop))
x_values = np.multiply(x_values, 36)

for base_directory in controls:
    get_running_regression_coefficients(base_directory, visual_onsets_file_list, odour_onsets_file_list, trial_start, trial_stop, "Vis_2_Stable_Response")