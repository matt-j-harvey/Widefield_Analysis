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
import sys

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Analysis/Movement_Controls/Bodycam_Analysis")

import Get_Bodycam_SVD_Tensor
import Match_Mousecam_Frames_To_Widefield_Frames


def factor_number(number_to_factor):

    factor_list = []

    for potential_factor in range(1, number_to_factor):
        if number_to_factor % potential_factor == 0:
            factor_pair = [potential_factor, int(number_to_factor/potential_factor)]
            factor_list.append(factor_pair)

    return factor_list


def get_best_grid(number_of_items):

    factors = factor_number(number_of_items)
    factor_difference_list = []

    #Get Difference Between All Factors
    for factor_pair in factors:
        factor_difference = abs(factor_pair[0] - factor_pair[1])
        factor_difference_list.append(factor_difference)

    #Select Smallest Factor difference
    smallest_difference = np.min(factor_difference_list)
    best_pair = factor_difference_list.index(smallest_difference)

    return factors[best_pair]


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
            return ai_filename


def get_video_details(video_file):

    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return frameCount, frameHeight, frameWidth


def load_ai_recorder_file(ai_recorder_file_location):
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


def match_mousecam_frames_to_widefield_frames(mousecam_onsets, base_directory):

    # Create Save Directory
    save_directory = base_directory + "/Mousecam_Details"
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # frame Times Keys=Realtime, Value=Frame_Index
    frame_times = np.load(base_directory + "/Stimuli_Onsets/Frame_Times.npy", allow_pickle=True)
    frame_times = frame_times[()]

    # Get Nearest Mousecam Frames
    mousecam_widefield_matching_dict = match_frames(mousecam_onsets, frame_times)

    return mousecam_widefield_matching_dict


def get_bodycam_filename(base_directory):

    file_list = os.listdir(base_directory)

    for file in file_list:
        file_split = file.split('_')
        if file_split[-1] == '1.mp4' and file_split[-2] == 'cam':
            return file


def get_offset(onset, stream, threshold=0.5):

    count = 50
    on = True
    while on:
        if stream[onset + count] < threshold:
            on = False
            return onset + count
        else:
            count += 1


def get_selected_widefield_frames(onsets, start_window, stop_window):

    selected_fames = []

    for onset in onsets:
        trial_start = onset + start_window
        trial_stop = onset + stop_window
        trial_frames = list(range(trial_start, trial_stop))
        selected_fames.append(trial_frames)

    return selected_fames


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

    return onset_times, onset_line


def check_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def perform_svd_on_video(video_array, number_of_components=100):

    # Assumes Video Data is a 3D array with the structure: Trials, Timepoints, height, width
    print("Video array shape", np.shape(video_array))
    number_of_trials = np.shape(video_array)[0]
    trial_length     = np.shape(video_array)[1]
    video_height     = np.shape(video_array)[2]
    video_width      = np.shape(video_array)[3]
    number_of_frames = number_of_trials * trial_length

    # Flatten Video
    video = np.reshape(video_array, (number_of_frames, video_height * video_width))

    # Perform PCA
    pca_model = TruncatedSVD(n_components=number_of_components)
    pca_model.fit(video)
    prinicpal_components = pca_model.components_
    transformed_data = pca_model.transform(video)

    # Put Data Back Into Original Shaoe
    transformed_data = np.reshape(transformed_data, (number_of_trials, trial_length, number_of_components))

    return transformed_data, prinicpal_components


def visualise_mousecam_components(components, video_height, video_width, save_directory):

    number_of_components = np.shape(components)[0]
    [rows, columns] = get_best_grid(number_of_components)

    axis_list = []
    figure_1 = plt.figure()
    for component_index in range(number_of_components):
        component_data = components[component_index]
        component_data = np.abs(component_data)
        component_data = np.reshape(component_data, (video_height, video_width))

        axis_list.append(figure_1.add_subplot(rows, columns, component_index + 1))
        axis_list[component_index].imshow(component_data, cmap='jet')
        axis_list[component_index].set_title(str(component_index))
        axis_list[component_index].axis('off')

    plt.savefig(save_directory + "/SVD_Components.png")
    plt.close()
    #plt.show()


def get_selected_mousecam_frames(selected_widefield_onsets,  widefield_mousecam_dict):

    selected_fames = []
    for trial in selected_widefield_onsets:
        trial_mousecam_frames = []
        for frame in trial:
            mousecam_frame = widefield_mousecam_dict[frame]
            trial_mousecam_frames.append(mousecam_frame)
        selected_fames.append(trial_mousecam_frames)

    selected_fames = np.array(selected_fames)
    return selected_fames


def match_frames(mousecam_onsets, frame_times):

    frame_onsets = list(frame_times.keys())
    mousecam_widefield_matching_dict = {}

    number_of_mousecam_onsets = len(mousecam_onsets)
    for mousecam_frame_index in range(number_of_mousecam_onsets):
        mousecam_frame_time = mousecam_onsets[mousecam_frame_index]

        nearest_widefield_frame_time = take_closest(frame_onsets, mousecam_frame_time)
        nearest_widefield_frame_index = frame_times[nearest_widefield_frame_time]

        mousecam_widefield_matching_dict[mousecam_frame_index] = nearest_widefield_frame_index

    return mousecam_widefield_matching_dict


def load_video_as_numpy_array(video_file, selected_mousecam_frames):

    # Open Video File
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get Trial Details
    number_of_trials = len(selected_mousecam_frames)
    trial_length = len(selected_mousecam_frames[0])
    extracted_frames = np.zeros((number_of_trials, trial_length, frameHeight, frameWidth, 3))

    # Extract Selected Frames
    frame_index = 0
    ret = True
    while (frame_index < frameCount and ret):
        ret, frame = cap.read()
        frame_index += 1

        # See If This Is a Frame We Want
        for trial in range(number_of_trials):
            for timepoint in range(trial_length):
                if frame_index == selected_mousecam_frames[trial][timepoint]:
                    extracted_frames[trial][timepoint] = frame

    cap.release()
    extracted_frames = extracted_frames[:,:,:,:,0]

    return extracted_frames


def get_selected_widefield_data(selected_widefield_onsets, widefield_data):

    selected_widefield_data = []

    for trial in selected_widefield_onsets:
        trial_data = []
        for frame in trial:
            frame_data = widefield_data[frame]
            trial_data.append(frame_data)

        selected_widefield_data.append(trial_data)

    return selected_widefield_data



def load_mask(home_directory):

    # Loads the mask for a video, returns a list of which pixels are included, as well as the original image height and width
    mask = np.load(home_directory + "/mask.npy")

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask>0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width

def create_image_from_data(data, image_height, image_width, indicies):
    template = np.zeros((image_height, image_width))
    data = np.nan_to_num(data)
    np.put(template, indicies, data)
    image = np.ndarray.reshape(template, (image_height, image_width))

    return image



def create_ai_recorder_regressors(base_directory, onsets_list, start_window, stop_window):

    # Get Window Size
    number_of_trials = len(onsets_list)
    window_size = stop_window - start_window


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




def create_visual_stimuli_regressors(onsets, start_window, stop_window, base_directory, visual_stimulus, condition_number):

    # Load AI Data
    ai_file_location = get_ai_filename(base_directory)
    ai_data = load_ai_recorder_file(base_directory + ai_file_location)
    stimuli_dictionary = create_stimuli_dictionary()
    visual_trace = ai_data[stimuli_dictionary[visual_stimulus]]

    # Load Frame Times
    time_frame_dict = np.load(base_directory + "/Stimuli_Onsets/Frame_Times.npy", allow_pickle=True)
    time_frame_dict = time_frame_dict[()]
    frame_time_dict = invert_dictionary(time_frame_dict)
    frame_time_list = list(time_frame_dict.keys())

    # Create Empty Design Matrix
    number_of_trials = len(onsets)
    trial_window_size = stop_window - start_window
    #print("Trial Window Size", trial_window_size)
    visual_stimuli_regressors = np.zeros((number_of_trials, trial_window_size, trial_window_size * 2))

    for visual_onset_index in range(number_of_trials):

        visual_onset = onsets[visual_onset_index]

        # Get Frame Of Stimulus Onset
        trial_relvative_frame_onset = -1 * start_window

        # Get Frame Of Stimulus Offset
        visual_time_onset = frame_time_dict[visual_onset]
        visual_time_offset = get_offset(visual_time_onset, visual_trace)
        closest_frame_time = take_closest(frame_time_list, visual_time_offset)
        closest_frame = time_frame_dict[closest_frame_time]
        stim_duration_in_frames = closest_frame - visual_onset

        if stim_duration_in_frames > stop_window:
            stim_duration_in_frames = stop_window
        trial_relative_frame_offset = trial_relvative_frame_onset + stim_duration_in_frames

        # Create Trial Regressor Matrix
        trial_regressor_matrix = np.zeros((trial_window_size, trial_window_size))
        stimuli_matrix = np.identity((stim_duration_in_frames))
        #print("Stim duration in frames", stim_duration_in_frames)
        #print("Trial regressor matrix", np.shape(trial_regressor_matrix))
        #print("Srimuli MAtrix Shape", np.shape(stimuli_matrix))
        trial_regressor_matrix[trial_relvative_frame_onset:trial_relative_frame_offset, trial_relvative_frame_onset:trial_relative_frame_offset] = stimuli_matrix

        zero_padding = np.zeros((trial_window_size, trial_window_size))

        if condition_number == 1:
            trial_regressor_matrix = np.concatenate([trial_regressor_matrix, zero_padding], axis=1)
        elif condition_number == 2:
            trial_regressor_matrix = np.concatenate([zero_padding, trial_regressor_matrix], axis=1)

        # Add Trial Matrix To Regressors
        visual_stimuli_regressors[visual_onset_index] = trial_regressor_matrix

        #plt.imshow(np.transpose(visual_stimuli_regressors[visual_onset_index]))
        #plt.show()

    return visual_stimuli_regressors






def perform_regression(design_matrix, widefield_matrix, save_directory):

    print("Design Matrix Shape", np.shape(design_matrix))
    print("Widefield MAtrix Shaoe", np.shape(widefield_matrix))

    # Reshape Widefield Data From (Trials, Timepoints, Pixels) to (Trials * TImepoints, Pixels)
    widefield_matrix = np.reshape(widefield_matrix, (np.shape(widefield_matrix)[0] * np.shape(widefield_matrix)[1], np.shape(widefield_matrix)[2]))

    print("Performing Regression")
    model = Ridge()
    model.fit(X=design_matrix, y=widefield_matrix)

    joblib.dump(model, save_directory + "/Linear_Model.pkl")





def create_running_tensor(onsets, trial_start, trial_stop, downsampled_running_trace):

    running_tensor = []
    for onset in onsets:
        start = onset + trial_start
        stop = onset + trial_stop
        trial_running_trace = downsampled_running_trace[start:stop]
        running_tensor.append(trial_running_trace)

    running_tensor = np.array(running_tensor)
    running_tensor = np.expand_dims(running_tensor, 2)
    return running_tensor


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




def create_design_matrix(activity_matrix, running_regressors, visual_stimuli_regressors, bodycam_regressors):

    # Get Data Details
    number_of_trials = np.shape(running_regressors)[0]
    trial_length = np.shape(running_regressors)[1]
    number_of_datapoints = number_of_trials * trial_length
    number_of_pixels = np.shape(activity_matrix)[2]
    number_of_bodycam_components = np.shape(bodycam_regressors)[2]

    # Reshape Each Feature from [Trials x Length x Features] to [Timepoints, Features]
    activity_matrix = np.ndarray.reshape(activity_matrix, (number_of_datapoints, number_of_pixels))
    running_regressors = np.ndarray.reshape(running_regressors, (number_of_datapoints, 1))
    visual_stimuli_regressors = np.ndarray.reshape(visual_stimuli_regressors, (number_of_datapoints, trial_length*2))
    bodycam_regressors = np.ndarray.reshape(bodycam_regressors, (number_of_datapoints, number_of_bodycam_components))

    print("Reshaped Activity Matrix Shape", np.shape(activity_matrix))
    print("Reshaped Running Regressors", np.shape(running_regressors))
    print("Visual Simuli Regressors", np.shape(visual_stimuli_regressors))
    #print("Bodycam Regressors", np.shape(bodycam_regressors))

    #Combine These Into A Single Matrix
    design_matrix = np.concatenate([visual_stimuli_regressors, running_regressors, bodycam_regressors], axis=1)
    print("Design Matrix", np.shape(design_matrix))

    return activity_matrix, design_matrix



def perform_ridge_regression(base_directory, onset_lists, start_window, stop_window, activity_tensor_list, stimuli_list, condition_1_bodycam_tensor, condition_2_bodycam_tensor):

    # Get Activity Tensors
    condition_1_activity_tensor = activity_tensor_list[0]
    condition_2_activity_tensor = activity_tensor_list[1]
    print("Condition 1 activity tensor", np.shape(condition_1_activity_tensor))
    print("Condition 2 activity tensor", np.shape(condition_2_activity_tensor))

    condition_1_onsets = onset_lists[0]
    condition_2_onsets = onset_lists[1]


    # Check We Have A Widefield To Mousecam Frame Dict
    if not os.path.exists(os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy")):
        Match_Mousecam_Frames_To_Widefield_Frames.match_mousecam_to_widefield_frames(base_directory)

    # Get Running Tensors
    downsampled_running_trace = np.load(os.path.join(base_directory, "Movement_Controls", "Downsampled_Running_Trace.npy"))
    condition_1_running_tensor = create_running_tensor(condition_1_onsets, start_window, stop_window, downsampled_running_trace)
    condition_2_running_tensor = create_running_tensor(condition_2_onsets, start_window, stop_window, downsampled_running_trace)

    print("Condition 1 running tensor shape", np.shape(condition_1_running_tensor))
    print("Condition 2 running tensor shapee", np.shape(condition_2_running_tensor))

    # Create Visual Stimuli Regressors

    print("Stimuli list", stimuli_list[0])
    print("Stimuli list", stimuli_list[1])
    condition_1_stimuli_regressors = create_visual_stimuli_regressors(condition_1_onsets, start_window, stop_window, base_directory, stimuli_list[0], 1)
    condition_2_stimuli_regressors = create_visual_stimuli_regressors(condition_2_onsets, start_window, stop_window, base_directory, stimuli_list[1], 2)

    # Stack The Two Conditions Ontop Of Eachother
    activity_matrix = np.vstack([condition_1_activity_tensor, condition_2_activity_tensor])
    visual_stimuli_regressors = np.vstack([condition_1_stimuli_regressors, condition_2_stimuli_regressors])
    running_regressors = np.vstack([condition_1_running_tensor, condition_2_running_tensor])
    bodycam_regresssors = np.vstack([condition_1_bodycam_tensor, condition_2_bodycam_tensor])

    print("Combined Activity Tensors", np.shape(activity_matrix))
    print("Combined Stimuli Regressors", np.shape(visual_stimuli_regressors))
    print("Combined Running Regressors", np.shape(running_regressors))

    # Create Design Matrix
    activity_matrix, design_matrix = create_design_matrix(activity_matrix, running_regressors, visual_stimuli_regressors, bodycam_regresssors)

    # Remove NaNs
    activity_matrix = np.nan_to_num(activity_matrix)

    # Perform Regression
    #(n_samples, n_features)
    model = Ridge()
    model.fit(X=design_matrix, y=activity_matrix)

    # Save Coefficients
    coefficients = model.coef_
    intercepts = model.intercept_

    save_directory = os.path.join(base_directory, "Movement_Controls", "Ridge_Regression")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    np.save(os.path.join(save_directory, "Coefficients.npy"), coefficients)
    np.save(os.path.join(save_directory, "Intercepts.npy"), intercepts)
    #np.save(os.path.join(save_directory, "Ridge_Regression_Bodycam_Components.npy"), bodycam_components)


