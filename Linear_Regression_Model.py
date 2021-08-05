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





def create_visual_stimuli_regressors(onsets, start_window, stop_window, base_directory, visual_stimulus):

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
    visual_stimuli_regressors = np.zeros((number_of_trials, trial_window_size))

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
        trial_relative_frame_offset = trial_relvative_frame_onset + stim_duration_in_frames

        #print("Visual time onset", visual_time_onset, "Visual time offset", visual_time_offset)
        #print("Onset: ", trial_relvative_frame_onset, "Offset: ", trial_relative_frame_offset)

        # Add To Visual Design Matrix
        visual_stimuli_regressors[visual_onset_index][trial_relvative_frame_onset:trial_relative_frame_offset] = np.ones(stim_duration_in_frames)

        #plt.scatter([visual_time_onset, visual_time_offset],[1, 1])
        #plt.plot(visual_trace)
        #plt.show()

    #plt.imshow(visual_design_matrix)
    #plt.show()
    visual_stimuli_regressors = np.reshape(visual_stimuli_regressors, (np.shape(visual_stimuli_regressors)[0], np.shape(visual_stimuli_regressors)[1], 1))
    return visual_stimuli_regressors



def create_bodycam_regressors(base_directory, onsets, save_directory, number_of_mousecam_components=100):

    # Get File Names
    bodycam_file = get_bodycam_filename(base_directory)
    ai_file = get_ai_filename(base_directory)

    # Get All Selected Widefield Frames
    selected_widefield_frames = get_selected_widefield_frames(onsets, start_window, stop_window)

    # Get Video Details
    bodycam_frames, video_height, video_width = get_video_details(base_directory + bodycam_file)

    # Get Number of Mousecam triggers
    stimuli_dictionary = create_stimuli_dictionary()
    ai_data = load_ai_recorder_file(base_directory + ai_file)
    mousecam_trace = ai_data[stimuli_dictionary["Mousecam"]]
    mousecam_onsets, mousecam_line = get_step_onsets(mousecam_trace, threshold=2, window=2)
    if bodycam_frames != len(mousecam_onsets):
        print("Frame Mismatch!", "Bodycam Frames: ", bodycam_frames, "Mousecam Onsets", len(mousecam_onsets))

    # Match Mousecam Frames To Widefield Frames
    mousecam_widefield_matching_dict = match_mousecam_frames_to_widefield_frames(mousecam_onsets, base_directory)
    widefield_mousecam_dict = invert_dictionary(mousecam_widefield_matching_dict)
    selected_mousecam_frames = get_selected_mousecam_frames(selected_widefield_frames, widefield_mousecam_dict)

    # Extract Selected Mousecam Frames From Video
    #video_array = load_video_as_numpy_array(base_directory + bodycam_file, selected_mousecam_frames)

    # Perform SVD on These Frames
    #transformed_data, components = perform_svd_on_video(video_array, number_of_components=number_of_mousecam_components)

    # Save The Resulting Components and Transformed Data
    video_array = None
    #np.save(save_directory + "/Mousecam_SVD_Transformed_Data.npy", transformed_data)
    #np.save(save_directory + "/Mousecam_SVD_Components.npy", components)

    transformed_data = np.load(save_directory + "/Mousecam_SVD_Transformed_Data.npy")

    # Visualise These Components
    mousecam_components = np.load(save_directory + "/Mousecam_SVD_Components.npy")
    visualise_mousecam_components(mousecam_components, video_height, video_width, save_directory)

    return transformed_data


def exclude_selected_bodycam_components(bodycam_matrix, selected_components):

    for component in selected_components:
        bodycam_matrix[component] = 0

    return bodycam_matrix



def create_design_matrix(ai_regressors, visual_stimuli_regressors, bodycam_regressors):

    # Get Data Details
    number_of_trials = np.shape(ai_regressors)[0]
    trial_length = np.shape(ai_regressors)[1]
    number_of_datapoints = number_of_trials * trial_length

    #Combine These Into A Single Matrix
    design_matrix = np.concatenate([ai_regressors, visual_stimuli_regressors, bodycam_regressors], axis=2)
    print("Design MAtrix", np.shape(design_matrix))

    # Reshape Design Matrix from (Trials, Timepoint, Regressors) To (Trials * Timepoints, Regressors)
    design_matrix = np.reshape(design_matrix, (number_of_datapoints, np.shape(design_matrix)[2]))

    return design_matrix


def perform_regression(design_matrix, widefield_matrix, save_directory):

    print("Design Matrix Shape", np.shape(design_matrix))
    print("Widefield MAtrix Shaoe", np.shape(widefield_matrix))

    # Reshape Widefield Data From (Trials, Timepoints, Pixels) to (Trials * TImepoints, Pixels)
    widefield_matrix = np.reshape(widefield_matrix, (np.shape(widefield_matrix)[0] * np.shape(widefield_matrix)[1], np.shape(widefield_matrix)[2]))

    print("Performing Regression")
    model = Ridge()
    model.fit(X=design_matrix, y=widefield_matrix)

    joblib.dump(model, save_directory + "/Linear_Model.pkl")



def explore_regression(base_directory, save_directory):

    # Load Model
    model = joblib.load(save_directory + "/Linear_Model.pkl")

    # Get Coefficients
    weights = model.coef_
    print("Weights", np.shape(weights))

    indicies, image_height, image_width = load_mask(base_directory)


    # Lick = 0
    lick_map = create_image_from_data(weights[:, 0], image_height, image_width, indicies)
    plt.title("Lick")
    plt.imshow(lick_map)
    plt.show()

    # Running = 1
    running_map = create_image_from_data(weights[:, 1], image_height, image_width, indicies)
    plt.title("Running")
    plt.imshow(running_map)
    plt.show()

    # Visual_stimuli
    vis_stim_map = create_image_from_data(weights[:, 2], image_height, image_width, indicies)
    plt.title("Visual")
    plt.imshow(vis_stim_map)
    plt.show()








# Perform Preperations

# Settings
base_directory = r"/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Switching_Analysis/Selected_sessions/NXAK16.1B/2021_06_23_Switching_Imaging/"
condition = "visual_context_stable_vis_2"
start_window = -10
stop_window = 100
number_of_mousecam_components = 100
mousecam_components_to_exclude = [2,3]

# Check Linear Model Directory Exists
linear_model_directory = base_directory + "/Linear_Model"
check_directory(linear_model_directory)

# Create Save Directory
save_directory = linear_model_directory + "/" + condition
check_directory(save_directory)

# Load Frame Onsets and Frame Times
frame_onsets_file = base_directory + "/Stimuli_Onsets/" + condition + "_frame_onsets.npy"
frame_onsets = np.load(frame_onsets_file)




# Extract Widefield Data
widefield_file = base_directory + "/Delta_F.h5"
widefield_data_file = tables.open_file(widefield_file, mode='r')
widefield_data = widefield_data_file.root['Data']

# Get All Selected Widefield Frames
selected_widefield_frames = get_selected_widefield_frames(frame_onsets, start_window, stop_window)

# Extract These Frames From THe Delta_F.h5 File
selected_widefield_data = get_selected_widefield_data(selected_widefield_frames, widefield_data)



# Create Regressors

# Create AI Regressors
ai_regressors = create_ai_recorder_regressors(base_directory, frame_onsets, start_window, stop_window)
print("Ai Regressors", np.shape(ai_regressors))

# Create Visual Stimuli Regressors
visual_stimuli_regressors = create_visual_stimuli_regressors(frame_onsets, start_window, stop_window, base_directory, "Visual 2")
print("Visual Stiimuli Regressors", np.shape(visual_stimuli_regressors))

# Create Bodycam Regressors
bodycam_regressors = create_bodycam_regressors(base_directory, frame_onsets, save_directory, number_of_mousecam_components=number_of_mousecam_components)
bodycam_regressors = exclude_selected_bodycam_components(bodycam_regressors, mousecam_components_to_exclude)
print("Bodycam regressors shape", np.shape(bodycam_regressors))


# Create Design Matrix
design_matrix = create_design_matrix(ai_regressors, visual_stimuli_regressors, bodycam_regressors)



# Perform Regression
perform_regression(design_matrix, selected_widefield_data, save_directory)


# Explore Regression
explore_regression(base_directory, save_directory)


# Create Design Matrix
# One Model For Each Context
# Running speed
# Lick Trace
# Behaviour Video SVDs



