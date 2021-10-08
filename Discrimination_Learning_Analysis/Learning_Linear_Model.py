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

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")
import Widefield_General_Functions











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
    [rows, columns] = Widefield_General_Functions.get_best_grid(number_of_components)

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

        nearest_widefield_frame_time = Widefield_General_Functions.take_closest(frame_onsets, mousecam_frame_time)
        nearest_widefield_frame_index = frame_times[nearest_widefield_frame_time]

        mousecam_widefield_matching_dict[mousecam_frame_index] = nearest_widefield_frame_index

    return mousecam_widefield_matching_dict




def get_selected_data(selected_onsets, data):

    selected_data = []

    for trial in selected_onsets:
        trial_data = []
        for frame in trial:
            frame_data = data[frame]
            trial_data.append(frame_data)

        selected_data.append(trial_data)

    return selected_data



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
    ai_file_location = Widefield_General_Functions.get_ai_filename(base_directory)
    ai_data = Widefield_General_Functions.load_ai_recorder_file(base_directory + ai_file_location)

    # Extract Traces
    stimuli_dict = Widefield_General_Functions.create_stimuli_dictionary()
    running_trace = ai_data[stimuli_dict["Running"]]
    lick_trace = ai_data[stimuli_dict["Lick"]]

    # Load Frame Times
    time_frame_dict = np.load(base_directory + "/Stimuli_Onsets/Frame_Times.npy", allow_pickle=True)
    time_frame_dict = time_frame_dict[()]
    frame_time_dict = Widefield_General_Functions.invert_dictionary(time_frame_dict)

    ai_data = np.zeros((number_of_trials, window_size, 2))
    for trial_index in range(number_of_trials):
        onset = onsets_list[trial_index]
        start_frame = onset + start_window
        stop_frame = onset + stop_window

        start_time = frame_time_dict[start_frame]
        stop_time = frame_time_dict[stop_frame]

        trial_lick_data = lick_trace[start_time:stop_time]
        trial_running_data = running_trace[start_time:stop_time]

        trial_lick_data = Widefield_General_Functions.ResampleLinear1D(trial_lick_data,       window_size)
        trial_running_data = Widefield_General_Functions.ResampleLinear1D(trial_running_data, window_size)

    ai_data[trial_index, :, 0] = trial_lick_data
    ai_data[trial_index, :, 1] = trial_running_data

    return ai_data





def create_visual_stimuli_regressors(combined_onsets, vis_1_onsets, vis_2_onsets, start_window, stop_window, base_directory):

    # Load AI Data
    ai_file_location = Widefield_General_Functions.get_ai_filename(base_directory)
    ai_data = Widefield_General_Functions.load_ai_recorder_file(base_directory + ai_file_location)
    stimuli_dictionary = Widefield_General_Functions.create_stimuli_dictionary()
    vis_1_trace = ai_data[stimuli_dictionary["Visual 1"]]
    vis_2_trace = ai_data[stimuli_dictionary["Visual 2"]]
    visual_trace = np.array([vis_1_trace, vis_2_trace])
    visual_trace = np.max(visual_trace, axis=0)

    # Load Frame Times
    time_frame_dict = np.load(base_directory + "/Stimuli_Onsets/Frame_Times.npy", allow_pickle=True)
    time_frame_dict = time_frame_dict[()]
    frame_time_dict = Widefield_General_Functions.invert_dictionary(time_frame_dict)
    frame_time_list = list(time_frame_dict.keys())

    # Create Empty Design Matrix
    number_of_trials = len(combined_onsets)
    trial_window_size = stop_window - start_window
    vis_1_regressors = np.zeros((number_of_trials, trial_window_size, trial_window_size))
    vis_2_regressors = np.zeros((number_of_trials, trial_window_size, trial_window_size))


    for visual_onset_index in range(number_of_trials):

        visual_onset = combined_onsets[visual_onset_index]

        # Get Frame Of Stimulus Onset
        trial_relvative_frame_onset = -1 * start_window

        # Get Frame Of Stimulus Offset
        visual_time_onset = frame_time_dict[visual_onset]
        visual_time_offset = get_offset(visual_time_onset, visual_trace)
        closest_frame_time = Widefield_General_Functions.take_closest(frame_time_list, visual_time_offset)
        closest_frame = time_frame_dict[closest_frame_time]
        stim_duration_in_frames = closest_frame - visual_onset
        trial_relative_frame_offset = trial_relvative_frame_onset + stim_duration_in_frames

        if trial_relative_frame_offset > trial_window_size:
            trial_relative_frame_offset = trial_window_size
            stim_duration_in_frames = stop_window

        # Create Trial Regressor Matrix
        trial_regressor_matrix = np.zeros((trial_window_size, trial_window_size))
        stimuli_matrix = np.identity((stim_duration_in_frames))
        trial_regressor_matrix[trial_relvative_frame_onset:trial_relative_frame_offset, trial_relvative_frame_onset:trial_relative_frame_offset] = stimuli_matrix

        # Add Trial Matrix To Regressors
        if visual_onset in vis_1_onsets:
            vis_1_regressors[visual_onset_index] = trial_regressor_matrix
        elif visual_onset in vis_2_onsets:
            vis_2_regressors[visual_onset_index] = trial_regressor_matrix


    return vis_1_regressors, vis_2_regressors



def create_bodycam_regressors(base_directory, onsets, save_directory, number_of_mousecam_components=100):

    # Load Bodycam SVD
    bodycam_data = np.load(base_directory + "/Mousecam_SVD/transformed_data.npy")
    print("Bodycam Data Shape", np.shape(bodycam_data))
    bodycam_frames = np.shape(bodycam_data)[0]


    # Get All Selected Widefield Frames
    selected_widefield_frames = get_selected_widefield_frames(onsets, start_window, stop_window)

    # Get Number of Mousecam triggers
    stimuli_dictionary = Widefield_General_Functions.create_stimuli_dictionary()
    ai_file = Widefield_General_Functions.get_ai_filename(base_directory)
    ai_data = Widefield_General_Functions.load_ai_recorder_file(base_directory + ai_file)
    mousecam_trace = ai_data[stimuli_dictionary["Mousecam"]]
    mousecam_onsets = Widefield_General_Functions.get_step_onsets(mousecam_trace, threshold=2, window=2)
    print("Mousecam Onsets", len(mousecam_onsets))
    if bodycam_frames != len(mousecam_onsets):
        print("Frame Mismatch!", "Bodycam Frames: ", bodycam_frames, "Mousecam Onsets", len(mousecam_onsets))

    # Match Mousecam Frames To Widefield Frames
    mousecam_widefield_matching_dict = match_mousecam_frames_to_widefield_frames(mousecam_onsets, base_directory)
    widefield_mousecam_dict = Widefield_General_Functions.invert_dictionary(mousecam_widefield_matching_dict)
    selected_mousecam_frames = get_selected_mousecam_frames(selected_widefield_frames, widefield_mousecam_dict)

    # Get Selected Data
    transformed_data = get_selected_data(selected_mousecam_frames, bodycam_data)

    # Visualise These Components
    mousecam_components = np.load(base_directory + "/Mousecam_SVD/components.npy")
    video_height = 480
    video_width = 640
    visualise_mousecam_components(mousecam_components, video_height, video_width, save_directory)

    return transformed_data


def exclude_selected_bodycam_components(bodycam_matrix, selected_components):

    for component in selected_components:
        bodycam_matrix[component] = 0

    return bodycam_matrix



def create_design_matrix(ai_regressors, vis_1_regressors, vis_2_regressors, bodycam_regressors):

    # Get Data Details
    number_of_trials = np.shape(ai_regressors)[0]
    trial_length = np.shape(ai_regressors)[1]
    number_of_datapoints = number_of_trials * trial_length

    #Combine These Into A Single Matrix
    design_matrix = np.concatenate([vis_1_regressors, vis_2_regressors, ai_regressors, bodycam_regressors], axis=2)
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



def explore_regression(base_directory, save_directory, start_window, stop_window):

    # Load Model
    model = joblib.load(save_directory + "/Linear_Model.pkl")

    # Get Coefficients
    weights = model.coef_
    print("Weights", np.shape(weights))

    indicies, image_height, image_width = load_mask(base_directory)
    trial_length = stop_window - start_window

    number_of_predictors = np.shape(weights)[1]

    # Visual_stimuli
    plt.ion()
    for timepoint in range(trial_length):
        vis_stim_map = create_image_from_data(weights[:, timepoint], image_height, image_width, indicies)
        plt.title("Visual: " + str(timepoint))
        plt.imshow(vis_stim_map, cmap='inferno', vmin=0)
        plt.draw()
        plt.pause(0.1)
        plt.clf()
    plt.ioff()

    plt.ion()
    for timepoint in range(trial_length, 2 * trial_length):
        vis_stim_map = create_image_from_data(weights[:, timepoint], image_height, image_width, indicies)
        plt.title("Visual: " + str(timepoint))
        plt.imshow(vis_stim_map, cmap='inferno', vmin=0)
        plt.draw()
        plt.pause(0.1)
        plt.clf()
    plt.ioff()


    # Lick = 0
    lick_map = create_image_from_data(weights[:, trial_length], image_height, image_width, indicies)
    plt.title("Lick")
    plt.imshow(lick_map, cmap='bwr')
    plt.show()

    # Running = 1
    running_map = create_image_from_data(weights[:, trial_length + 1], image_height, image_width, indicies)
    plt.title("Running")
    plt.imshow(running_map, cmap='bwr')
    plt.show()

    # Visualise_Video_Contributions
    mousecam_components = np.load(save_directory + "/Mousecam_SVD_Components.npy")

    mousecam_component = 0
    for predictor in range(trial_length + 2, number_of_predictors):

        figure_1 = plt.figure()
        widefield_axis = figure_1.add_subplot(1,2,1)
        mousecam_axis = figure_1.add_subplot(1,2,2)

        # Create Regression Map
        weight_map = create_image_from_data(weights[:,  predictor], image_height, image_width, indicies)

        # Create Bodycam Image
        bodycam_image = mousecam_components[mousecam_component]
        bodycam_image = np.reshape(bodycam_image, (480, 640))

        plt.title("Bodycam Component: " + str(mousecam_component))

        widefield_axis.imshow(weight_map, cmap='bwr')
        mousecam_axis.imshow(abs(bodycam_image), cmap='jet', vmin=0)
        plt.show()

        mousecam_component += 1




def compare_visual_regressors(base_directory, save_directory, start_window, stop_window):

    # Load Model
    model = joblib.load(save_directory + "/Linear_Model.pkl")

    # Get Coefficients
    weights = model.coef_
    print("Weights", np.shape(weights))

    indicies, image_height, image_width = load_mask(base_directory)
    trial_length = stop_window - start_window

    number_of_predictors = np.shape(weights)[1]

    # Visual_stimuli

    for timepoint in range(trial_length):

        vis_1_stim_map = create_image_from_data(weights[:, timepoint], image_height, image_width, indicies)
        vis_2_stim_map = create_image_from_data(weights[:, timepoint + trial_length], image_height, image_width, indicies)
        difference = np.subtract(vis_1_stim_map, vis_2_stim_map)

        figure_1 = plt.figure()
        vis_1_axis = figure_1.add_subplot(1,3,1)
        vis_2_axis = figure_1.add_subplot(1,3,2)
        diff_axis = figure_1.add_subplot(1,3,3)

        vmin = 0
        vmax = np.max(np.concatenate([vis_1_stim_map, vis_2_stim_map]))


        plt.suptitle("Timepoint: " + str(start_window + timepoint))
        vis_1_axis.imshow(vis_1_stim_map, cmap='jet', vmin=vmin, vmax=vmax)
        vis_2_axis.imshow(vis_2_stim_map, cmap='jet', vmin=vmin, vmax=vmax)

        diff_magnitude = np.max(np.abs(difference))
        diff_axis.imshow(difference, cmap='bwr', vmin=-1*diff_magnitude, vmax=diff_magnitude)

        plt.show()



# Perform Preperations

# Settings
root_directory = "/media/matthew/Seagate Expansion Drive2/Longitudinal_Analysis/NXAK4.1B/"
session_list = ["2021_02_04_Discrimination_Imaging",
                "2021_02_06_Discrimination_Imaging",
                "2021_02_08_Discrimination_Imaging",
                "2021_02_10_Discrimination_Imaging",
                "2021_02_12_Discrimination_Imaging",
                "2021_02_14_Discrimination_Imaging",
                "2021_02_22_Discrimination_Imaging"]


# // Setup Details and File Structure //
base_directory = root_directory + session_list[-1]
start_window = -10
stop_window = 100
number_of_mousecam_components = 30
mousecam_components_to_exclude = [7, 9, 16, 18]

# Check Linear Model Directory Exists
linear_model_directory = base_directory + "/Linear_Model"
Widefield_General_Functions.check_directory(linear_model_directory)


#// Load Widefield Data //

# Load Frame Onsets and Frame Times
vis_1_onsets_file = base_directory + "/Stimuli_Onsets/All_vis_1_frame_indexes.npy"
vis_2_onsets_file = base_directory + "/Stimuli_Onsets/All_vis_2_frame_indexes.npy"
vis_1_onsets = np.load(vis_1_onsets_file)
vis_2_onsets = np.load(vis_2_onsets_file)

# Get Sample
#vis_1_onsets = vis_1_onsets[0:10]
#vis_2_onsets = vis_2_onsets[0:10]

stimuli_onsets = np.concatenate([vis_1_onsets, vis_2_onsets])
print("Stimuli Onsets", np.shape(stimuli_onsets))


# Extract Widefield Data
widefield_file = base_directory + "/Delta_F.h5"
widefield_data_file = tables.open_file(widefield_file, mode='r')
widefield_data = widefield_data_file.root['Data']

# Get All Selected Widefield Frames
selected_widefield_frames = get_selected_widefield_frames(stimuli_onsets, start_window, stop_window)

# Extract These Frames From The Delta_F.h5 File
selected_widefield_data = get_selected_data(selected_widefield_frames, widefield_data)


#/// Create Regressors ///

# Create AI Regressors
ai_regressors = create_ai_recorder_regressors(base_directory, stimuli_onsets, start_window, stop_window)
print("Ai Regressors", np.shape(ai_regressors))

# Create Visual Stimuli Regressors
vis_1_regressors, vis_2_regressors = create_visual_stimuli_regressors(stimuli_onsets, vis_1_onsets, vis_2_onsets, start_window, stop_window, base_directory)
print("Vis 1 Regressors", np.shape(vis_1_regressors))
print("Vis 2 Regressors", np.shape(vis_2_regressors))

# Create Bodycam Regressors
bodycam_regressors = create_bodycam_regressors(base_directory, stimuli_onsets, linear_model_directory, number_of_mousecam_components=number_of_mousecam_components)
#bodycam_regressors = exclude_selected_bodycam_components(bodycam_regressors, mousecam_components_to_exclude)
print("Bodycam regressors shape", np.shape(bodycam_regressors))


# Create Design Matrix
design_matrix = create_design_matrix(ai_regressors, vis_1_regressors, vis_2_regressors, bodycam_regressors)

# Perform Regression
perform_regression(design_matrix, selected_widefield_data, linear_model_directory)

# Explore Regression
explore_regression(base_directory, linear_model_directory, start_window, stop_window)


compare_visual_regressors(base_directory, linear_model_directory, start_window, stop_window)

# Create Design Matrix
# One Model For Each Context
# Running speed
# Lick Trace
# Behaviour Video SVDs