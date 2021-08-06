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
from tensorly.decomposition import parafac, CP, non_negative_parafac



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

def check_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


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


def get_nearest_frame(stimuli_onsets, frame_onsets):


    frame_times = frame_onsets.keys()
    nearest_frames = []
    window_size = 50

    for onset in stimuli_onsets:
        smallest_distance = 1000
        closest_frame = None

        window_start = onset - window_size
        window_stop  = onset + window_size

        for timepoint in range(window_start, window_stop):

            #There is a frame at this time
            if timepoint in frame_times:
                distance = abs(onset - timepoint)

                if distance < smallest_distance:
                    smallest_distance = distance
                    closest_frame = frame_onsets[timepoint]

        if closest_frame != None:
            if closest_frame > 11:
                nearest_frames.append(closest_frame)

    nearest_frames = np.array(nearest_frames)
    return nearest_frames


def get_selected_widefield_frames(onsets, start_window, stop_window):

    selected_fames = []

    for onset in onsets:
        trial_start = onset + start_window
        trial_stop = onset + stop_window
        trial_frames = list(range(trial_start, trial_stop))
        selected_fames.append(trial_frames)

    return selected_fames



def get_selected_widefield_data(selected_widefield_onsets, widefield_data):

    selected_widefield_data = []

    for trial in selected_widefield_onsets:
        trial_data = []
        for frame in trial:
            frame_data = widefield_data[frame]
            trial_data.append(frame_data)

        selected_widefield_data.append(trial_data)

    selected_widefield_data = np.array(selected_widefield_data)
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


base_directory = r"/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Switching_Analysis/Selected_sessions/NXAK10.1A/2021_05_20_Switching_Imaging/"

# Save Factors
save_directory = base_directory + "/Tensor_Decomposition"
if not os.path.exists(save_directory):
    os.mkdir(save_directory)


# Save Factors
save_directory = base_directory + "/Tensor_Decomposition"
if not os.path.exists(save_directory):
    os.mkdir(save_directory)



# Extract Widefield Data
widefield_file = base_directory + "/Delta_F.h5"
widefield_data_file = tables.open_file(widefield_file, mode='r')
widefield_data = widefield_data_file.root['Data']


# Get Onsets Of All Visual Stimuli
ai_file = get_ai_filename(base_directory)
ai_data = load_ai_recorder_file(base_directory + ai_file)

stimuli_dictionary = create_stimuli_dictionary()
vis_1_trace = ai_data[stimuli_dictionary["Visual 1"]]
vis_2_trace = ai_data[stimuli_dictionary["Visual 2"]]
combined_visual_trace = np.array([vis_1_trace, vis_2_trace])
combined_visual_trace = np.max(combined_visual_trace, axis=0)

all_visual_onsets = get_step_onsets(combined_visual_trace)
frame_onsets = np.load(base_directory + "/Stimuli_Onsets/Frame_Times.npy", allow_pickle=True)
frame_onsets = frame_onsets[()]

all_visual_frame_onsets = get_nearest_frame(all_visual_onsets, frame_onsets)
#all_visual_frame_onsets = all_visual_frame_onsets[0:10]
print("All visual frame onsets", all_visual_frame_onsets)

# Extract Widefield Data
start_window = -10
stop_window = 100
number_of_factors = 30

# Get All Selected Widefield Frames
selected_widefield_frames = get_selected_widefield_frames(all_visual_frame_onsets, start_window, stop_window)


# Extract These Frames From THe Delta_F.h5 File
print("Getting Selected Data")
selected_widefield_data = get_selected_widefield_data(selected_widefield_frames, widefield_data)
print("Widefield Data Shape", np.shape(selected_widefield_data))

# Perform Tensor Decomposition
weights, factors = non_negative_parafac(selected_widefield_data, rank=number_of_factors, init='random', verbose=1, n_iter_max=100)
print("Tensor shoape", np.shape(factors))
trial_loadings = factors[0]
time_loadings  = factors[1]
pixel_loadings = factors[2]


np.save(save_directory + "/Trial_Loadings.npy", trial_loadings)
np.save(save_directory + "/Time_Loadings.npy", time_loadings)
np.save(save_directory + "/Pixel_Loadings.npy", pixel_loadings)


# Plot Factors
trial_loadings = np.load(save_directory + "/Trial_Loadings.npy", allow_pickle=True)
time_loadings = np.load(save_directory + "/Time_Loadings.npy", allow_pickle=True)
pixel_loadings = np.load(save_directory + "/Pixel_Loadings.npy", allow_pickle=True)

indicies, image_height, image_width = load_mask(base_directory)


rows = 3
columns = 1

#Plot Directory
plot_directory = save_directory + "/Factor_Plots"
check_directory(plot_directory)


for factor in range(number_of_factors):
    print("Plotting Factor: ", factor)
    figure_1 = plt.figure(figsize=(20,10))
    figure_1.suptitle('Factor: ' + str(factor), fontsize=16)

    pixel_axis = figure_1.add_subplot(rows, columns, 1)
    time_axis  = figure_1.add_subplot(rows, columns, 2)
    trial_axis = figure_1.add_subplot(rows, columns, 3)

    pixel_axis.set_title("Pixel Loadings")
    time_axis.set_title("Time Loadings")
    trial_axis.set_title("Trial Loadings")

    factor_data = pixel_loadings[:, factor]
    time_data = time_loadings[:, factor]
    trial_data = trial_loadings[:, factor]

    factor_image = create_image_from_data(factor_data, image_height, image_width, indicies)
    #factor_range = np.max(np.abs(factor_image))
    pixel_axis.axis('off')

    pixel_axis.imshow(factor_image, cmap='inferno', vmin=0)
    time_axis.plot(time_data)
    trial_axis.plot(trial_data)


    plt.savefig(plot_directory + "/" + str(factor).zfill(3) + ".png")
    plt.close()



