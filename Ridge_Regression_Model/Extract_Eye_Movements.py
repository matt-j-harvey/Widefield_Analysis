import math

import pandas as pd
import tables
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
from sklearn.decomposition import TruncatedSVD, IncrementalPCA
from scipy import ndimage, interpolate
from tqdm import tqdm
import pickle
import pybresenham
import cv2
from ellipse import LsqEllipse
from matplotlib.patches import Ellipse
from bisect import bisect_left


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

def invert_dictionary(dictionary):
    inv_map = {v: k for k, v in dictionary.items()}
    return inv_map

def get_deeplabcut_filename(base_directory):
    file_list = os.listdir(base_directory)
    for file_name in file_list:
        if "DLC_resnet101_matt_eyecam_model" in file_name:
            return file_name

def get_eyecam_filename(base_directory):
    file_list = os.listdir(base_directory)
    for file_name in file_list:
        if "_cam_2" in file_name:
            return file_name



def load_eyecam_video(eyecam_file, sample_size=1000):

    # Open Video File
    cap = cv2.VideoCapture(eyecam_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("frame count", frameCount)

    frame_sample_data = []
    for frame_index in tqdm(range(frameCount)):
        # Set Current Frame Data To Preceeding
        ret, frame_data = cap.read()
        frame_data = frame_data[:, :, 0]
        frame_sample_data.append(frame_data)

    frame_sample_data = np.array(frame_sample_data)
    return frame_sample_data


def get_sustained_changed_in_direction(eye_movement_list, threshold, subsequent_window, subsequent_threshold):

    number_of_timepoints = len(eye_movement_list)
    eye_movement_event_list = np.zeros(number_of_timepoints)
    for timepoint_index in range(number_of_timepoints - subsequent_window):

        event_status = False

        # Check If Above Threshold
        if eye_movement_list[timepoint_index] > threshold:

            # Check If Subsequent Movements Are Not
            if np.max(eye_movement_list[timepoint_index+1:timepoint_index + subsequent_window]) < subsequent_threshold:
                event_status = True

        if event_status == True:
            eye_movement_event_list[timepoint_index] = 1

    return eye_movement_event_list


def extract_pupil_positions(deeplabcut_dataframe, save_directory,
                            blink_probability_threshold,
                            initial_movement_threshold,
                            subsequent_window,
                            subsequent_movement_threshold):


    # Extract Deeplabcut Data
    pupil_left_x = np.array(deeplabcut_dataframe[('DLC_resnet101_matt_eyecam_modelAug9shuffle1_260000', 'pupil_edge_left', 'x')])
    pupil_left_y = deeplabcut_dataframe[('DLC_resnet101_matt_eyecam_modelAug9shuffle1_260000', 'pupil_edge_left', 'y')]
    pupil_left_p = deeplabcut_dataframe[('DLC_resnet101_matt_eyecam_modelAug9shuffle1_260000', 'pupil_edge_left', 'likelihood')]

    pupil_right_x = deeplabcut_dataframe[('DLC_resnet101_matt_eyecam_modelAug9shuffle1_260000', 'pupil_edge_right', 'x')]
    pupil_right_y = deeplabcut_dataframe[('DLC_resnet101_matt_eyecam_modelAug9shuffle1_260000', 'pupil_edge_right', 'y')]
    pupil_right_p = deeplabcut_dataframe[('DLC_resnet101_matt_eyecam_modelAug9shuffle1_260000', 'pupil_edge_right', 'likelihood')]

    pupil_bottom_x = deeplabcut_dataframe[('DLC_resnet101_matt_eyecam_modelAug9shuffle1_260000', 'pupil_edge_bottom', 'x')]
    pupil_bottom_y = deeplabcut_dataframe[('DLC_resnet101_matt_eyecam_modelAug9shuffle1_260000', 'pupil_edge_bottom', 'y')]
    pupil_bottom_p = deeplabcut_dataframe[('DLC_resnet101_matt_eyecam_modelAug9shuffle1_260000', 'pupil_edge_bottom', 'likelihood')]

    pupil_top_x = deeplabcut_dataframe[('DLC_resnet101_matt_eyecam_modelAug9shuffle1_260000', 'pupil_edge_top', 'x')]
    pupil_top_y = deeplabcut_dataframe[('DLC_resnet101_matt_eyecam_modelAug9shuffle1_260000', 'pupil_edge_top', 'y')]
    pupil_top_p = deeplabcut_dataframe[('DLC_resnet101_matt_eyecam_modelAug9shuffle1_260000', 'pupil_edge_top', 'likelihood')]

    # Calculate_Pupil_Stats
    number_of_timepoints = np.shape(pupil_left_x)[0]

    area_list = []
    x_pos_list = []
    y_pos_list = []
    for timepoint in tqdm(range(number_of_timepoints)):
        x = [pupil_left_x[timepoint], pupil_top_x[timepoint], pupil_right_x[timepoint], pupil_bottom_x[timepoint]]
        y = [pupil_left_y[timepoint], pupil_top_y[timepoint], pupil_right_y[timepoint], pupil_bottom_y[timepoint]]

        # append the starting x,y coordinates
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]

        # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
        # is needed in order to force the spline fit to pass through all the input points.
        tck, u = interpolate.splprep([x, y], s=0, per=True)

        # evaluate the spline fits for 1000 evenly spaced distance values
        xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)

        elipse_array = np.array(list(zip(xi, yi)))
        reg = LsqEllipse().fit(elipse_array)
        center, width, height, phi = reg.as_parameters()
        pupil_area = math.pi * height * width
        area_list.append(pupil_area)
        x_pos_list.append(center[0])
        y_pos_list.append(center[1])


    eye_movement_list = []
    for timepoint in range(1, number_of_timepoints):
        a = np.array([x_pos_list[timepoint], y_pos_list[timepoint]])
        b = np.array([x_pos_list[timepoint - 1], y_pos_list[timepoint - 1]])
        dist = np.linalg.norm(a - b)
        eye_movement_list.append(dist)


    # Get Movement Events
    eye_movement_event_list = get_sustained_changed_in_direction(eye_movement_list, initial_movement_threshold, subsequent_window, subsequent_movement_threshold)
    print("EYe movement list", len(eye_movement_list))
    print("eye movement event list", len(eye_movement_event_list))
    # plt.plot(area_list, c='b')
    # plt.show()

    # Blink Criteria - Any pupil maker probability <0.8
    number_of_timepoints = len(eye_movement_list)


    blink_list = []
    for timepoint_index in range(number_of_timepoints):
        timepoint_pupil_probability_list = [
            pupil_top_p[timepoint_index],
            pupil_bottom_p[timepoint_index],
            pupil_left_p[timepoint_index],
            pupil_right_p[timepoint_index]
        ]

        if np.min(timepoint_pupil_probability_list) < blink_probability_threshold:
            start_window = np.max([0, timepoint_index-5])
            stop_window = np.min([number_of_timepoints-1, timepoint_index + 5])
            eye_movement_list[start_window:stop_window] = np.zeros(10)
            eye_movement_event_list[start_window:stop_window] = 0
            blink_list.append(1)
        else:
            blink_list.append(0)

    plt.plot(eye_movement_list, alpha=0.5, c='b')
    plt.plot(eye_movement_event_list, alpha=0.5, c='g')
    plt.show()

    # Save These
    np.save(os.path.join(save_directory, "eye_movements.npy"), eye_movement_list)
    np.save(os.path.join(save_directory, "blinks.npy"), blink_list)
    np.save(os.path.join(save_directory, "eye_movement_events"), eye_movement_event_list)
    np.save(os.path.join(save_directory, "pupil_top_p.npy"), pupil_top_p)
    np.save(os.path.join(save_directory, "pupil_bottom_p.npy"), pupil_bottom_p)
    np.save(os.path.join(save_directory, "pupil_left_p.npy"), pupil_left_p)
    np.save(os.path.join(save_directory, "pupil_right_p.npy"), pupil_right_p)

    return blink_list, eye_movement_event_list


def match_events_to_widefield_frames(base_directory, blink_events_list, eye_movement_events_list):

    # Load Widefield To Mousecam Frame Dict
    widefield_frame_time_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    widefield_frame_times = list(widefield_frame_time_dict.keys())
    print("Widefield Frame Times", widefield_frame_times)

    mousecam_frame_time_dict =  np.load(os.path.join(base_directory, "Stimuli_Onsets", "Mousecam_Frame_Times.npy"), allow_pickle=True)[()]
    mousecam_frame_time_dict = invert_dictionary(mousecam_frame_time_dict)
    print("Mousecam Frame Times", list(mousecam_frame_time_dict.keys())[0:10])

    print("Blink event length", len(blink_events_list))
    print("Eye movement event length", len(eye_movement_events_list))

    number_of_widefield_frames = len(widefield_frame_times)
    number_of_mousecam_frames = len(blink_events_list)

    print("Number of wideifeld frames", number_of_widefield_frames)
    matched_blink_events = np.zeros(number_of_widefield_frames)
    matched_eye_movement_events = np.zeros(number_of_widefield_frames)

    for mousecam_frame_index in range(number_of_mousecam_frames):

        if blink_events_list[mousecam_frame_index] == 1:
            mousecam_frame_time = mousecam_frame_time_dict[mousecam_frame_index]
            nearest_widefield_frame_time = take_closest(myList=widefield_frame_times, myNumber=mousecam_frame_time)
            nearest_widefield_frame_index = widefield_frame_times.index(nearest_widefield_frame_time)
            matched_blink_events[nearest_widefield_frame_index] = 1

        if eye_movement_events_list[mousecam_frame_index] == 1:
            mousecam_frame_time = mousecam_frame_time_dict[mousecam_frame_index]
            nearest_widefield_frame_time = take_closest(myList=widefield_frame_times, myNumber=mousecam_frame_time)
            nearest_widefield_frame_index = widefield_frame_times.index(nearest_widefield_frame_time)
            matched_eye_movement_events[nearest_widefield_frame_index] = 1

    # Save These
    np.save(os.path.join(base_directory, "Eyecam_Analysis", "Matched_Eye_Movement_Events.npy"), matched_eye_movement_events)
    np.save(os.path.join(base_directory, "Eyecam_Analysis", "Matched_Blink_Events.npy"), matched_blink_events)



def extract_eye_movements(base_directory,
                          blink_probability_threshold=0.8,
                          initial_movement_threshold=10,
                          subsequent_window=10,
                          subsequent_movement_threshold=5):

    deeplabcut_filename = get_deeplabcut_filename(base_directory)
    analysed_eye_file = os.path.join(base_directory, deeplabcut_filename)
    save_directory = os.path.join(base_directory, "Eyecam_Analysis")

    # Check Save Directory Exists
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Open Deeplabcut File
    deeplabcut_dataframe = pd.read_hdf(analysed_eye_file)

    # Extract Pupil Positions
    blink_event_list, eye_movement_event_list = extract_pupil_positions(deeplabcut_dataframe, save_directory,  blink_probability_threshold, initial_movement_threshold, subsequent_window, subsequent_movement_threshold)

    # Match Events To Widefield Frames
    match_events_to_widefield_frames(base_directory, blink_event_list, eye_movement_event_list)


base_directory = r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_17_Switching_Imaging"
extract_eye_movements(base_directory)
"""
print("pupil left")
print(pupil_left_data)
# Extract Data
x_coords_list = []
y_coords_list = []
probability_list = []
for limb_index in range(number_of_limbs):
    limb_x_coords = deeplabcut_dataframe[columns[limb_index * 3 + 0]]
    limb_y_coords = deeplabcut_dataframe[columns[limb_index * 3 + 1]]
    limb_likelihood = deeplabcut_dataframe[columns[limb_index * 3 + 2]]

    x_coords_list.append(limb_x_coords)
    y_coords_list.append(limb_y_coords)
    probability_list.append(limb_likelihood)
"""