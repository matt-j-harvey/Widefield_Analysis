import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import MSELoss
from torch.optim import SGD
import tables

import random

import math
import numpy as np
import pandas as pd
import os
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


def get_labeled_data_filename(bodycam_files):
    for file in bodycam_files:
        if "resnet" in file:
            return file


def load_deeplabcut_data(base_directory):

    # Get Filename
    bodycam_directory = os.path.join(base_directory, "Bodycam_Analysis")
    bodycam_files = os.listdir(bodycam_directory)
    labeled_data_file = get_labeled_data_filename(bodycam_files)
    labelled_dataframe = pd.read_hdf(os.path.join(bodycam_directory, labeled_data_file))

    upper_limb_right_x = np.array(labelled_dataframe['DLC_resnet101_Matt_Bodycam_AllJul19shuffle1_1000000', 'bodypart1', 'x'])
    upper_limb_right_y = np.array(labelled_dataframe['DLC_resnet101_Matt_Bodycam_AllJul19shuffle1_1000000', 'bodypart1', 'y'])

    lower_limb_right_x = np.array(labelled_dataframe['DLC_resnet101_Matt_Bodycam_AllJul19shuffle1_1000000', 'bodypart2', 'x'])
    lower_limb_right_y = np.array(labelled_dataframe['DLC_resnet101_Matt_Bodycam_AllJul19shuffle1_1000000', 'bodypart2', 'y'])

    lower_limb_left_x = np.array(labelled_dataframe['DLC_resnet101_Matt_Bodycam_AllJul19shuffle1_1000000', 'bodypart3', 'x'])
    lower_limb_left_y = np.array(labelled_dataframe['DLC_resnet101_Matt_Bodycam_AllJul19shuffle1_1000000', 'bodypart3', 'y'])

    upper_limb_left_x = np.array(labelled_dataframe['DLC_resnet101_Matt_Bodycam_AllJul19shuffle1_1000000', 'objectA', 'x'])
    upper_limb_left_y = np.array(labelled_dataframe['DLC_resnet101_Matt_Bodycam_AllJul19shuffle1_1000000', 'objectA', 'y'])

    combined_data = np.vstack([upper_limb_right_x, upper_limb_right_y, lower_limb_right_x, lower_limb_right_y, lower_limb_left_x, lower_limb_left_y, upper_limb_left_x, upper_limb_left_y])
    combined_data = np.transpose(combined_data)

    return combined_data


def match_limbdata_to_widefield_frames(base_directory, limb_data):

    # Load Mousecam Frame Times Dict
    mousecam_frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Mousecam_Frame_Times.npy"), allow_pickle=True)[()]
    widefield_frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]

    mousecam_keys = list(mousecam_frame_times.keys())
    widefield_keys = list(widefield_frame_times.keys())

    matched_data = []
    for widefield_frame_time in widefield_keys:
        closest_mousecam_frame_time = take_closest(mousecam_keys, widefield_frame_time)
        closest_mousecam_frame_index = mousecam_frame_times[closest_mousecam_frame_time]
        matched_data.append(limb_data[closest_mousecam_frame_index])

    matched_data = np.array(matched_data)
    np.save(os.path.join(base_directory, "Bodycam_Analysis", "Matched_Limb_Data.npy"), matched_data)

def take_random_samples(number_of_samples, sample_length, widefield_data, limb_data):

    limb_data_samples = []
    widefield_data_samples = []

    number_of_timepoints = np.shape(widefield_data)[0]

    for sample_index in range(number_of_samples):
        trial_start = random.randint(0, number_of_timepoints-sample_length)
        trial_stop = trial_start + sample_length

        limb_data_samples.append(limb_data[trial_start:trial_stop])
        widefield_data_samples.append(widefield_data[trial_start:trial_stop])

    limb_data_samples = np.array(limb_data)
    widefield_data_samples = np.array(widefield_data_samples)

    return limb_data_samples, widefield_data_samples





# Load Data
base_directory = r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging"

# Load Delta F Data
delta_f_file = os.path.join(base_directory, "Delta_F.h5")
delta_f_file_object = tables.open_file(delta_f_file, "r")
delta_f_data = delta_f_file_object.root.Data

# Load Limb Data
limb_data = load_deeplabcut_data(base_directory)
#match_limbdata_to_widefield_frames(base_directory, limb_data)
limb_data = np.load(os.path.join(base_directory, "Bodycam_Analysis", "Matched_Limb_Data.npy"))

# Get Some Training Data
sequence_length = 60
number_of_samples = 100
limb_data_samples, widefield_data_samples = take_random_samples(number_of_samples, sequence_length, delta_f_data, limb_data)
print("Limb Data Samples", limb_data_samples)
print("Widefield Data Samples", widefield_data_samples)

# Fit Model
ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
model = nn.Transformer()


# define the optimization
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

# enumerate epochs
for epoch in range(100):
    """
