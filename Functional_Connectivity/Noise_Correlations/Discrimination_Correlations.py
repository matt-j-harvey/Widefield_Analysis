import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
import tables
import random
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist, cdist

def sort_matrix(matrix):

    # Cluster Matrix
    Z = ward(pdist(matrix))

    # Get Dendogram Leaf Order
    new_order = leaves_list(Z)

    # Sorted Matrix
    sorted_matrix = matrix[:, new_order][new_order]

    return sorted_matrix


def normalise_activity_matrix(activity_matrix):

    # Subtract Min
    min_vector = np.min(activity_matrix, axis=0)
    activity_matrix = np.subtract(activity_matrix, min_vector)

    # Divide By Max
    max_vector = np.max(activity_matrix, axis=0)
    activity_matrix = np.divide(activity_matrix, max_vector)

    return activity_matrix


def get_activity_tensor(activity_matrix, onset_list, start_window, stop_window):

    activity_tensor = []

    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window
        trial_activity = activity_matrix[trial_start:trial_stop]
        activity_tensor.append(trial_activity)

    activity_tensor = np.array(activity_tensor)
    return activity_tensor




def get_noise_correlations(activity_matrix, condition_1_onsets, condition_2_onsets, start_window, stop_window):

    condition_1_tensor = get_activity_tensor(activity_matrix, condition_1_onsets, start_window, stop_window)
    condition_2_tensor = get_activity_tensor(activity_matrix, condition_2_onsets, start_window, stop_window)

    condition_1_mean = np.mean(condition_1_tensor, axis=0)
    condition_2_mean = np.mean(condition_2_tensor, axis=0)

    condition_1_tensor = np.subtract(condition_1_tensor, condition_1_mean)
    condition_2_tensor = np.subtract(condition_2_tensor, condition_2_mean)

    condition_1_trials, trial_length, number_of_regions = np.shape(condition_1_tensor)
    condition_2_trials, trial_length, number_of_regions = np.shape(condition_2_tensor)

    condition_1_tensor = np.reshape(condition_1_tensor, (condition_1_trials * trial_length, number_of_regions))
    condition_2_tensor = np.reshape(condition_2_tensor, (condition_2_trials * trial_length, number_of_regions))

    combined_tensor = np.vstack([condition_1_tensor, condition_2_tensor])

    comined_correlation_matrix = np.corrcoef(np.transpose(combined_tensor))
    return comined_correlation_matrix


def get_signal_correlations(activity_matrix, condition_1_onsets, condition_2_onsets, start_window, stop_window):

    condition_1_tensor = get_activity_tensor(activity_matrix, condition_1_onsets, start_window, stop_window)
    condition_2_tensor = get_activity_tensor(activity_matrix, condition_2_onsets, start_window, stop_window)

    condition_1_mean = np.mean(condition_1_tensor, axis=0)
    condition_2_mean = np.mean(condition_2_tensor, axis=0)

    combined_tensor = np.vstack([condition_1_mean, condition_2_mean])

    comined_correlation_matrix = np.corrcoef(np.transpose(combined_tensor))
    return comined_correlation_matrix


# Settings
condition_1 = "visual_1_all_onsets.npy"
condition_2 = "visual_2_all_onsets.npy"
start_window = -14
stop_window = 40
trial_length = stop_window - start_window

# Load Neural Data
base_directory = r"C:\Users\matth\Documents\Functional-Connectivity_V2\Parcellated_Delta_F\NXAK7.1B\2021_02_22_Discrimination_Imaging"
activity_matrix = np.load(os.path.join(base_directory, "Cluster_Activity_Matrix.npy"))
print("Delta F Matrix Shape", np.shape(activity_matrix))

# Normalise Activity Matrix
activity_matrix = normalise_activity_matrix(activity_matrix)

# Remove Background Activity
activity_matrix = activity_matrix[:, 1:]

# Load Onsets
vis_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_1))
vis_2_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_2))

# Get Noise Correlations
noise_correlation_matrix = get_noise_correlations(activity_matrix, vis_1_onsets, vis_2_onsets, start_window, stop_window)
signal_correlation_matrix = get_signal_correlations(activity_matrix, vis_1_onsets, vis_2_onsets, start_window, stop_window)

figure_1 = plt.figure()
signal_axis = figure_1.add_subplot(1,2,1)
noise_axis = figure_1.add_subplot(1,2,2)

signal_axis.imshow(signal_correlation_matrix, cmap='bwr', vmin=-1, vmax=1)
noise_axis.imshow(noise_correlation_matrix, cmap='bwr', vmin=-1, vmax=1)

signal_axis.set_title("Signal Correlation")
noise_axis.set_title("Noise Correlation")

plt.show()