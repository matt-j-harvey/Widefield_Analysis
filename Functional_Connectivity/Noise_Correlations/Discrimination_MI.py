import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
import tables
import random
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist, cdist
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

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

        if trial_stop < np.shape(activity_matrix)[0]:
            trial_activity = activity_matrix[trial_start:trial_stop]
            activity_tensor.append(trial_activity)

    activity_tensor = np.array(activity_tensor)
    return activity_tensor




def get_mutual_info(activity_matrix, condition_1_onsets, condition_2_onsets, start_window, stop_window):

    condition_1_tensor = get_activity_tensor(activity_matrix, condition_1_onsets, start_window, stop_window)
    condition_2_tensor = get_activity_tensor(activity_matrix, condition_2_onsets, start_window, stop_window)
    print("Condition 1 tensor", np.shape(condition_1_tensor))
    print("Condition 2 tensor", np.shape(condition_2_tensor))

    condition_1_mean = np.mean(condition_1_tensor, axis=0)
    condition_2_mean = np.mean(condition_2_tensor, axis=0)

    condition_1_tensor = np.subtract(condition_1_tensor, condition_1_mean)
    condition_2_tensor = np.subtract(condition_2_tensor, condition_2_mean)

    condition_1_trials, trial_length, number_of_regions = np.shape(condition_1_tensor)
    condition_2_trials, trial_length, number_of_regions = np.shape(condition_2_tensor)

    condition_1_tensor = np.reshape(condition_1_tensor, (condition_1_trials * trial_length, number_of_regions))
    condition_2_tensor = np.reshape(condition_2_tensor, (condition_2_trials * trial_length, number_of_regions))

    combined_tensor = np.vstack([condition_1_tensor, condition_2_tensor])
    print("combined Tensor Shape", np.shape(combined_tensor))

    combined_tensor = np.nan_to_num(combined_tensor)

    mutual_info_matrix = np.zeros((number_of_regions, number_of_regions))
    for region_1_index in range(number_of_regions):
        region_1_trace = combined_tensor[:, region_1_index]

        for region_2_index in range(region_1_index, number_of_regions):
            region_2_trace = combined_tensor[:, region_2_index]

            mutual_info = mutual_info_regression(X=region_1_trace.reshape(-1, 1) , y=region_2_trace)
            mutual_info_matrix[region_1_index, region_2_index] = mutual_info[0]
            mutual_info_matrix[region_2_index, region_1_index] = mutual_info[0]

    return mutual_info_matrix


def get_signal_correlations(activity_matrix, condition_1_onsets, condition_2_onsets, start_window, stop_window):

    condition_1_tensor = get_activity_tensor(activity_matrix, condition_1_onsets, start_window, stop_window)
    condition_2_tensor = get_activity_tensor(activity_matrix, condition_2_onsets, start_window, stop_window)

    condition_1_mean = np.mean(condition_1_tensor, axis=0)
    condition_2_mean = np.mean(condition_2_tensor, axis=0)

    combined_tensor = np.vstack([condition_1_mean, condition_2_mean])

    comined_correlation_matrix = np.corrcoef(np.transpose(combined_tensor))
    return comined_correlation_matrix


def analyse_mutual_info(base_directory, visualise=False):

    # Settings
    condition_1 = "visual_1_all_onsets.npy"
    condition_2 = "visual_2_all_onsets.npy"
    start_window = -10
    stop_window = 14
    trial_length = stop_window - start_window

    # Load Neural Data
    activity_matrix = np.load(os.path.join(base_directory, "Movement_Correction", "Motion_Corrected_Residuals.npy"))
    print("Delta F Matrix Shape", np.shape(activity_matrix))

    # Normalise Activity Matrix
    activity_matrix = normalise_activity_matrix(activity_matrix)

    # Remove Background Activity
    activity_matrix = activity_matrix[:, 1:]

    # Load Onsets
    vis_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_1))
    vis_2_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_2))

    # Get Noise Correlations
    noise_mi_matrix = get_mutual_info(activity_matrix, vis_1_onsets, vis_2_onsets, start_window, stop_window)
    np.fill_diagonal(noise_mi_matrix, 0)
    if visualise == True:
        figure_1 = plt.figure()

        noise_axis = figure_1.add_subplot(1,1,1)
        noise_axis.imshow(noise_mi_matrix, cmap='jet')
        noise_axis.set_title("Noise MI")

        plt.show()
    return noise_mi_matrix



def analyse_mi_over_learning(session_list):

    mi_matrix_list = []
    for session in session_list:
        mi = analyse_mutual_info(session)
        mi_matrix_list.append(mi)

    figure_1 = plt.figure()
    rows = 1
    columns = len(mi_matrix_list)

    for session_index in range(len(mi_matrix_list)):
        axis = figure_1.add_subplot(rows, columns, session_index + 1)
        axis.imshow(mi_matrix_list[session_index], cmap='jet')

    plt.show()

    final_difference = np.subtract(mi_matrix_list[-1], mi_matrix_list[1])
    plt.imshow(final_difference, cmap='jet')
    plt.show()


def comapre_pre_and_post(meta_session_list):

    number_of_sessions = len(meta_session_list)

    rows = 1
    columns = number_of_sessions
    figure_1 = plt.figure()

    final_correlation_list = []
    start_correlation_list = []

    for session_index in range(number_of_sessions):
        session_list = meta_session_list[session_index]
        noise_correlation_matrix_list = []
        for session in session_list:
            noise_correlations = analyse_noise_correlations(session)
            noise_correlation_matrix_list.append(noise_correlations)

        axis = figure_1.add_subplot(rows, columns, session_index + 1)
        final_difference = np.subtract(noise_correlation_matrix_list[-1], noise_correlation_matrix_list[1])
        axis.imshow(final_difference, cmap='bwr', vmin=-1, vmax=1)

        final_correlation_list.append(noise_correlation_matrix_list[-1])
        start_correlation_list.append(noise_correlation_matrix_list[1])

    t_values, p_values = stats.ttest_rel(final_correlation_list, start_correlation_list, axis=0)
    print(np.shape(t_values))

    thresholded_t_values = np.where(p_values < 0.05, t_values, 0)
    plt.show()

    plt.imshow(thresholded_t_values, cmap='bwr', vmin=-2, vmax=2)
    plt.show()

    sorted_thresholded_t_values = sort_matrix(thresholded_t_values)
    plt.imshow(sorted_thresholded_t_values, cmap='bwr', vmin=-2, vmax=2)
    plt.show()

    t_values = np.nan_to_num(t_values)
    sorted_t_values = sort_matrix(t_values)
    sorted_t_values = np.abs(sorted_t_values)
    plt.imshow(sorted_t_values)
    plt.show()




session_list = [


    ["/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_01_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_03_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_05_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_07_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_09_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_11_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_13_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_15_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_17_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_19_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_22_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_24_Discrimination_Imaging"],

    ["/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging"],

    ["/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_04_29_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_01_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_03_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_05_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_07_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_09_Discrimination_Imaging"],

    ["/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_09_25_Discrimination_Imaging",
     "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_09_29_Discrimination_Imaging",
     "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A//2021_10_01_Discrimination_Imaging",
     "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_10_03_Discrimination_Imaging",
     "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_10_05_Discrimination_Imaging",
     "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_10_07_Discrimination_Imaging",
     "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging"],

    ["/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_15_Discrimination_Imaging",
     "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_16_Discrimination_Imaging",
     "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_17_Discrimination_Imaging",
     "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_19_Discrimination_Imaging",
     "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_23_Discrimination_Imaging",
     "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_25_Discrimination_Imaging",
     "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_14_Discrimination_Imaging"],

    ["/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging",
     "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_16_Discrimination_Imaging",
     "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_17_Discrimination_Imaging",
     "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_19_Discrimination_Imaging",
     "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_21_Discrimination_Imaging",
     "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_24_Discrimination_Imaging"],
]


analyse_mi_over_learning(session_list[-1])
