import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import tables
import random
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist, cdist
from scipy import stats

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





def get_noise_correlations_single_stimuli(activity_matrix, condition_onsets, start_window, stop_window):

    condition_tensor = get_activity_tensor(activity_matrix, condition_onsets, start_window, stop_window)

    condition_mean = np.mean(condition_tensor, axis=0)

    condition_tensor = np.subtract(condition_tensor, condition_mean)

    condition_trials, trial_length, number_of_regions = np.shape(condition_tensor)

    condition_tensor = np.reshape(condition_tensor, (condition_trials * trial_length, number_of_regions))

    comined_correlation_matrix = np.corrcoef(np.transpose(condition_tensor))

    return comined_correlation_matrix



def get_noise_correlations(activity_matrix, condition_1_onsets, condition_2_onsets, start_window, stop_window):

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



def analyse_noise_correlations(base_directory, visualise=False):

    # Settings
    condition_1 = "visual_1_all_onsets.npy"
    condition_2 = "visual_2_all_onsets.npy"
    start_window = -10
    stop_window = 14
    trial_length = stop_window - start_window

    # Load Neural Data
    #activity_matrix = np.load(os.path.join(base_directory, "Movement_Correction", "Motion_Corrected_Residuals.npy"))
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
    noise_correlation_matrix = get_noise_correlations_single_stimuli(activity_matrix, vis_onsets, start_window, stop_window)
    signal_correlation_matrix = get_noise_correlations_single_stimuli(activity_matrix, vis_onsets, start_window, stop_window)

    if visualise == True:
        figure_1 = plt.figure()
        signal_axis = figure_1.add_subplot(1,2,1)
        noise_axis = figure_1.add_subplot(1,2,2)

        signal_axis.imshow(signal_correlation_matrix, cmap='bwr', vmin=-1, vmax=1)
        noise_axis.imshow(noise_correlation_matrix, cmap='bwr', vmin=-1, vmax=1)

        signal_axis.set_title("Signal Correlation")
        noise_axis.set_title("Noise Correlation")

        plt.show()
    return noise_correlation_matrix



def analyse_noise_correlations_single_stimuli(base_directory, condition, start_window, stop_window, visualise=False):

    # Settings
    trial_length = stop_window - start_window

    # Load Neural Data
    #activity_matrix = np.load(os.path.join(base_directory, "Movement_Correction", "Motion_Corrected_Residuals.npy"))
    activity_matrix = np.load(os.path.join(base_directory, "Cluster_Activity_Matrix.npy"))
    print("Delta F Matrix Shape", np.shape(activity_matrix))

    # Normalise Activity Matrix
    activity_matrix = normalise_activity_matrix(activity_matrix)

    # Remove Background Activity
    activity_matrix = activity_matrix[:, 1:]

    # Load Onsets
    vis_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition))

    # Get Noise Correlations
    noise_correlation_matrix = get_noise_correlations_single_stimuli(activity_matrix, vis_onsets, start_window, stop_window)
    signal_correlation_matrix = get_noise_correlations_single_stimuli(activity_matrix, vis_onsets, start_window, stop_window)

    if visualise == True:
        figure_1 = plt.figure()
        signal_axis = figure_1.add_subplot(1,2,1)
        noise_axis = figure_1.add_subplot(1,2,2)

        signal_axis.imshow(signal_correlation_matrix, cmap='bwr', vmin=-1, vmax=1)
        noise_axis.imshow(noise_correlation_matrix, cmap='bwr', vmin=-1, vmax=1)

        signal_axis.set_title("Signal Correlation")
        noise_axis.set_title("Noise Correlation")

        plt.show()
    return noise_correlation_matrix



def analyse_noise_correlations_over_learning(session_list):
    noise_correlation_matrix_list = []
    for session in session_list:
        noise_correlations = analyse_noise_correlations(session)
        noise_correlation_matrix_list.append(noise_correlations)

    figure_1 = plt.figure()
    rows = 1
    columns = len(noise_correlation_matrix_list)

    for session_index in range(len(noise_correlation_matrix_list)):
        axis = figure_1.add_subplot(rows, columns, session_index + 1)
        axis.imshow(noise_correlation_matrix_list[session_index], cmap='bwr', vmin=-1, vmax=1)

    plt.show()

    final_difference = np.subtract(noise_correlation_matrix_list[2], noise_correlation_matrix_list[-1])
    plt.imshow(final_difference, cmap='bwr', vmin=-1, vmax=1)
    plt.show()





def analyse_noise_correlations_over_learning_seperate_stimuli(session_list, condition_1, condition_2, start_window, stop_window):

    # Create Correlation Matrix List
    difference_matrix_list = []

    for session in session_list:
        condition_1_noise_correlations = analyse_noise_correlations_single_stimuli(session, condition_1, start_window, stop_window)
        condition_2_noise_correlations = analyse_noise_correlations_single_stimuli(session, condition_2, start_window, stop_window)

        """
        plt.imshow(condition_1_noise_correlations)
        plt.show()

        plt.imshow(condition_2_noise_correlations)
        plt.show()
        """

        difference_matrix = np.subtract(condition_1_noise_correlations, condition_2_noise_correlations)
        difference_matrix_list.append(difference_matrix)

    # Plot All Matricies
    """
    figure_1 = plt.figure()
    number_of_sessions = len(session_list)
    rows = 1
    columns = number_of_sessions
    gridspec_1 = GridSpec(rows, columns)

    for session_index in range(number_of_sessions):
        session_axis = figure_1.add_subplot(gridspec_1[0, session_index])
        session_axis.imshow(difference_matrix_list[session_index], cmap='bwr', vmin=-1, vmax=1)
    plt.show()
    """

    return difference_matrix_list[0], difference_matrix_list[-1]

    """
    final_difference = np.subtract(noise_correlation_matrix_list[2], noise_correlation_matrix_list[-1])
    plt.imshow(final_difference, cmap='bwr', vmin=-1, vmax=1)
    plt.show()
    """




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
    plt.show()

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

"""

session_list = [

    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging"]
"""


start_window = -10
stop_window = 14
condition_1 = "visual_1_all_onsets.npy"
condition_2 = "visual_2_all_onsets.npy"

pre_diff_list = []
post_diff_list = []

for group in session_list:
    pre_diff, post_diff = analyse_noise_correlations_over_learning_seperate_stimuli(group, condition_1, condition_2, start_window, stop_window)
    pre_diff_list.append(pre_diff)
    post_diff_list.append(post_diff)

pre_diff_mean = np.mean(pre_diff_list, axis=0)
magnitude = np.max(np.abs(pre_diff_mean))
plt.imshow(pre_diff_mean, cmap='bwr', vmin=-magnitude, vmax=magnitude)
plt.show()

post_diff_mean = np.mean(post_diff_list, axis=0)
magnitude = np.max(np.abs(post_diff_mean))
plt.imshow(post_diff_mean, cmap='bwr', vmin=-magnitude, vmax=magnitude)
plt.show()

t_scores, p_scores = stats.ttest_rel(pre_diff_list, post_diff_list, axis=0)
plt.imshow(t_scores)
plt.show()
