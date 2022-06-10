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

        if onset > 1500:
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



def split_sessions_By_d_prime(session_list, intermediate_threshold, post_threshold):

    pre_learning_sessions = []
    intermediate_learning_sessions = []
    post_learning_sessions = []

    # Iterate Throug Sessions
    for session in session_list:

        # Load D Prime
        behavioural_dictionary = np.load(os.path.join(session, "Behavioural_Measures", "Performance_Dictionary.npy"), allow_pickle=True)[()]
        d_prime = behavioural_dictionary["visual_d_prime"]

        if d_prime >= post_threshold:
            post_learning_sessions.append(session)

        if d_prime < post_threshold and d_prime >= intermediate_threshold:
            intermediate_learning_sessions.append(session)

        if d_prime < intermediate_threshold:
            pre_learning_sessions.append(session)

    return pre_learning_sessions, intermediate_learning_sessions, post_learning_sessions


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







def get_noise_correlations_for_session_list(session_list, condition_1, condition_2, start_window, stop_window):

    # Create Correlation Matrix List
    condition_1_noise_correlation_matrix_list = []
    condition_2_noise_correlation_matrix_list = []

    for session in session_list:
        condition_1_noise_correlations = analyse_noise_correlations_single_stimuli(session, condition_1, start_window, stop_window)
        condition_2_noise_correlations = analyse_noise_correlations_single_stimuli(session, condition_2, start_window, stop_window)

        condition_1_noise_correlation_matrix_list.append(condition_1_noise_correlations)
        condition_2_noise_correlation_matrix_list.append(condition_2_noise_correlations)

    condition_1_noise_correlation_matrix_list = np.array(condition_1_noise_correlation_matrix_list)
    condition_2_noise_correlation_matrix_list = np.array(condition_2_noise_correlation_matrix_list)

    mean_condition_1_matrix = np.mean(condition_1_noise_correlation_matrix_list, axis=0)
    mean_condition_2_matrix = np.mean(condition_2_noise_correlation_matrix_list, axis=0)

    print("Mean condition 1 matrix", np.shape(mean_condition_1_matrix))
    return mean_condition_1_matrix, mean_condition_2_matrix



def analyse_noise_correlations_over_learning_seperate_stimuli(session_list, condition_1, condition_2, start_window, stop_window):

    # Split Sessions By Performance
    intermediate_threshold = 1
    post_threshold = 2
    pre_learning_sessions, intermediate_learning_sessions, post_learning_sessions = split_sessions_By_d_prime(session_list, intermediate_threshold, post_threshold)


    # Get Correlation Matricies For Each Performance Class
    pre_learning_vis_1_noise_correlations, pre_learning_vis_2_noise_correlations = get_noise_correlations_for_session_list(pre_learning_sessions, condition_1, condition_2, start_window, stop_window)
    pre_learning_difference = np.subtract(pre_learning_vis_1_noise_correlations, pre_learning_vis_2_noise_correlations)

    intermediate_vis_1_noise_correlations, intermediate_vis_2_noise_correlations = get_noise_correlations_for_session_list(intermediate_learning_sessions, condition_1, condition_2, start_window, stop_window)
    intermediate_learning_difference = np.subtract(intermediate_vis_1_noise_correlations, intermediate_vis_2_noise_correlations)

    post_learning_vis_1_noise_correlations, post_learning_vis_2_noise_correlations = get_noise_correlations_for_session_list(post_learning_sessions, condition_1, condition_2, start_window, stop_window)
    post_learning_difference = np.subtract(post_learning_vis_1_noise_correlations, post_learning_vis_2_noise_correlations)


    # Plot All Matricies
    figure_1 = plt.figure()
    number_of_sessions = len(session_list)
    rows = 3
    columns = 3
    gridspec_1 = GridSpec(rows, columns)


    # Create Axes
    pre_learning_vis_1_axis = figure_1.add_subplot(gridspec_1[0, 0])
    pre_learning_vis_2_axis = figure_1.add_subplot(gridspec_1[1, 0])
    pre_learning_diff_axis = figure_1.add_subplot(gridspec_1[2, 0])

    intermediate_vis_1_axis = figure_1.add_subplot(gridspec_1[0, 1])
    intermediate_vis_2_axis = figure_1.add_subplot(gridspec_1[1, 1])
    intermediate_learning_diff_axis = figure_1.add_subplot(gridspec_1[2, 1])

    post_learning_vis_1_axis = figure_1.add_subplot(gridspec_1[0, 2])
    post_learning_vis_2_axis = figure_1.add_subplot(gridspec_1[1, 2])
    post_learning_diff_axis = figure_1.add_subplot(gridspec_1[2, 2])


    # Plot Items
    pre_learning_vis_1_axis.imshow(pre_learning_vis_1_noise_correlations, cmap='bwr', vmin=-1, vmax=1)
    pre_learning_vis_2_axis.imshow(pre_learning_vis_2_noise_correlations, cmap='bwr', vmin=-1, vmax=1)
    pre_learning_diff_axis.imshow(pre_learning_difference, cmap='bwr', vmin=-1, vmax=1)

    intermediate_vis_1_axis.imshow(intermediate_vis_1_noise_correlations, cmap='bwr', vmin=-1, vmax=1)
    intermediate_vis_2_axis.imshow(intermediate_vis_2_noise_correlations, cmap='bwr', vmin=-1, vmax=1)
    intermediate_learning_diff_axis.imshow(intermediate_learning_difference, cmap='bwr', vmin=-1, vmax=1)

    post_learning_vis_1_axis.imshow(post_learning_vis_1_noise_correlations, cmap='bwr', vmin=-1, vmax=1)
    post_learning_vis_2_axis.imshow(post_learning_vis_2_noise_correlations, cmap='bwr', vmin=-1, vmax=1)
    post_learning_diff_axis.imshow(post_learning_difference, cmap='bwr', vmin=-1, vmax=1)

    plt.show()

    np.save("/media/matthew/Expansion/Widefield_Analysis/Discrimination_Analysis/Noise_Correlation_Changes/Post_Learning_Changes/Pre_Learning_Vis_1.npy", pre_learning_vis_1_noise_correlations)
    np.save("/media/matthew/Expansion/Widefield_Analysis/Discrimination_Analysis/Noise_Correlation_Changes/Post_Learning_Changes/Intermediate_Learning_Vis_1.npy", intermediate_vis_1_noise_correlations)
    np.save("/media/matthew/Expansion/Widefield_Analysis/Discrimination_Analysis/Noise_Correlation_Changes/Post_Learning_Changes/Post_Learning_Vis_1.npy", post_learning_vis_1_noise_correlations)

    np.save("/media/matthew/Expansion/Widefield_Analysis/Discrimination_Analysis/Noise_Correlation_Changes/Post_Learning_Changes/Post_learning_Differences.npy", post_learning_difference)

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



session_list = [

    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging"]




start_window = -10
stop_window = 14
condition_1 = "visual_1_all_onsets.npy"
condition_2 = "visual_2_all_onsets.npy"
analyse_noise_correlations_over_learning_seperate_stimuli(session_list, condition_1, condition_2, start_window, stop_window)