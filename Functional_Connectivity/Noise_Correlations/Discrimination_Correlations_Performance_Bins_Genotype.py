import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import tables
import random
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist, cdist
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

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

        print(session, "d prime: ", d_prime)

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







def get_noise_correlations_for_session_list(session_list, condition, start_window, stop_window):

    # Create Correlation Matrix List
    noise_correlation_matrix_list = []

    for session in session_list:
        noise_correlations = analyse_noise_correlations_single_stimuli(session, condition, start_window, stop_window)
        noise_correlation_matrix_list.append(noise_correlations)

    noise_correlation_matrix_list = np.array(noise_correlation_matrix_list)
    mean_matrix = np.mean(noise_correlation_matrix_list, axis=0)

    return mean_matrix


def list_noise_correlations_for_session_list(session_list, condition, start_window, stop_window):

    # Create Correlation Matrix List
    noise_correlation_matrix_list = []

    for session in session_list:
        noise_correlations = analyse_noise_correlations_single_stimuli(session, condition, start_window, stop_window)
        noise_correlation_matrix_list.append(noise_correlations)

    return noise_correlation_matrix_list



def analyse_noise_correlations_over_learning_seperate_stimuli(control_session_list, mutant_session_list, condition, start_window, stop_window):

    # Split Sessions By Performance
    intermediate_threshold = 1
    post_threshold = 2
    control_pre_learning_sessions,  control_intermediate_learning_sessions,  control_post_learning_sessions = split_sessions_By_d_prime(control_session_list, intermediate_threshold, post_threshold)
    mutant_pre_learning_sessions,  mutant_intermediate_learning_sessions,  mutant_post_learning_sessions = split_sessions_By_d_prime(mutant_session_list, intermediate_threshold, post_threshold)

    # Get Correlation Matricies For Each Performance Class
    control_pre_learning_noise_correlations             = get_noise_correlations_for_session_list(control_pre_learning_sessions, condition, start_window, stop_window)
    control_intermediate_learning_noise_correlations    = get_noise_correlations_for_session_list(control_intermediate_learning_sessions, condition, start_window, stop_window)
    control_post_learning_noise_correlations            = get_noise_correlations_for_session_list(control_post_learning_sessions, condition, start_window, stop_window)

    mutant_pre_learning_noise_correlations              = get_noise_correlations_for_session_list(mutant_pre_learning_sessions, condition, start_window, stop_window)
    mutant_intermediate_noise_correlations              = get_noise_correlations_for_session_list(mutant_intermediate_learning_sessions, condition, start_window, stop_window)
    mutant_post_learning_correlations                   = get_noise_correlations_for_session_list(mutant_post_learning_sessions, condition, start_window, stop_window)

    diff_pre = np.subtract(control_pre_learning_noise_correlations, mutant_pre_learning_noise_correlations)
    diff_intermediate = np.subtract(control_intermediate_learning_noise_correlations, mutant_intermediate_noise_correlations)
    diff_post = np.subtract(control_post_learning_noise_correlations, mutant_post_learning_correlations)


    # Plot All Matricies
    figure_1 = plt.figure()
    rows = 3
    columns = 3
    gridspec_1 = GridSpec(rows, columns)


    # Create Axes
    control_pre_axis            = figure_1.add_subplot(gridspec_1[0, 0])
    control_intermediate_axis   = figure_1.add_subplot(gridspec_1[0, 1])
    control_post_axis           = figure_1.add_subplot(gridspec_1[0, 2])

    mutant_pre_axis             = figure_1.add_subplot(gridspec_1[1, 0])
    mutant_intermediate_axis    = figure_1.add_subplot(gridspec_1[1, 1])
    mutant_post_axis            = figure_1.add_subplot(gridspec_1[1, 2])

    diff_pre_axis = figure_1.add_subplot(gridspec_1[2, 0])
    diff_intermediate_axis = figure_1.add_subplot(gridspec_1[2, 1])
    diff_post_axis = figure_1.add_subplot(gridspec_1[2, 2])


    # Plot Items
    """
    control_pre_axis.imshow(control_pre_learning_noise_correlations, cmap='bwr', vmin=-1, vmax=1)
    control_intermediate_axis.imshow(control_intermediate_learning_noise_correlations, cmap='bwr', vmin=-1, vmax=1)
    control_post_axis.imshow(control_post_learning_noise_correlations, cmap='bwr', vmin=-1, vmax=1)

    mutant_pre_axis.imshow(mutant_pre_learning_noise_correlations, cmap='bwr', vmin=-1, vmax=1)
    mutant_intermediate_axis.imshow(mutant_intermediate_noise_correlations, cmap='bwr', vmin=-1, vmax=1)
    mutant_post_axis.imshow(mutant_post_learning_correlations, cmap='bwr', vmin=-1, vmax=1)
    """

    vmin=0
    vmax=1
    cmap='jet'
    control_pre_axis.imshow(control_pre_learning_noise_correlations, cmap=cmap, vmin=vmin, vmax=vmax)
    control_intermediate_axis.imshow(control_intermediate_learning_noise_correlations, cmap=cmap, vmin=vmin, vmax=vmax)
    control_post_axis.imshow(control_post_learning_noise_correlations, cmap=cmap, vmin=vmin, vmax=vmax)

    mutant_pre_axis.imshow(mutant_pre_learning_noise_correlations, cmap=cmap, vmin=vmin, vmax=vmax)
    mutant_intermediate_axis.imshow(mutant_intermediate_noise_correlations, cmap=cmap, vmin=vmin, vmax=vmax)
    mutant_post_axis.imshow(mutant_post_learning_correlations, cmap=cmap, vmin=vmin, vmax=vmax)


    diff_pre_axis.imshow(diff_pre, cmap='bwr', vmin=-1, vmax=1)
    diff_intermediate_axis.imshow(diff_intermediate, cmap='bwr', vmin=-1, vmax=1)
    diff_post_axis.imshow(diff_post, cmap='bwr', vmin=-1, vmax=1)


    plt.show()





def test_differences_for_significance(control_session_list, mutant_session_list, condition, start_window, stop_window):

    # Split Sessions By Performance
    intermediate_threshold = 1
    post_threshold = 2
    control_pre_learning_sessions, control_intermediate_learning_sessions, control_post_learning_sessions = split_sessions_By_d_prime(control_session_list, intermediate_threshold, post_threshold)
    mutant_pre_learning_sessions, mutant_intermediate_learning_sessions, mutant_post_learning_sessions = split_sessions_By_d_prime(mutant_session_list, intermediate_threshold, post_threshold)

    print("Control Post Learning Sessions", control_post_learning_sessions)
    print("Mutant Post Learning Sessions", mutant_post_learning_sessions)

    # Get Correlation Matricies For Each Performance Class
    control_pre_learning_noise_correlations = list_noise_correlations_for_session_list(control_pre_learning_sessions, condition, start_window, stop_window)
    control_intermediate_learning_noise_correlations = list_noise_correlations_for_session_list(control_intermediate_learning_sessions, condition, start_window, stop_window)
    control_post_learning_noise_correlations = list_noise_correlations_for_session_list(control_post_learning_sessions, condition, start_window, stop_window)

    mutant_pre_learning_noise_correlations = list_noise_correlations_for_session_list(mutant_pre_learning_sessions, condition, start_window, stop_window)
    mutant_intermediate_noise_correlations = list_noise_correlations_for_session_list(mutant_intermediate_learning_sessions, condition, start_window, stop_window)
    mutant_post_learning_correlations = list_noise_correlations_for_session_list(mutant_post_learning_sessions, condition, start_window, stop_window)

    # Test For Significance
    pre_t, pre_p = stats.ttest_ind(control_pre_learning_noise_correlations, mutant_pre_learning_noise_correlations, axis=0)
    int_t, int_p = stats.ttest_ind(control_intermediate_learning_noise_correlations, mutant_intermediate_noise_correlations, axis=0)
    post_t, post_p = stats.ttest_ind(control_post_learning_noise_correlations, mutant_post_learning_correlations, axis=0)

    # Perform FDR Correction
    corrected_pre_p = fdrcorrection(np.ndarray.flatten(pre_p))[0]
    corrected_int_p = fdrcorrection(np.ndarray.flatten(int_p))[0]
    corrected_post_p = fdrcorrection(np.ndarray.flatten(post_p))[0]

    corrected_pre_p = np.reshape(corrected_pre_p, np.shape(pre_p))
    corrected_int_p = np.reshape(corrected_int_p, np.shape(int_p))
    corrected_post_p = np.reshape(corrected_post_p, np.shape(post_p))

    # Threshold
    """
    thresholded_pre_t = np.where(pre_p < 0.05, pre_t, 0)
    thresholded_int_t = np.where(int_p < 0.05, int_t, 0)
    thresholded_post_t = np.where(post_p < 0.05, post_t, 0)
    """

    thresholded_pre_t = np.where(corrected_pre_p == True, pre_t, 0)
    thresholded_int_t = np.where(corrected_int_p == True, int_t, 0)
    thresholded_post_t = np.where(corrected_post_p == True, post_t, 0)


    # Plot Matricies
    figure_1 = plt.figure()
    rows = 1
    columns = 3
    gridspec_1 = GridSpec(rows, columns)

    # Create Axes
    diff_pre_axis = figure_1.add_subplot(gridspec_1[0, 0])
    diff_intermediate_axis = figure_1.add_subplot(gridspec_1[0, 1])
    diff_post_axis = figure_1.add_subplot(gridspec_1[0, 2])

    magnitude = 3
    diff_pre_axis.imshow(thresholded_pre_t, cmap='bwr',          vmin=-magnitude, vmax=magnitude)
    diff_intermediate_axis.imshow(thresholded_int_t, cmap='bwr', vmin=-magnitude, vmax=magnitude)
    diff_post_axis.imshow(thresholded_post_t, cmap='bwr',        vmin=-magnitude, vmax=magnitude)

    np.save("/media/matthew/Expansion/Widefield_Analysis/Discrimination_Analysis/Noise_Correlation_Changes/Genotype_Changes/Post_Learning_Sig.npy", thresholded_post_t)

    plt.show()


control_session_list = [
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    #"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging",

    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_16_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_17_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_19_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_21_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_24_Discrimination_Imaging",

    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_09_25_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_09_29_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_10_01_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_10_03_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_10_05_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_10_07_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging",

]


mutant_session_list = [
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_08_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_10_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_12_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_14_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_16_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_18_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_23_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_25_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_27_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_03_01_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_03_03_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging",

    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_09_28_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_09_30_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_09_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_11_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_13_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_15_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_17_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging",

    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK16.1B/2021_04_30_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK16.1B/2021_05_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK16.1B/2021_05_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK16.1B/2021_05_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK16.1B/2021_05_08_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK16.1B/2021_05_10_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK16.1B/2021_05_12_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK16.1B/2021_05_14_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK16.1B/2021_05_16_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK16.1B/2021_05_18_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK16.1B/2021_05_20_Discrimination_Imaging",

]




start_window = -10
stop_window = 14
condition_1 = "visual_1_all_onsets.npy"
condition_2 = "visual_2_all_onsets.npy"

analyse_noise_correlations_over_learning_seperate_stimuli(control_session_list, mutant_session_list, condition_1, start_window, stop_window)
test_differences_for_significance(control_session_list, mutant_session_list, condition_1, start_window, stop_window)