import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from matplotlib import gridspec
from scipy import stats
from skimage.feature import canny

def get_trial_tensor(delta_f_matrix, onset_list, start_window, stop_window):
    number_of_timepoints = np.shape(delta_f_matrix)[0]

    trial_tensor = []
    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window

        if trial_start >= 0 and trial_stop < number_of_timepoints - 1:
            trial_data = delta_f_matrix[trial_start:trial_stop]
            trial_tensor.append(trial_data)

    trial_tensor = np.array(trial_tensor)

    return trial_tensor


def normalise_activity_matrix(activity_matrix):
    # Subtract Min
    min_vector = np.min(activity_matrix, axis=0)
    activity_matrix = np.subtract(activity_matrix, min_vector)

    # Divide By Max
    max_vector = np.max(activity_matrix, axis=0)
    activity_matrix = np.divide(activity_matrix, max_vector)

    return activity_matrix


def get_sig_differences(condition_1_tensor, condition_2_tensor):
    number_of_timepoints = np.shape(condition_1_tensor)[1]

    p_array = []
    t_array = []

    for timepoint_index in range(number_of_timepoints):
        condition_1_vector = condition_1_tensor[:, timepoint_index]
        condition_2_vector = condition_2_tensor[:, timepoint_index]

        print("Condition 1 vector", np.shape(condition_1_vector))
        print("Condition 2 vector", np.shape(condition_2_vector))

        t_vector, p_vector = stats.ttest_ind(condition_1_vector, condition_2_vector)
        p_array.append(p_vector)
        t_array.append(t_vector)

    p_array = np.array(p_array)
    t_array = np.array(t_array)

    return t_array, p_array


def check_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)





def view_parcellated_activity_in_brain_space(activity_vector):

    # Load Clustes
    clusters = np.load(r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Final_Consensus_Clusters.npy")
    image_height, image_width = np.shape(clusters)
    unique_clusters = list(np.unique(clusters))

    # Remove Background
    unique_clusters.remove(0)
    reconstructed_image = np.ones((image_height, image_width, 4))

    positive_cmap = cm.get_cmap("Reds")
    negative_cmap = cm.get_cmap("Blues")

    edge_pixels = canny(clusters.astype("float32"))

    number_of_clusters = len(unique_clusters)
    for cluster_index in range(number_of_clusters):
        cluster = unique_clusters[cluster_index]

        cluster_pixels = np.nonzero(np.where(clusters==cluster, 1, 0))
        cluster_activity = activity_vector[cluster_index]

        if cluster_activity >= 0:
            cluster_colour = positive_cmap(cluster_activity)
        else:
            cluster_colour = negative_cmap(cluster_activity)

        reconstructed_image[cluster_pixels] = cluster_colour

    reconstructed_image[edge_pixels] = (0,0,0,0)
    return reconstructed_image


def view_parcellated_differences(session_list, condition_1, condition_2, start_window, stop_window, bin_size):



    # Iterate Through Each Session
    number_of_sessions = len(session_list)
    number_of_timepoints = stop_window - start_window

    # Create Figure
    figure_1 = plt.figure(figsize=(20, 15), dpi=80)
    rows = number_of_sessions
    columns = int(number_of_timepoints / bin_size)
    gridspec_1 = gridspec.GridSpec(rows, columns, figure=figure_1)

    for session_index in range(number_of_sessions):
        base_directory = session_list[session_index]

        # Load Neural Data
        activity_matrix = np.load(os.path.join(base_directory, "Cluster_Activity_Matrix.npy"))

        # Normalise Activity Matrix
        activity_matrix = normalise_activity_matrix(activity_matrix)

        # Remove Background Activity
        activity_matrix = activity_matrix[:, 1:]

        # Load Onsets
        vis_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_1))
        vis_2_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_2))

        # Get Trial Tensors
        vis_1_tensor = get_trial_tensor(activity_matrix, vis_1_onsets, start_window, stop_window)
        vis_2_tensor = get_trial_tensor(activity_matrix, vis_2_onsets, start_window, stop_window)

        # Get Average Tensors
        vis_1_mean = np.mean(vis_1_tensor, axis=0)
        vis_2_mean = np.mean(vis_2_tensor, axis=0)
        difference_tensor = np.subtract(vis_1_mean, vis_2_mean)

        bin_count = 0
        for timepoint in range(0, number_of_timepoints, bin_size):
            data_sample = difference_tensor[timepoint:timepoint+bin_size]
            data_sample = np.mean(data_sample, axis=0)

            axis = figure_1.add_subplot(gridspec_1[session_index, bin_count])
            reconstructed_image = view_parcellated_activity_in_brain_space(data_sample)
            axis.imshow(reconstructed_image)
            axis.axis('off')
            bin_count += 1
    plt.show()



session_list = [
     "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging"]


# Decoding Parameters
start_window = -14
stop_window = 34
bin_size = 8
condition_1 = "visual_1_all_onsets.npy"
condition_2 = "visual_2_all_onsets.npy"
plot_save_directory = "/media/matthew/Expansion/Widefield_Analysis/Discrimination_Analysis/Average_Rasters"


view_parcellated_differences(session_list, condition_1, condition_2, start_window, stop_window, bin_size)