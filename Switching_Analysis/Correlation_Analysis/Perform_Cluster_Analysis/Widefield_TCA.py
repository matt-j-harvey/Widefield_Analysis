import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FactorAnalysis, TruncatedSVD, FastICA, PCA, NMF
import networkx as nx
import cv2
from matplotlib.pyplot import cm
from matplotlib.colors import to_rgb
import os
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
from tensorly.decomposition import parafac, CP, non_negative_parafac



def create_trial_tensor(delta_f_matrix, onsets, start_window, stop_window):
    # Given A List Of Trial Onsets - Create A 3 Dimensional Tensor (Trial x Neuron x Trial_Aligned_Timepoint)

    number_of_timepoints = np.shape(delta_f_matrix)[1]

    # Transpose Delta F Matrix So Its Time x Neurons
    delta_f_matrix = np.transpose(delta_f_matrix)

    selected_data = []
    for onset in onsets:
        trial_start = onset + start_window
        trial_stop = onset + stop_window

        if trial_stop < number_of_timepoints:
            trial_data = delta_f_matrix[int(trial_start):int(trial_stop)]
            selected_data.append(trial_data)

    selected_data = np.array(selected_data)

    return selected_data




def get_block_boundaries(combined_onsets, visual_context_onsets, odour_context_onsets):

    visual_blocks = []
    odour_blocks = []

    current_block_start = 0
    current_block_end = None

    # Get Initial Onset
    if combined_onsets[0] in visual_context_onsets:
        current_block_type = 0
    elif combined_onsets[0] in odour_context_onsets:
        current_block_type = 1
    else:
        print("Error! onsets not in either vidual or oflactory onsets")

    # Iterate Through All Subsequent Onsets
    number_of_onsets = len(combined_onsets)
    for onset_index in range(1, number_of_onsets):

        # Get Onset
        onset = combined_onsets[onset_index]

        # If we are currently in an Visual Block
        if current_block_type == 0:

            # If The Next Onset is An Odour Block - Block Finish, add Block To Boundaries
            if onset in odour_context_onsets:
                current_block_end = onset_index-1
                visual_blocks.append([current_block_start, current_block_end])
                current_block_type = 1
                current_block_start = onset_index

        # If we Are currently in an Odour BLock
        if current_block_type == 1:

            # If The NExt Onset Is a Visual Trial - BLock Finish Add Block To Block Boundaires
            if onset in visual_context_onsets:
                current_block_end = onset_index - 1
                odour_blocks.append([current_block_start, current_block_end])
                current_block_type = 0
                current_block_start = onset_index

    return visual_blocks, odour_blocks


def perform_factor_analysis(tensor, n_components=7):

    # Remove Nans
    tensor = np.nan_to_num(tensor)

    # Get Tensor Structure
    number_of_trials = np.shape(tensor)[0]
    number_of_clusters = np.shape(tensor)[1]

    # Concatenate Trials
    tensor = np.reshape(tensor, (number_of_trials, number_of_clusters * number_of_clusters))

    print("Reshaped Tensor Shape", np.shape(tensor))

    # Perform Factor Analysis
    #tensor = np.clip(tensor, a_min=0, a_max=None)
    model = FactorAnalysis(n_components=n_components)
    model.fit(tensor)

    # Get Components
    components = model.components_

    # Factor Trajectories
    low_dimensional_trajectories = model.transform(tensor)

    print("Trajectories Shape", np.shape(low_dimensional_trajectories))

    return components,  low_dimensional_trajectories


def get_cluster_image(base_directory, cluster_activity):

    # Downsample Mask
    downsampled_indicies, downsampled_height, downsampled_width = downsample_mask(base_directory)

    # Load Cluster Pixels
    clusters = np.load("/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/clean_clusters.npy", allow_pickle=True)
    number_of_clusters = len(clusters)

    # View
    image = np.zeros((downsampled_height * downsampled_width))
    for cluster_index in range(number_of_clusters):
        cluster = clusters[cluster_index]
        activity = cluster_activity[cluster_index]
        for pixel in cluster:
            pixel_index = downsampled_indicies[pixel]
            image[pixel_index] = activity

    image = np.ndarray.reshape(image, (downsampled_height, downsampled_width))

    return image


def plot_factors_combined(cluster_loadings, time_loadings, trial_loadings, visual_blocks, odour_blocks, base_directory):

    print("Cluster Loadings Shape", np.shape(cluster_loadings))
    print("Time Loadings Shape", np.shape(time_loadings))
    print("Trial Loadings Shape", np.shape(trial_loadings))

    number_of_factors = np.shape(cluster_loadings)[1]

    rows = number_of_factors
    columns = 3

    figure_count = 1
    figure_1 = plt.figure()
    #figure_1.suptitle(session_name)

    for factor in range(number_of_factors):

        cluster_axis = figure_1.add_subplot(rows, columns, figure_count + 0)
        time_axis   = figure_1.add_subplot(rows, columns, figure_count + 1)
        trial_axis  = figure_1.add_subplot(rows, columns, figure_count + 2)

        figure_count += 3

        cluster_axis.set_title("Factor " + str(factor) + " Cluster Loadings")
        time_axis.set_title("Factor " + str(factor) + " Time Loadings")
        trial_axis.set_title("Factor " + str(factor) + " Trial Loadings")

        cluster_data = cluster_loadings[:, factor]
        time_data = time_loadings[:, factor]
        trial_data = trial_loadings[:, factor]

        time_axis.plot(time_data)
        trial_axis.plot(trial_data, c='orange')

        # Plot Weight Matrix
        cluster_image = get_cluster_image(base_directory, cluster_data)
        cluster_magnitude = np.max(np.abs(cluster_image))
        cluster_axis.imshow(cluster_image, cmap='plasma', vmax=cluster_magnitude, vmin=0)

        # Highligh Blocks
        for block in visual_blocks:
            trial_axis.axvspan(block[0], block[1], alpha=0.2, color='blue')
        for block in odour_blocks:
            trial_axis.axvspan(block[0], block[1], alpha=0.2, color='green')

    figure_1.set_size_inches(18.5, 16)
    figure_1.tight_layout()
    plt.show()


def downsample_mask(base_directory):

    # Load Mask
    mask = np.load(base_directory + "/mask.npy")

    # Downsample Mask
    original_height = np.shape(mask)[0]
    original_width = np.shape(mask)[1]
    downsampled_height = int(original_height/2)
    downsampled_width = int(original_width/2)
    downsampled_mask = cv2.resize(mask, dsize=(downsampled_width, downsampled_height))

    # Binairse Mask
    downsampled_mask = np.where(downsampled_mask > 0.1, 1, 0)
    downsampled_mask = downsampled_mask.astype(int)

    flat_mask = np.ndarray.flatten(downsampled_mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, downsampled_height, downsampled_width


def perform_widefield_TCA(session_list):

    trial_start = -10
    trial_stop = 50

    for base_directory in session_list:
        session_name = base_directory.split("/")[-3]

        # Load Activity Matrix
        cluster_activity_matrix_file = base_directory + "/Cluster_Activity_Matrix.npy"
        activity_matrix = np.load(cluster_activity_matrix_file)
        activity_matrix = np.nan_to_num(activity_matrix)

        # Load Stimuli Onsets
        stimuli_onsets_directory = base_directory + "/Stimuli_Onsets/"
        visual_context_onsets_vis_1 = np.load(stimuli_onsets_directory + "vis_context_vis_1_onsets.npy")
        visual_context_onsets_vis_2 = np.load(stimuli_onsets_directory + "vis_context_vis_2_onsets.npy")
        odour_context_onsets_vis_1  = np.load(stimuli_onsets_directory + "odour_context_vis_1_onsets.npy")
        odour_context_onsets_vis_2  = np.load(stimuli_onsets_directory + "odour_context_vis_2_onsets.npy")

        # Arrange Onsets
        all_onsets = np.concatenate([visual_context_onsets_vis_1, visual_context_onsets_vis_2, odour_context_onsets_vis_1, odour_context_onsets_vis_2])
        all_onsets.sort()

        visual_context_onsets = np.concatenate([visual_context_onsets_vis_1, visual_context_onsets_vis_2])
        visual_context_onsets.sort()

        odour_context_onsets = np.concatenate([odour_context_onsets_vis_1, odour_context_onsets_vis_2])
        odour_context_onsets.sort()

        all_vis_1_onsets = np.concatenate([visual_context_onsets_vis_1, odour_context_onsets_vis_1])
        all_vis_2_onsets = np.concatenate([visual_context_onsets_vis_2, odour_context_onsets_vis_2])

        # Get Block Boundaries
        visual_blocks, odour_blocks = get_block_boundaries(all_onsets, visual_context_onsets, odour_context_onsets)

        # Create Trial Tensor
        trial_tensor = create_trial_tensor(activity_matrix, all_onsets, trial_start, trial_stop)

        # Perform Tensor Decomposition
        weights, factors = non_negative_parafac(trial_tensor, rank=12, init='svd', verbose=1, n_iter_max=250)

        trial_loadings = factors[0]
        time_loadings = factors[1]
        neuron_loadings = factors[2]

        # Plot These Factors
        plot_factors_combined(neuron_loadings, time_loadings, trial_loadings, visual_blocks, odour_blocks, base_directory)

    print("Mean  Scores", mean_scores)


#"/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/",

session_list = ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging/"]

perform_widefield_TCA(session_list)