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



def create_correlation_tensor(activity_matrix, onsets, start_window, stop_window):

    print("Activity Matrix Shape", np.shape(activity_matrix))

    # Get Tensor Details
    number_of_clusters = np.shape(activity_matrix)[0]
    number_of_trials = np.shape(onsets)[0]

    # Create Empty Tensor To Hold Data
    correlation_tensor = np.zeros((number_of_trials, number_of_clusters, number_of_clusters))

    plt.ion()
    figure_1 = plt.figure()

    # Get Correlation Matrix For Each Trial
    for trial_index in range(0, number_of_trials):

        # Get Trial Activity
        trial_start = onsets[trial_index] + start_window
        trial_stop = onsets[trial_index] + stop_window
        trial_activity = activity_matrix[:, trial_start:trial_stop]

        print("Trial: ", trial_index, " of ", number_of_trials, " Onset: ", trial_start, " Offset: ", trial_stop)

        # Get Trial Correlation Matrix
        trial_correlation_matrix = np.zeros((number_of_clusters, number_of_clusters))

        for cluster_1_index in range(number_of_clusters):
            cluster_1_trial_trace = trial_activity[cluster_1_index]

            for cluster_2_index in range(cluster_1_index + 1, number_of_clusters):
                cluster_2_trial_trace = trial_activity[cluster_2_index]


                correlation = np.corrcoef(cluster_1_trial_trace, cluster_2_trial_trace)[0][1]

                trial_correlation_matrix[cluster_1_index][cluster_2_index] = correlation
                trial_correlation_matrix[cluster_2_index][cluster_1_index] = correlation

        correlation_tensor[trial_index] = trial_correlation_matrix
        plt.title(str(trial_index))
        axis_1 = figure_1.add_subplot(1, 2, 1)
        axis_2 = figure_1.add_subplot(1, 2, 2)
        axis_1.imshow(trial_correlation_matrix, cmap='bwr', vmin=-1, vmax=1)
        axis_2.imshow(trial_activity, vmin=0, vmax=1, cmap='inferno')
        plt.draw()
        plt.pause(0.01)
        plt.clf()

    #plt.imshow(np.mean(correlation_tensor, axis=0), cmap='bwr', vmin=-1, vmax=1)
    #plt.show()

    return correlation_tensor



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




def plot_factors_combined(trial_loadings, weight_loadings, visual_blocks, odour_blocks):

    print("Weight Data Shape", np.shape(weight_loadings))
    print("TRial Loadings Shape", np.shape(trial_loadings))

    number_of_factors = np.shape(trial_loadings)[1]
    number_of_correlations = np.shape(weight_loadings)[1]
    number_of_clusters = int(math.sqrt(number_of_correlations))

    print("Number of Factors", number_of_factors)
    print("Number of correlations", number_of_correlations)
    print("Number of clusters", number_of_clusters)

    rows = number_of_factors
    columns = 2

    figure_count = 1
    figure_1 = plt.figure()
    #figure_1.suptitle(session_name)
    for factor in range(number_of_factors):
        weights_axis = figure_1.add_subplot(rows,  columns, figure_count)
        trial_axis = figure_1.add_subplot(rows, columns, figure_count + 1)
        figure_count += 2

        weights_axis.set_title("Factor " + str(factor) + " Weight Loadings")
        trial_axis.set_title("Factor " + str(factor) + " Trial Loadings")

        weight_data = weight_loadings[factor]
        trial_data = trial_loadings[:, factor]

        # Plot Weight Matrix
        weight_data = np.reshape(weight_data, (number_of_clusters, number_of_clusters))
        weights_axis.imshow(weight_data, cmap='bwr')

        trial_axis.plot(trial_data, c='orange')

        # Highligh Blocks
        for block in visual_blocks:
            trial_axis.axvspan(block[0], block[1], alpha=0.2, color='blue')
        for block in odour_blocks:
            trial_axis.axvspan(block[0], block[1], alpha=0.2, color='green')

    figure_1.set_size_inches(18.5, 16)
    figure_1.tight_layout()
    plt.show()
    plt.close()


def compare_pre_stimulus_correlations(session_list):
    trial_start = -70
    trial_stop = -14

    for base_directory in session_list:
        session_name = base_directory.split("/")[-3]

        # Load Activity Matrix
        cluster_activity_matrix_file = base_directory + "/Cluster_Activity_Matrix.npy"
        activity_matrix = np.load(cluster_activity_matrix_file)
        activity_matrix = np.nan_to_num(activity_matrix)

        plt.imshow(activity_matrix, aspect=100, cmap='jet', vmin=0, vmax=1)
        plt.show()

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

        """
        # Create Trial Tensor
        trial_correlation_tensor = create_correlation_tensor(activity_matrix, all_onsets, trial_start, trial_stop)

        # Save Trial Tensor
        np.save(base_directory + "FA_Correlation_Tensor.npy", trial_correlation_tensor)

        trial_correlation_tensor = np.load(base_directory + "FA_Correlation_Tensor.npy")
        print("Correlation Tensor Shape", np.shape(trial_correlation_tensor))

        # Perform Dimensionality Reduction
        components, low_dimensional_trajectories = perform_factor_analysis(trial_correlation_tensor)

        # Save These
        np.save(base_directory + "FA_Components.npy", components)
        np.save(base_directory + "FA_L_D_Trajectories.npy", low_dimensional_trajectories)
        """

        # Load These
        components = np.load(base_directory + "FA_Components.npy")
        low_dimensional_trajectories = np.load(base_directory + "FA_L_D_Trajectories.npy")

        """
        for x in range(7):
            trajectory = low_dimensional_trajectories[:, x]
            plt.plot(trajectory)
            plt.show()
        """

        # Plot These Factors
        plot_factors_combined(low_dimensional_trajectories, components, visual_blocks, odour_blocks)

    print("Mean  Scores", mean_scores)


#"/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/",

session_list = [
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging/"]

compare_pre_stimulus_correlations(session_list)