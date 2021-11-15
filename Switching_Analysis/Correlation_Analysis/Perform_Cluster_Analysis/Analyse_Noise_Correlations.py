import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import networkx as nx
import cv2
from matplotlib.pyplot import cm
from matplotlib.colors import to_rgb
import os
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt




def draw_brain_network(base_directory, adjacency_matrix, session_name):

    # Load Cluster Centroids
    cluster_centroids = np.load(base_directory + "/Cluster_Centroids.npy")

    # Create NetworkX Graph
    graph = nx.from_numpy_matrix(adjacency_matrix)

    # Get Edge Weights
    edges = graph.edges()
    weights = [graph[u][v]['weight'] for u, v in edges]
    weights = np.divide(weights, np.max(np.abs(weights)))

    # Get Edge Colours
    colourmap = cm.get_cmap('bwr')
    colours = []
    for weight in weights:
        colour = colourmap(weight)
        colours.append(colour)

    # Load Cluster Outlines
    cluster_outlines = np.load("/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/clean_clusters_outline.npy")
    plt.imshow(cluster_outlines, cmap='binary', vmax=2)

    image_height = np.shape(cluster_outlines)[0]

    # Draw Graph
    # Invert Cluster Centroids
    inverted_centroids = []
    for centroid in cluster_centroids:
        y_value = centroid[1]
        x_value = centroid[0]
        inverted_y = image_height - y_value
        inverted_centroids.append([x_value, inverted_y])

    plt.title(session_name)
    nx.draw(graph, pos=inverted_centroids, node_size=1,  width=weights, edge_color=colours)
    plt.show()
    #plt.savefig(base_directory + "/" + session_name + "_Signficant_Correlation_Changes.png")
    #plt.close()




def sort_matrix(matrix):

    # Cluster Matrix
    Z = ward(pdist(matrix))

    # Get Dendogram Leaf Order
    new_order = leaves_list(Z)

    # Sorted Matrix
    sorted_matrix = matrix[:, new_order][new_order]

    return sorted_matrix


def plot_correlation_matirices(matrix_1, matrix_2):

    figure_1 = plt.figure()
    matrix_1_axis   = figure_1.add_subplot(1, 3, 1)
    matrix_2_axis   = figure_1.add_subplot(1, 3, 2)
    difference_axis = figure_1.add_subplot(1, 3, 3)

    meta_array = np.array([matrix_1, matrix_2])
    delta_array = np.diff(meta_array, axis=0)[0]

    # Cluster Delta Array
    Z = ward(pdist(delta_array))
    new_order = leaves_list(Z)

    # Sort All Arrays By This Order
    sorted_matrix_1 = matrix_1[:, new_order][new_order]
    sorted_matrix_2 = matrix_2[:, new_order][new_order]
    sorted_delta_matrix = delta_array[:, new_order][new_order]

    """
    matrix_1_axis.imshow(sorted_matrix_1, cmap='bwr', vmin=-1, vmax=1)
    matrix_2_axis.imshow(sorted_matrix_2,  cmap='bwr', vmin=-1, vmax=1)
    difference_axis.imshow(sorted_delta_matrix, cmap='bwr', vmin=-1, vmax=1)
    plt.show()
    """
    return delta_array



def get_activity_tensor(activity_matrix, onsets, start_window, stop_window):

    # Get Tensor Details
    number_of_clusters = np.shape(activity_matrix)[0]
    number_of_trials = np.shape(onsets)[0]
    number_of_timepoints = stop_window - start_window

    # Create Empty Tensor To Hold Data
    trial_tensor = np.zeros((number_of_trials, number_of_timepoints, number_of_clusters))

    # Get Correlation Matrix For Each Trial
    for trial_index in range(number_of_trials):
        print("Trial: ", trial_index, " of ", number_of_trials)

        # Get Trial Activity
        trial_start = onsets[trial_index] + start_window
        trial_stop = onsets[trial_index] + stop_window
        trial_activity = activity_matrix[:, trial_start:trial_stop]

        trial_activity = np.transpose(trial_activity)

        print("Trial Activity Shape", np.shape(trial_activity))
        trial_tensor[trial_index] = trial_activity

    return trial_tensor


def get_noise_correlations(tensor):

    # Get Tensor Structure
    number_of_trials = np.shape(tensor)[0]
    number_of_timepoints = np.shape(tensor)[1]
    number_of_clusters = np.shape(tensor)[2]

    # Get Mean Trace
    mean_trace = np.mean(tensor, axis=0)

    # Subtract Mean Trace
    subtracted_tensor = np.subtract(tensor, mean_trace)

    # Concatenate Trials
    concatenated_subtracted_tensor = np.reshape(subtracted_tensor, (number_of_trials * number_of_timepoints, number_of_clusters))
    concatenated_subtracted_tensor = np.transpose(concatenated_subtracted_tensor)

    # Calculate Correlation Matrix
    noise_correlation_matrix = np.corrcoef(concatenated_subtracted_tensor)
    mean_correlation_matrix = np.corrcoef(np.transpose(mean_trace))
    """"
    # View Process For Sanity Checking
    print("Noise Correlations Raw Tensor", np.shape(tensor))
    
    print("Mean Trace Shape", np.shape(mean_trace))
    plt.title("Mean Trace")
    plt.imshow(np.transpose(mean_trace))
    plt.show()
    
    print("Subtracted tensor shape", np.shape(tensor))
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 3, 1)
    axis_2 = figure_1.add_subplot(1, 3, 2)
    axis_3 = figure_1.add_subplot(1, 3, 3)

    plt.title("Subtracted Tensor")
    axis_1.imshow(np.transpose(mean_trace))
    axis_2.imshow(np.transpose(tensor[0]))
    axis_3.imshow(np.transpose(subtracted_tensor[0]))
    plt.show()
    
    print("Reshaped Subtracted Tensor", np.shape(subtracted_tensor))
    plt.title("Concatenated Subtracted Tensor")
    plt.imshow(np.transpose(concatenated_subtracted_tensor))
    plt.show()
    
    correlation_matrix = sort_matrix(correlation_matrix)
    plt.imshow(correlation_matrix, cmap='bwr', vmin=-1, vmax=1)
    plt.show()
    """
    return noise_correlation_matrix, mean_correlation_matrix


def compare_noise_correlations(session_list):

    trial_start = 0
    trial_stop = 40

    mean_delta_arrays = []

    for base_directory in session_list:

        # Get Session Name
        session_name = base_directory.split("/")[-3]
        print(session_name)

        # Create Save Directory
        save_directory = base_directory + "/Noise_Correlations"
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)

        # Load Activity Matrix
        cluster_activity_matrix_file = base_directory + "/Cluster_Activity_Matrix.npy"
        activity_matrix = np.load(cluster_activity_matrix_file)

        # Load Stimuli Onsets
        visual_context_onsets  = np.load(base_directory + r"/Stimuli_Onsets/odour_context_stable_vis_2_frame_onsets.npy")
        odour_context_onsets = np.load(base_directory + r"/Stimuli_Onsets/visual_context_stable_vis_2_frame_onsets.npy")

        # Create Activity Tensors
        visual_context_tensor = get_activity_tensor(activity_matrix, visual_context_onsets, trial_start, trial_stop)
        odour_context_tensor = get_activity_tensor(activity_matrix, odour_context_onsets, trial_start, trial_stop)

        print("Visual context tensor", np.shape(visual_context_tensor))

        # Get Noise Correlation Matirices
        visual_context_noise_correlations, visual_context_mean_correlations = get_noise_correlations(visual_context_tensor)
        odour_context_noise_correlations, odour_context_mean_correlations = get_noise_correlations(odour_context_tensor)

        plt.imshow(visual_context_noise_correlations, cmap='bwr', vmin=-1, vmax=1)
        plt.show()

        plt.imshow(odour_context_noise_correlations, cmap='bwr', vmin=-1, vmax=1)
        plt.show()

        # Get Difference In Noise Correlations
        noise_delta_matrix = np.diff(np.array([visual_context_noise_correlations, odour_context_noise_correlations]), axis=0)[0]

        # Get Difference In Mean Correlations
        mean_delta_matrix = np.diff(np.array([visual_context_mean_correlations, odour_context_mean_correlations]), axis=0)[0]

        #sorted_delta_matrix = sort_matrix(delta_matrix)
        #plt.imshow(sorted_delta_matrix, cmap='bwr', vmin=-1, vmax=1)
        #plt.show()


        # Save Difference
        np.save(save_directory + "/Noise_Correlation_Delta_Matrix.npy", noise_delta_matrix)
        np.save(save_directory + "/Mean_Correlation_Delta_Matrix.npy", mean_delta_matrix)

        """
        # View Matricies
        delta_matrix = plot_correlation_matirices(visual_context_noise_correlations, odour_context_noise_correlations)

        # Draw Brain Map
        threshold = 1
        threshold_delta_matrix = np.where(abs(delta_matrix) > threshold, delta_matrix, 0)
        #draw_brain_network(base_directory, threshold_delta_matrix, session_name)

        mean_delta_arrays.append(delta_matrix)
  
    mean_delta_matrix = np.mean(mean_delta_arrays, axis=0)
    threshold = 0.6
    threshold_delta_matrix = np.where(abs(mean_delta_matrix) > threshold, mean_delta_matrix, 0)
    draw_brain_network(base_directory, threshold_delta_matrix, session_name)


    """

clusters_file = "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/clean_clusters.npy"

session_list = [
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging/",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging/",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging/",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging"]


compare_noise_correlations(session_list)