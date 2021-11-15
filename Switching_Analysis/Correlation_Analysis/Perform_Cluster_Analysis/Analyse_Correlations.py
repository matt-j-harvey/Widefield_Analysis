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

def create_correlation_tensor(activity_matrix, onsets, start_window, stop_window):

    # Get Tensor Details
    number_of_clusters = np.shape(activity_matrix)[0]
    number_of_trials = np.shape(onsets)[0]

    # Create Empty Tensor To Hold Data
    correlation_tensor = np.zeros((number_of_trials, number_of_clusters, number_of_clusters))

    # Get Correlation Matrix For Each Trial
    for trial_index in range(number_of_trials):
        print("Trial: ", trial_index, " of ", number_of_trials)

        # Get Trial Activity
        trial_start = onsets[trial_index] + start_window
        trial_stop = onsets[trial_index] + stop_window
        trial_activity = activity_matrix[:, trial_start:trial_stop]

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
        #plt.imshow(trial_correlation_matrix, cmap='bwr', vmin=-1, vmax=1)
        #plt.show()

    #plt.imshow(np.mean(correlation_tensor, axis=0), cmap='bwr', vmin=-1, vmax=1)
    #plt.show()

    return correlation_tensor


def plot_correlation_matirices(visual_context_correlation_tensor, odour_context_correlation_tensor):

    figure_1 = plt.figure()
    visual_context_axis = figure_1.add_subplot(1, 3, 1)
    odour_context_axis = figure_1.add_subplot(1, 3, 2)
    difference_axis = figure_1.add_subplot(1, 3, 3)

    mean_visual_context_correlation_matrix = np.mean(visual_context_correlation_tensor, axis=0)
    mean_odour_context_correlation_matrix = np.mean(odour_context_correlation_tensor, axis=0)
    difference_correlation_matrix = np.subtract(mean_visual_context_correlation_matrix, mean_odour_context_correlation_matrix)

    visual_context_axis.imshow(mean_visual_context_correlation_matrix, cmap='bwr', vmin=-1, vmax=1)
    odour_context_axis.imshow(mean_odour_context_correlation_matrix, cmap='bwr', vmin=-1, vmax=1)
    difference_axis.imshow(difference_correlation_matrix, cmap='bwr', vmin=-1, vmax=1)
    plt.show()


def get_significant_changes(base_directory, clusters_file, session_name, visual_context_correlation_tensor, odour_context_correlation_tensor):


    # Compute Significant Differences
    t_statistics, p_values = stats.ttest_ind(a=visual_context_correlation_tensor, b=odour_context_correlation_tensor, axis=0)

    p_value_list = np.copy(p_values)
    p_value_list = np.ndarray.flatten(p_value_list)
    p_value_list = list(p_value_list)
    p_value_list.sort(reverse=True)
    print("P Value List", p_value_list)

    # Get Bonferroni Corrected P Value
    number_of_tests = len(p_value_list)
    p_threshold = 0.05 / number_of_tests
    print("Corrected P Threshold", p_threshold)

    # Show Significant Changes
    signficance_map = np.where(p_values < p_threshold, t_statistics, 0)

    # Exclude Midline
    signficance_map = exclude_clusters(signficance_map)

    plt.title("Signficant Correlation Changes")
    plt.imshow(signficance_map, cmap='bwr', vmin=-2, vmax=2)
    plt.savefig(base_directory + "/Significant_Correlation_Changes.png")
    plt.close()

    """
    # Get Correlation Modulation Map
    mean_odour_correlation_matrix = np.mean(odour_context_correlation_tensor, axis=0)
    mean_visual_correlation_matrix = np.mean(visual_context_correlation_tensor, axis=0)
    correlation_modulation = np.subtract(mean_visual_correlation_matrix, mean_odour_correlation_matrix)
    correlation_modulation_map = np.where(p_values < 0.01, correlation_modulation, 0)
    """

    #plot_correlation_maps(base_directory, correlation_modulation_map, clusters_file)
    #draw_modulation_map(base_directory, np.abs(signficance_map), clusters_file, "Significant Correlation Changes")
    draw_brain_network(base_directory, np.abs(signficance_map), session_name)


def decode_context_from_correlation_matrix(base_directory, clusters_file):

    # Load Tensors
    visual_context_correlation_tensor = np.load(base_directory + "/Visual_Context_Correlation_Tensor.npy")
    odour_context_correlation_tensor = np.load(base_directory + "/Odour_Context_Correlation_Tensor.npy")

    # Flatten Tensors
    number_of_clusters = np.shape(visual_context_correlation_tensor)[1]
    number_of_visual_trials = np.shape(visual_context_correlation_tensor)[0]
    number_of_odour_trials = np.shape(odour_context_correlation_tensor)[0]
    visual_context_correlation_tensor = np.ndarray.reshape(visual_context_correlation_tensor, (number_of_visual_trials, number_of_clusters * number_of_clusters))
    odour_context_correlation_tensor = np.ndarray.reshape(odour_context_correlation_tensor, (number_of_odour_trials, number_of_clusters * number_of_clusters))

    # Combine Datasets and Create Labels
    visual_context_labels = np.zeros(number_of_visual_trials)
    odour_context_labels = np.ones(number_of_odour_trials)
    labels = np.concatenate([visual_context_labels, odour_context_labels])
    dataset = np.concatenate([visual_context_correlation_tensor,  odour_context_correlation_tensor])

    # Create Model
    model = LogisticRegression()

    # Perform Decoding
    score_list = []
    coefficient_matrix = []

    number_of_folds = 5
    for fold in range(number_of_folds):

        # Split Data Into Train and Test Sets
        X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2)

        # Train  Model
        model.fit(X_train, y_train)
        coefficient_matrix.append(model.coef_)

        # Test Model
        score = model.score(X_test, y_test)
        score_list.append(score)

    coefficient_matrix = np.array(coefficient_matrix)
    print("Coefficeint matrix", np.shape(coefficient_matrix))
    coefficient_matrix = np.mean(coefficient_matrix, axis=0)

    coefficient_matrix = np.ndarray.reshape(coefficient_matrix, (number_of_clusters, number_of_clusters))
    coefficient_matrix = np.abs(coefficient_matrix)
    #draw_modulation_map(base_directory, coefficient_matrix, clusters_file)
    print("Score List", score_list)

    return score_list



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


def get_cluster_centroids(base_directory, clusters_file):

    # Downsample Mask
    downsampled_indicies, downsampled_height, downsampled_width = downsample_mask(base_directory)

    # Load Clusters
    clusters = np.load(clusters_file, allow_pickle=True)
    number_of_clusters = len(clusters)

    # Get Cluster Centroids
    cluster_centroids = np.zeros((number_of_clusters, 2))
    for cluster_index in range(number_of_clusters):
        cluster = clusters[cluster_index][0]
        cluster_pixel_coordinates = []

        for pixel in cluster:

            # Convert Pixel Index Into X Y Coords
            pixel_index = downsampled_indicies[pixel]
            y_coordinate = int(pixel_index / downsampled_width)
            x_coordinate = pixel_index - (downsampled_width * y_coordinate)
            y_coordinate = downsampled_height - y_coordinate

            cluster_pixel_coordinates.append([x_coordinate, y_coordinate])

        cluster_pixel_coordinates = np.array(cluster_pixel_coordinates)
        cluster_centroid = np.mean(cluster_pixel_coordinates, axis=0)
        cluster_centroids[cluster_index] = cluster_centroid

    np.save(base_directory + "/Cluster_Centroids.npy", cluster_centroids)
    #plt.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1])
    #plt.show()



def draw_modulation_map(base_directory, adjacency_matrix, clusters_file, save_name):

    # Downsample Mask
    downsampled_indicies, downsampled_height, downsampled_width = downsample_mask(base_directory)

    # Create Regional Modulation Vector
    regional_modulation = np.mean(adjacency_matrix, axis=1)

    # Load Clusters
    clusters = np.load(clusters_file, allow_pickle=True)
    number_of_clusters = len(clusters)

    # Draw Brain Mask
    image = np.zeros((downsampled_height * downsampled_width))
    for cluster_index in range(number_of_clusters):
        cluster = clusters[cluster_index]
        for pixel in cluster:
            pixel_index = downsampled_indicies[pixel]
            image[pixel_index] = regional_modulation[cluster_index]

    image = np.ndarray.reshape(image, (downsampled_height, downsampled_width))

    plt.title(save_name)
    plt.imshow(image, cmap='jet')
    plt.savefig(base_directory + "/" + save_name + ".png")
    plt.close()


def rgba_to_rgb(rgba_value):

    source_r = rgba_value[0]
    source_g = rgba_value[1]
    source_b = rgba_value[2]
    source_a = rgba_value[3]
    background_colour = [1,1,1,1]

    Target_R = ((1 - source_a) * background_colour[0]) + (source_a * source_r)
    Target_G = ((1 - source_a) * background_colour[1]) + (source_a * source_g)
    Target_B = ((1 - source_a) * background_colour[2]) + (source_a * source_b)

    return [Target_R, Target_G, Target_B]


def plot_correlation_maps(base_directory, correlation_modulation_map, clusters_file):

    # Downsample Mask
    downsampled_indicies, downsampled_height, downsampled_width = downsample_mask(base_directory)

    # Load Clusters
    clusters = np.load(clusters_file, allow_pickle=True)
    number_of_clusters = len(clusters)

    # Get  Colourmaps
    red_colourmap = cm.get_cmap('Reds')
    blue_colourmap = cm.get_cmap('Blues')
    selection_colourmap = cm.get_cmap('plasma')

    save_directory = base_directory + "/Correlation_Modulation_Maps/"
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Draw Brain Mask
    for current_cluster_index in range(number_of_clusters):
        print("Cluster: ", current_cluster_index, " of ", number_of_clusters)


        # Create Empty Image
        image = np.zeros((downsampled_height * downsampled_width))
        region_image = np.zeros((downsampled_height * downsampled_width))

        # Get Correlation Modulation For This Cluster
        cluster_correlations = correlation_modulation_map[current_cluster_index]

        # Iterate Through All Other Clusters
        for other_cluster_index in range(number_of_clusters):

            # Get Modulation For Other Cluster
            correlation_modulation = cluster_correlations[other_cluster_index]

            # Assign colour
            if correlation_modulation >= 0:
                colour = red_colourmap(correlation_modulation)
#
            elif correlation_modulation < 0:
                colour = blue_colourmap(abs(correlation_modulation))

            # Colour Pixels
            other_cluster = clusters[other_cluster_index]
            for pixel in other_cluster:
                pixel_index = downsampled_indicies[pixel]
                image[pixel_index] = correlation_modulation


        # Highlight Current Cluster
        current_cluster = clusters[current_cluster_index]
        current_colour = selection_colourmap(255)

        region_image[downsampled_indicies] = 0.3
        for pixel in current_cluster:
            pixel_index = downsampled_indicies[pixel]
            region_image[pixel_index] += 1

        image = np.ndarray.reshape(image, (downsampled_height, downsampled_width))
        region_image = np.ndarray.reshape(region_image, (downsampled_height, downsampled_width))
        #rgb_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        figure_1 = plt.figure()
        axis_1 = figure_1.add_subplot(1,2,1)
        axis_2 = figure_1.add_subplot(1,2,2)

        axis_1.imshow(region_image, cmap='Blues')
        axis_2.imshow(image, cmap='bwr', vmin=-1, vmax=1)

        plt.savefig(save_directory + str(current_cluster_index).zfill(3) + ".png")
        plt.close()



def draw_brain_network(base_directory, adjacency_matrix, session_name):

    # Load Cluster Centroids
    cluster_centroids = np.load(base_directory + "/Cluster_Centroids.npy")

    # Create NetworkX Graph
    graph = nx.from_numpy_matrix(adjacency_matrix)

    # Get Edge Weights
    edges = graph.edges()
    weights = [graph[u][v]['weight'] for u, v in edges]
    weights = np.divide(weights, np.max(weights))

    # Get Edge Colours
    colourmap = cm.get_cmap('plasma')
    colours = []
    for weight in weights:
        colour = colourmap(weight)
        colours.append(colour)

    # Load Cluster Outlines
    cluster_outlines = np.load("/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/clean_clusters_outline.npy")
    plt.imshow(cluster_outlines)

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
    plt.savefig(base_directory + "/" + session_name + "_Signficant_Correlation_Changes.png")
    plt.close()


def exclude_clusters(matrix):

    excluded_cluster_list = [2, 119, 156, 160, 217]

    for cluster in excluded_cluster_list:
        matrix[cluster] = 0
        matrix[:, cluster] =0
    return matrix

def view_cluster_centroids(base_directory):

    # Load Cluster Centroids
    cluster_centroids = np.load(base_directory + "/Cluster_Centroids.npy")

    # Load Clusters
    cluster_outlines = "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/clean_clusters_outline.npy"
    cluster_outlines = np.load(cluster_outlines)

    cluster_outlines = np.flip(cluster_outlines, 0)

    plt.imshow(cluster_outlines, origin='lower')
    plt.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1])
    plt.show()


def compare_pre_stimulus_correlations(session_list):
    trial_start = -70
    trial_stop = -14

    for base_directory in session_list:
        session_name = base_directory.split("/")[-3]
        session_name = session_name + "Stimulus_Evoked_"


        print(session_name)
        # view_cluster_centroids(base_directory)

        # Load Activity Matrix
        cluster_activity_matrix_file = base_directory + "/Cluster_Activity_Matrix.npy"
        activity_matrix = np.load(cluster_activity_matrix_file)


        # Load Stimuli Onsets
        vis_1_odour_context_onsets  = np.load(base_directory + r"/Stimuli_Onsets/odour_context_stable_vis_1_frame_onsets.npy")
        vis_1_visual_context_onsets = np.load(base_directory + r"/Stimuli_Onsets/visual_context_stable_vis_1_frame_onsets.npy")
        vis_2_odour_context_onsets  = np.load(base_directory + r"/Stimuli_Onsets/odour_context_stable_vis_2_frame_onsets.npy")
        vis_2_visual_context_onsets = np.load(base_directory + r"/Stimuli_Onsets/visual_context_stable_vis_2_frame_onsets.npy")

        visual_context_onsets = np.concatenate([vis_1_visual_context_onsets, vis_2_visual_context_onsets])
        odour_context_onsets = np.concatenate([vis_1_odour_context_onsets, vis_2_odour_context_onsets])

        visual_context_correlation_tensor = create_correlation_tensor(activity_matrix, visual_context_onsets, trial_start, trial_stop)
        odour_context_correlation_tensor = create_correlation_tensor(activity_matrix, odour_context_onsets, trial_start, trial_stop)

        np.save(base_directory + "/Visual_Context_Correlation_Tensor.npy", visual_context_correlation_tensor)
        np.save(base_directory + "/Odour_Context_Correlation_Tensor.npy", odour_context_correlation_tensor)

        visual_context_correlation_tensor = np.load(base_directory + "/Visual_Context_Correlation_Tensor.npy")
        odour_context_correlation_tensor = np.load(base_directory + "/Odour_Context_Correlation_Tensor.npy")

        # View Mean Matricies
        get_cluster_centroids(base_directory, clusters_file)
        plot_correlation_matirices(visual_context_correlation_tensor, odour_context_correlation_tensor)
        get_significant_changes(base_directory, clusters_file, session_name)
        # score_list = decode_context_from_correlation_matrix(base_directory, clusters_file)
        # mean_score = np.mean(score_list)
        # mean_scores.append(mean_score)

    print("Mean  Scores", mean_scores)





def compare_response_correlations(session_list):

    trial_start = 0
    trial_stop = 40

    for base_directory in session_list:

        # Get Session Name
        session_name = base_directory.split("/")[-3]
        session_name = session_name + "Stimulus_Evoked_"

        # Load Activity Matrix
        cluster_activity_matrix_file = base_directory + "/Cluster_Activity_Matrix.npy"
        activity_matrix = np.load(cluster_activity_matrix_file)

        # Load Stimuli Onsets
        visual_context_onsets  = np.load(base_directory + r"/Stimuli_Onsets/odour_context_stable_vis_2_frame_onsets.npy")
        odour_context_onsets = np.load(base_directory + r"/Stimuli_Onsets/visual_context_stable_vis_2_frame_onsets.npy")

        # Create Correlation Tensors
        visual_context_correlation_tensor = create_correlation_tensor(activity_matrix, visual_context_onsets, trial_start, trial_stop)
        odour_context_correlation_tensor = create_correlation_tensor(activity_matrix, odour_context_onsets, trial_start, trial_stop)

        np.save(base_directory + "/Stimulus_Evoked_Visual_Context_Correlation_Tensor.npy", visual_context_correlation_tensor)
        np.save(base_directory + "/Stimulus-Evoked_Odour_Context_Correlation_Tensor.npy", odour_context_correlation_tensor)

        visual_context_correlation_tensor = np.load(base_directory + "/Stimulus_Evoked_Visual_Context_Correlation_Tensor.npy")
        odour_context_correlation_tensor = np.load(base_directory + "/Stimulus-Evoked_Odour_Context_Correlation_Tensor.npy")

        # View Mean Matricies
        get_cluster_centroids(base_directory, clusters_file)
        plot_correlation_matirices(visual_context_correlation_tensor, odour_context_correlation_tensor)
        get_significant_changes(base_directory, clusters_file, session_name, visual_context_correlation_tensor, odour_context_correlation_tensor)
        # score_list = decode_context_from_correlation_matrix(base_directory, clusters_file)
        # mean_score = np.mean(score_list)
        # mean_scores.append(mean_score)

    print("Mean  Scores", mean_scores)

clusters_file = "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/clean_clusters.npy"

session_list = ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging/"]


session_list = ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging"]

session_list = ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK12.1F/2021_09_22_Transition_Imaging"]

session_list = ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN78.1A/2020_12_09_Switching_Imaging"]

compare_pre_stimulus_correlations(session_list)