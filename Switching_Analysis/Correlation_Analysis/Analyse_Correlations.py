import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import networkx as nx
import cv2
from matplotlib.pyplot import cm

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


def get_significant_changes(base_directory):

    # Load Tensors
    visual_context_correlation_tensor = np.load(base_directory + "/Visual_Context_Correlation_Tensor.npy")
    odour_context_correlation_tensor = np.load(base_directory + "/Odour_Context_Correlation_Tensor.npy")

    # Compute Signficant Differences
    t_statistics, p_values = stats.ttest_ind(a=visual_context_correlation_tensor, b=odour_context_correlation_tensor, axis=0)

    p_value_list = np.copy(p_values)
    p_value_list = np.ndarray.flatten(p_value_list)
    p_value_list = list(p_value_list)
    p_value_list.sort()
    print("P Value List", p_value_list)

    # Show Signficant Changes
    signficance_map = np.where(p_values < 0.01, t_statistics, 0)
    print("P Vlaues", )

    magnitue = np.max(np.abs(t_statistics))
    plt.imshow(signficance_map, cmap='bwr', vmin=-2, vmax=2)
    plt.show()

    draw_modulation_map(base_directory, np.abs(signficance_map))
    draw_brain_network(base_directory, np.abs(signficance_map))


def decode_context_from_correlation_matrix(base_directory):

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
    draw_modulation_map(base_directory, coefficient_matrix)
    print("Score List", score_list)




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

def get_cluster_centroids(base_directory):

    # Downsample Mask
    downsampled_indicies, downsampled_height, downsampled_width = downsample_mask(base_directory)

    # Load Clusters
    clusters = np.load(base_directory + "/Clusters.npy", allow_pickle=True)
    number_of_clusters = len(clusters)

    # Get Cluster Centroids
    cluster_centroids = np.zeros((number_of_clusters, 2))
    for cluster_index in range(number_of_clusters):
        cluster = clusters[cluster_index]
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
    plt.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1])
    plt.show()



def draw_modulation_map(base_directory, adjacency_matrix):

    # Downsample Mask
    downsampled_indicies, downsampled_height, downsampled_width = downsample_mask(base_directory)

    # Create Regional Modulation Vector
    regional_modulation = np.mean(adjacency_matrix, axis=1)

    # Load Clusters
    clusters = np.load(base_directory + "/Clusters.npy", allow_pickle=True)
    number_of_clusters = len(clusters)

    # Draw Brain Mask
    image = np.zeros((downsampled_height * downsampled_width))
    for cluster_index in range(number_of_clusters):
        cluster = clusters[cluster_index]
        for pixel in cluster:
            pixel_index = downsampled_indicies[pixel]
            image[pixel_index] = regional_modulation[cluster_index]

    image = np.ndarray.reshape(image, (downsampled_height, downsampled_width))

    plt.imshow(image, cmap='jet')
    plt.show()


def draw_brain_network(base_directory, adjacency_matrix):

    # Load Cluster Centroids
    cluster_centroids = np.load(base_directory + "/Cluster_Centroids.npy")

    # Create NetworkX Graph
    graph = nx.from_numpy_matrix(adjacency_matrix)

    # Get Edge Weights
    edges = graph.edges()
    weights = [graph[u][v]['weight'] for u, v in edges]
    weights = np.divide(weights, np.max(weights))

    # Get Edge Colours
    print("Weights", np.shape(weights))
    colourmap = cm.get_cmap('plasma')
    colours = []
    for weight in weights:
        colour = colourmap(weight)
        colours.append(colour)

    # Draw Graph
    nx.draw(graph, pos=cluster_centroids, node_size=1,  width=weights, edge_color=colours)

    plt.show()



base_directory ="/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Switching_Analysis/Selected_sessions/NXAK4.1B/2021_03_04_Switching_Imaging/"

# Load Activity Matrix
"""
cluster_activity_matrix_file = base_directory + "/Cluster_Activity_Matrix.npy"
activity_matrix = np.load(cluster_activity_matrix_file)

# Load Stimuli Onsets
vis_1_odour_context_onsets  = np.load(base_directory + r"/Stimuli_Onsets/odour_context_stable_vis_1_frame_onsets.npy")
vis_1_visual_context_onsets = np.load(base_directory + r"/Stimuli_Onsets/visual_context_stable_vis_1_frame_onsets.npy")
vis_2_odour_context_onsets  = np.load(base_directory + r"/Stimuli_Onsets/odour_context_stable_vis_2_frame_onsets.npy")
vis_2_visual_context_onsets = np.load(base_directory + r"/Stimuli_Onsets/visual_context_stable_vis_2_frame_onsets.npy")

visual_context_onsets = np.concatenate([vis_1_visual_context_onsets, vis_2_visual_context_onsets])
odour_context_onsets = np.concatenate([vis_1_odour_context_onsets, vis_2_odour_context_onsets])

# Create Correlation Tensors
number_of_visual_context_trials = np.shape(visual_context_onsets)[0]
number_of_odour_context_trials = np.shape(odour_context_onsets)[0]

visual_context_correlation_tensor = create_correlation_tensor(activity_matrix, visual_context_onsets, -75, 0)
odour_context_correlation_tensor = create_correlation_tensor(activity_matrix, odour_context_onsets, -75, 0)

np.save(base_directory + "/Visual_Context_Correlation_Tensor.npy", visual_context_correlation_tensor)
np.save(base_directory + "/Odour_Context_Correlation_Tensor.npy", odour_context_correlation_tensor)

"""


visual_context_correlation_tensor =  np.load(base_directory + "/Visual_Context_Correlation_Tensor.npy")
odour_context_correlation_tensor = np.load(base_directory + "/Odour_Context_Correlation_Tensor.npy")

# View Mean Matricies
#plot_correlation_matirices(visual_context_correlation_tensor, odour_context_correlation_tensor)

#get_significant_changes(base_directory)
decode_context_from_correlation_matrix(base_directory)

#get_cluster_centroids(base_directory)
