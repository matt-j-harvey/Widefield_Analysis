import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import networkx as nx
import cv2
from matplotlib.pyplot import cm
from matplotlib.colors import to_rgb
import os
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist, cdist
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm

from ..Widefield_Utils import widefield_utils

import Create_Activity_Tensor
import Noise_Correlation_Utils


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





def concatenate_and_subtract_mean(tensor):

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

    return concatenated_subtracted_tensor



def correlate_one_with_many(one, many):

    c = 1 - cdist(one, many, metric='correlation')[0]
    print(c)
    plt.plot(one)
    plt.plot(many)
    plt.show()


    return c



def get_correlation_matrix(region_trace, activity_trace):

    print("Correlating")
    number_of_regions = np.shape(activity_trace)[0]
    print("Number Of Regions")

    """
    print(region_trace)
    plt.plot(region_trace)
    plt.show()
    """

    correlation_vector = []
    for region_index in range(number_of_regions):
        seed_region_trace = activity_trace[region_index]

        print("seed region trae", np.shape(seed_region_trace))

        correlation_coefficient = np.corrcoef(region_trace, seed_region_trace)
        print("Correlation Coefficient", correlation_coefficient)

        """
        plt.plot(region_trace)
        plt.plot(seed_region_trace, c='r')
        plt.show()
        """
        correlation_vector.append(correlation_coefficient[0, 1])

    return correlation_vector





def analyse_noise_correlations(context_1_activity_tensor_list, context_2_activity_tensor_list, context_1_region_tensor_list, context_2_region_tensor_list):

    # Concatenate And Subtract Mean For Each Tensor
    context_1_noise_activity_tensor_list = []
    for tensor in context_1_activity_tensor_list:
        tensor = concatenate_and_subtract_mean(tensor)
        context_1_noise_activity_tensor_list.append(tensor)

    context_1_noise_region_tensor_list = []
    for tensor in context_1_region_tensor_list:
        tensor = concatenate_and_subtract_mean_region(tensor)
        context_1_noise_region_tensor_list.append(tensor)

    context_2_noise_activity_tensor_list = []
    for tensor in context_2_activity_tensor_list:
        tensor = concatenate_and_subtract_mean(tensor)
        context_2_noise_activity_tensor_list.append(tensor)

    context_2_noise_region_tensor_list = []
    for tensor in context_2_region_tensor_list:
        tensor = concatenate_and_subtract_mean_region(tensor)
        context_2_noise_region_tensor_list.append(tensor)


    """
    print("COmapring Region Tensors")
    for tensor in context_1_noise_region_tensor_list:
        plt.plot(tensor)
        plt.show()
    """



    context_1_concatenated_activity_tensor  = np.hstack(context_1_noise_activity_tensor_list)
    context_1_concatenated_region_tensor    = np.hstack(context_1_noise_region_tensor_list)
    context_2_concatenated_activity_tensor  = np.hstack(context_2_noise_activity_tensor_list)
    context_2_concatenated_region_tensor    = np.hstack(context_2_noise_region_tensor_list)

    print("Context 1 Concat Activity Tensor", np.shape(context_1_concatenated_activity_tensor))
    print("Context 1 Concat Region Tensor", np.shape(context_1_concatenated_region_tensor))
    print("Context 2 Concat Activity Tensor", np.shape(context_2_concatenated_activity_tensor))
    print("Context 2 Concat Region Tensor", np.shape(context_2_concatenated_region_tensor))

    """
    print("Plotting Region Tensors")
    plt.plot(context_1_concatenated_region_tensor)
    plt.plot(context_2_concatenated_region_tensor)
    plt.show()
    """

    # Get Correlation Map
    context_1_correlation_map = get_correlation_matrix(context_1_concatenated_region_tensor, context_1_concatenated_activity_tensor)
    context_2_correlation_map = get_correlation_matrix(context_2_concatenated_region_tensor, context_2_concatenated_activity_tensor)

    return context_1_correlation_map, context_2_correlation_map


def downsample_mask_further(indicies, image_height, image_width, downsample_size=100):
    template = np.zeros((image_height*image_width))
    template[indicies] = 1
    template = np.reshape(template, (image_height, image_width))
    template = resize(template, (downsample_size, downsample_size), anti_aliasing=True)
    template = np.reshape(template, (downsample_size * downsample_size))
    template = np.where(template > 0.5, 1, 0)
    template_indicies = np.nonzero(template)
    return template_indicies, downsample_size, downsample_size


def downsample_tensor(activity_tensor, full_indicies, full_image_height, full_image_width, downsample_indices, downsample_height, downsample_width):

    downsampled_tensor = []
    for trial in activity_tensor:
        downsampled_trial = []
        for frame in trial:
            frame = Noise_Correlation_Utils.create_image_from_data(frame, full_indicies, full_image_height, full_image_width)
            frame = resize(frame, (downsample_height, downsample_width))
            frame = np.reshape(frame, (downsample_height * downsample_width))
            frame = frame[downsample_indices]
            downsampled_trial.append(frame)
        downsampled_tensor.append(downsampled_trial)

    return downsampled_tensor


def analyse_noise_correlations(base_directory, onset_files, tensor_names, start_window, stop_window, tensor_save_directory):

    # Get File Structure
    split_base_directory = Path(base_directory).parts
    mouse_name = split_base_directory[-2]
    session_name = split_base_directory[-1]

    colourmap = Noise_Correlation_Utils.get_musall_cmap()

    # Get Data Structure
    number_of_conditions = len(onset_files)

    # Load Combined Mask
    indicies, image_height, image_width = Noise_Correlation_Utils.load_tight_mask()

    # Downsample Further
    downsample_indicies, downsample_height, downsample_width = downsample_mask_further(indicies, image_height, image_width)

    correlation_matrix_list = []
    for condition_index in range(number_of_conditions):

        # Create Activity Tensor
        #Create_Activity_Tensor.create_activity_tensor(base_directory, onset_files[condition_index], start_window, stop_window, tensor_save_directory, start_cutoff=3000)

        # Load Activity Tensor
        activity_tensor = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, tensor_names[condition_index] + "_Activity_Tensor.npy"))

        # Concatenate and Subtract Mean
        activity_tensor = downsample_tensor(activity_tensor,  indicies, image_height, image_width, downsample_indicies, downsample_height, downsample_width)
        activity_tensor = concatenate_and_subtract_mean(activity_tensor)

        # Calculate Correlation Matrix
        correlation_matrix = np.corrcoef(activity_tensor, rowvar=False)
        correlation_matrix_list.append(correlation_matrix)

        # Save Correlation Matrix
        np.save(os.path.join(tensor_save_directory, mouse_name, session_name, tensor_names[condition_index] + "_Noise_Correlation_Matrix.npy"), correlation_matrix)




    correlation_modulation = np.subtract(correlation_matrix_list[0], correlation_matrix_list[1])

    #correlation_modulation_image = Noise_Correlation_Utils.create_image_from_data(correlation_modulation, downsample_indicies, downsample_height, downsample_width)
    #plt.imshow(correlation_modulation_image, cmap=colourmap, vmin=-0.5, vmax=0.5)
    #plt.show()

    plt.title(mouse_name + "_" + session_name + "_Modulation")
    plt.imshow(correlation_modulation, cmap=colourmap, vmin=-0.5, vmax=0.5)
    plt.savefig(os.path.join(tensor_save_directory, mouse_name + "_" + session_name + "_Noise_Modulation.png"))
    plt.close()


session_list = [

                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_20_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_22_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_24_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_14_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_16_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_18_Transition_Imaging",

                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_17_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_19_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_23_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_30_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_06_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_08_Transition_Imaging",

                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_15_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_17_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_19_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_22_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_24_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_26_Transition_Imaging",

                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_14_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_20_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_26_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_05_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_08_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_10_Transition_Imaging",

                "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_28_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_05_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_09_Switching_Imaging",

                "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_29_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_05_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging",

                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_02_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_04_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_06_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_02_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_08_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_10_Transition_Imaging",

                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_26_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_28_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_02_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_23_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_31_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_04_02_Transition_Imaging",

                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_21_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_23_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_11_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_13_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_15_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_17_Transition_Imaging",

                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_14_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_20_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_22_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_29_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_03_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_05_Transition_Imaging",

                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_31_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_02_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_04_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_08_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_10_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_12_Transition_Imaging",

                "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_13_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_15_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_17_Switching_Imaging",
]


# Get Analysis Details
analysis_name = "Unrewarded_Contextual_Modulation"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Noise_Correlation_Utils.load_analysis_container(analysis_name)
tensor_save_directory = r"/media/matthew/29D46574463D2856/Widefield_Analysis/Noise_Correlation_Analysis/Activity_Tensors"
for base_directory in tqdm(session_list):
    print("Analysing Session: ", base_directory)
    analyse_noise_correlations(base_directory, onset_files, tensor_names, start_window, stop_window, tensor_save_directory)
