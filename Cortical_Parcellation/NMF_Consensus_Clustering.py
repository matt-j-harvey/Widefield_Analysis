import numpy as np
import os
from sklearn.decomposition import NMF
from sklearn.cluster import AffinityPropagation, SpectralClustering
import matplotlib.pyplot as plt
import sys
import tables

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")
import Widefield_General_Functions


def scale_components(components):

    # Scale To Be Between 0 and 1
    scaled_components = []

    number_of_components = np.shape(components)[0]
    for component_index in range(number_of_components):
        component_data = components[component_index]
        component_data = np.subtract(component_data, np.min(component_data))
        component_data = np.divide(component_data, np.max(component_data))
        scaled_components.append(component_data)

    scaled_components = np.array(scaled_components)

    return scaled_components



def threshold_components(components):

    number_of_components = np.shape(components)[0]
    thresholded_components = []

    for component_index in range(number_of_components):

        component_data = components[component_index]

        threshold = np.percentile(component_data, 70)

        component_data = np.where(component_data > threshold, component_data, 0)

        thresholded_components.append(component_data)

    thresholded_components = np.array(thresholded_components)

    return thresholded_components


def assign_labels(component_data):

    pixel_max_values = np.max(component_data, axis=0)

    number_of_pixels = np.shape(component_data)[1]

    label_list = []
    for pixel_index in range(number_of_pixels):
        pixel_value_list = list(component_data[:, pixel_index])
        pixel_max_value = pixel_max_values[pixel_index]
        pixel_label = pixel_value_list.index(pixel_max_value)
        label_list.append(pixel_label)

    label_list = np.array(label_list)
    return label_list



def create_affinity_matrix(session_list, label_directory):

    # Get Number Of Sessions
    number_of_sessions = len(session_list)

    # Load Label Lists
    label_matrix = []
    for base_directory in session_list:

        # Get Session Name
        split_directory = base_directory.split('/')
        session_name = split_directory[-2] + "_" + split_directory[-1]

        # Load Label List
        label_list = np.load(os.path.join(label_directory, session_name + ".npy"))

        # Add To Matrix
        label_matrix.append(label_list)

    label_matrix = np.array(label_matrix)



    # Get Affinity Increment
    number_of_pixels = np.shape(label_matrix)[1]
    affinity_increment = 1

    # Create Tables File
    dimensions = (0, number_of_pixels)
    storage_path = "/home/matthew/Documents/NMF_Parcellation_Data/Affinity_Matrix.h5"
    storage_file = tables.open_file(storage_path, mode='w')
    storage_array = storage_file.create_earray(storage_file.root, 'data', tables.UInt8Atom(), shape=dimensions, expectedrows=number_of_pixels)

    # Create Affinity Matrix
    for pixel_index_1 in range(number_of_pixels):
        print("Pixel: ", pixel_index_1)
        pixel_1_affinity_vector = np.zeros(number_of_pixels)

        for pixel_index_2 in range(number_of_pixels):

            for session_index in range(number_of_sessions):
                if label_matrix[session_index, pixel_index_1] == label_matrix[session_index, pixel_index_2]:
                    pixel_1_affinity_vector[pixel_index_2] += affinity_increment

        storage_array.append([pixel_1_affinity_vector])
        storage_file.flush()




    # View Labels
    """
    figure_1 = plt.figure()
    rows = 1
    columns = number_of_sessions

    for session_index in range(number_of_sessions):

        session_labels = label_matrix[session_index]

        # Load Mask
        indicies, image_height, image_width = Widefield_General_Functions.load_mask(session_list[session_index])

        label_image = Widefield_General_Functions.create_image_from_data(session_labels, indicies, image_height, image_width)

        axis = figure_1.add_subplot(rows, columns, session_index + 1)
        axis.imshow(label_image, cmap='gist_rainbow')

    plt.show()
    """


def cluster_affinity_matrix():

    # Load Delta F Matrix
    delta_f_matrix_filepath = "/home/matthew/Documents/NMF_Parcellation_Data/Affinity_Matrix.h5"
    delta_f_matrix_container = tables.open_file(delta_f_matrix_filepath, mode='r')
    affinity_matrix = delta_f_matrix_container.root['data']
    affinity_matrix = np.array(affinity_matrix)

    #affinity_matrix = 8 - affinity_matrix


    model = SpectralClustering(affinity='precomputed')
    labels = model.fit_predict(affinity_matrix)

    np.save("/home/matthew/Documents/NMF_Parcellation_Data/Labels.npy", labels)






controls = [
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_02_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_08_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_23_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_31_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_15_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_10_29_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_11_05_Transition_Imaging"
            ]




mutants = [
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_10_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK24.1C/2021_11_10_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive1/Transition_2/NXAK20.1B/2021_11_22_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Transition_2/NXAK20.1B/2021_11_24_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Transition_2/NXAK20.1B/2021_11_26_Transition_Imaging",
]



save_directory = "/home/matthew/Documents/NMF_Parcellation_Data"

#cluster_affinity_matrix()
"""
# Load Data
tensor_list = []
number_of_components = 30
model = NMF(n_components=number_of_components)

for base_directory in controls:

    # Get Session Name
    split_directory = base_directory.split('/')
    session_name = split_directory[-2] + "_" + split_directory[-1]
    print("Decomposing: ", session_name)

    # Load Trial Average Data
    nmf_tensor = np.load(os.path.join(base_directory, "Activity_Tensors", "NMF_Segmentation_Activity_Tensor.npy"))

    # Perform NMF
    model.fit(nmf_tensor)
    component_list = model.components_

    # Scale Components
    component_list = scale_components(component_list)

    # Threshold Components
    component_list = threshold_components(component_list)

    # Assign Labels
    label_list = assign_labels(component_list)

    # Save Data
    np.save(os.path.join(save_directory, "NMF_Data", session_name), component_list)
    np.save(os.path.join(save_directory, "Session_Labels", session_name), label_list)
    """

#label_directory = os.path.join(save_directory, "Session_Labels")
#create_affinity_matrix(controls, label_directory)