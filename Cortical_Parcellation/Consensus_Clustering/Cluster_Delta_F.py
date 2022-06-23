import numpy as np
import h5py
from scipy import ndimage
import matplotlib.pyplot as plt
import os
from skimage.feature import canny
from datetime import datetime


def load_generous_mask(home_directory):

    # Loads the mask for a video, returns a list of which pixels are included, as well as the original image height and width
    mask = np.load(home_directory + "/Generous_Mask.npy")

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask>0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width


def get_chunk_structure(chunk_size, array_size):
    number_of_chunks = int(np.ceil(array_size / chunk_size))
    remainder = array_size % chunk_size

    # Get Chunk Sizes
    chunk_sizes = []
    if remainder == 0:
        for x in range(number_of_chunks):
            chunk_sizes.append(chunk_size)

    else:
        for x in range(number_of_chunks - 1):
            chunk_sizes.append(chunk_size)
        chunk_sizes.append(remainder)

    # Get Chunk Starts
    chunk_starts = []
    chunk_start = 0
    for chunk_index in range(number_of_chunks):
        chunk_starts.append(chunk_size * chunk_index)

    # Get Chunk Stops
    chunk_stops = []
    chunk_stop = 0
    for chunk_index in range(number_of_chunks):
        chunk_stop += chunk_sizes[chunk_index]
        chunk_stops.append(chunk_stop)

    return number_of_chunks, chunk_sizes, chunk_starts, chunk_stops




def transform_clusters(clusters, variable_dictionary, invert=False):

    # Unpack Dict
    angle = variable_dictionary['rotation']
    x_shift = variable_dictionary['x_shift']
    y_shift = variable_dictionary['y_shift']

    # Invert
    if invert == True:
        angle = -1 * angle
        x_shift = -1 * x_shift
        y_shift = -1 * y_shift

    transformed_clusters = np.zeros(np.shape(clusters))

    unique_clusters = list(np.unique(clusters))
    for cluster in unique_clusters:
        cluster_mask = np.where(clusters == cluster, 1, 0)
        cluster_mask = ndimage.rotate(cluster_mask, angle, reshape=False, prefilter=True)
        cluster_mask = np.roll(a=cluster_mask, axis=0, shift=y_shift)
        cluster_mask = np.roll(a=cluster_mask, axis=1, shift=x_shift)
        cluster_indicies = np.nonzero(cluster_mask)
        transformed_clusters[cluster_indicies] = cluster

    return transformed_clusters



def get_contours(pixel_assignments):

    unique_clusters = list(np.unique(pixel_assignments))
    smoothed_template = np.zeros(np.shape(pixel_assignments))

    for cluster in unique_clusters:

        cluster_mask = np.where(pixel_assignments == cluster, 1, 0)

        edges = canny(cluster_mask.astype('float32'), sigma=5)
        edge_indexes = np.nonzero(edges)
        smoothed_template[edge_indexes] = 1

    return smoothed_template


def get_cluster_sizes(clusters):

    unique_clusters = list(np.unique(clusters))
    cluster_size_list = []

    for cluster in unique_clusters:
        cluster_mask = np.where(clusters == cluster, 1, 0)
        cluster_size = np.sum(cluster_mask)
        cluster_size_list.append(cluster_size)

    return cluster_size_list



def cluster_delta_f(base_directory):

    # Load Clusters
    clusters = np.load("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Final_Consensus_Clusters.npy", allow_pickle=True)
    clusters = np.ndarray.astype(clusters, int)

    # Load Delta F Data
    data_file = os.path.join(base_directory, "Delta_F.hdf5")
    file_object = h5py.File(data_file, 'r')
    data_matrix = file_object["Data"]
    print("Data Matrix Shape", np.shape(data_matrix))

    # Load Alignment Dictionary
    alignment_dictionary = np.load(os.path.join(base_directory, "Cluster_Alignment_Dictionary.npy"), allow_pickle=True)[()]

    # Align Clusters
    aligned_clusters = transform_clusters(clusters, alignment_dictionary, invert=True)

    # Get Contours
    contours = get_contours(aligned_clusters)

    # Load Max Projection
    max_projection = np.load(os.path.join(base_directory, "max_projection.npy"))
    contour_indexes = np.nonzero(contours)
    max_projection[contour_indexes] = 0
    #plt.imshow(max_projection)
    #plt.show()

    # Load Mask
    indicies, image_height, image_width = load_generous_mask(base_directory)

    # Flatten Clusters
    flat_clusters = np.ndarray.flatten(clusters)

    # Get Cluster Sizes
    cluster_sizes = get_cluster_sizes(clusters)

    # Create Cluster Activity Matrix
    number_of_clusters = len(list(np.unique(clusters))) #We Dont Cluster 0, That is Background
    print("Number of clusters", number_of_clusters)
    number_of_datapoints, selected_pixels = np.shape(data_matrix)
    cluster_activity_matrix = np.zeros((number_of_datapoints, number_of_clusters))

    # Get Chunk Structure
    preffered_chunk_size = 10000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = get_chunk_structure(preffered_chunk_size, number_of_datapoints)

    # Cluster Each Chunk
    for chunk_index in range(number_of_chunks):
        print("Chunk: ", chunk_index, " of ", number_of_chunks, " at ", datetime.now())

        chunk_start = chunk_starts[chunk_index]
        chunk_stop = chunk_stops[chunk_index]
        chunk_data = data_matrix[chunk_start:chunk_stop]

        for pixel in range(selected_pixels):
            pixel_index = indicies[pixel]
            pixel_cluster = flat_clusters[pixel_index]

            if pixel_cluster != 0:
                cluster_size = cluster_sizes[pixel_cluster]
                pixel_data = chunk_data[:, pixel]
                cluster_activity_matrix[chunk_start:chunk_stop, pixel_cluster] += np.divide(pixel_data, cluster_size)

    # Save Matrix
    np.save(base_directory + "/Cluster_Activity_Matrix.npy", cluster_activity_matrix)


session_list = [

    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_01_Discrimination_Imaging",
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
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_24_Discrimination_Imaging",

    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging",

    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_04_29_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_01_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_03_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_05_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_07_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_09_Discrimination_Imaging",

    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_09_25_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_09_29_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A//2021_10_01_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_10_03_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_10_05_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_10_07_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging",

    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_15_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_16_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_17_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_19_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_23_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_25_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_14_Discrimination_Imaging",

    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_16_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_17_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_19_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_21_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_24_Discrimination_Imaging",

]



session_list = [
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging"]

session_list = [
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
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_03_03_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging",
]


session_list = [
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
]

session_list = [
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

session_list = [
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK22.1A/2021_11_05_Transition_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging",
]

session_list = [
    "/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_15_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_16_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_17_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_19_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_23_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_25_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_14_Discrimination_Imaging",
]

session_list = ["/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_21_Discrimination_Imaging"]


session_list = [
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_01_Discrimination_Imaging",
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
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_24_Discrimination_Imaging",
]

session_list = [
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_04_29_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_01_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_03_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_05_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_07_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_09_Discrimination_Imaging"
]

for session in session_list:
    print("Clustering Session: ", session)
    print("At: ", datetime.now())
    cluster_delta_f(session)
