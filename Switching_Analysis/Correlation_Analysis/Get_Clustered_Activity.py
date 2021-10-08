import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt


def convert_downsampled_delta_f_to_cluster_traces(base_directory):

    # Load Clusters
    clusters = np.load(base_directory + "/Clusters.npy", allow_pickle=True)

    # Load Delta F Data
    downsampled_data_file = base_directory + "/Downsampled_Delta_F.hdf5"
    downsampled_file_object = h5py.File(downsampled_data_file, 'r')
    data_matrix = downsampled_file_object["Data"]

    # Create Cluster Activity Matrix
    number_of_clusters = len(clusters)
    number_of_datapoints = np.shape(data_matrix)[1]
    cluster_activity_matrix = np.zeros((number_of_clusters, number_of_datapoints))

    for cluster_index in range(number_of_clusters):
        print("cluster: ", cluster_index, " of ", number_of_clusters)
        cluster_pixels = clusters[cluster_index]
        cluster_pixels.sort()
        pixel_traces = data_matrix[cluster_pixels]
        region_trace = np.mean(pixel_traces, axis=0)
        cluster_activity_matrix[cluster_index] = region_trace

    # Save Matrix
    np.save(base_directory + "/Cluster_Activity_Matrix.npy", cluster_activity_matrix)

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

def view_cluster_activity_matrix(base_directory):

    cluster_activity_matrix = np.load(base_directory + "/Cluster_Activity_Matrix.npy")
    print("Clusterctivity mtrxi shape", np.shape(cluster_activity_matrix))


    # Downsample Mask
    downsampled_indicies, downsampled_height, downsampled_width = downsample_mask(base_directory)

    # Load Clusters
    clusters = np.load(base_directory + "/Clusters.npy", allow_pickle=True)

    plt.ion()
    number_of_timepoints = np.shape(cluster_activity_matrix)[1]
    number_of_clusters = len(clusters)

    for timepoint in range(number_of_timepoints):
        image = np.zeros((downsampled_height * downsampled_width))
        activity = cluster_activity_matrix[:, timepoint]

        for cluster_index in range(number_of_clusters):
            cluster = clusters[cluster_index]
            for pixel in cluster:
                pixel_index = downsampled_indicies[pixel]
                image[pixel_index] = activity[cluster_index]

        image = np.ndarray.reshape(image, (downsampled_height, downsampled_width))
        plt.imshow(image, cmap='inferno', vmin=0, vmax=1)
        plt.draw()
        plt.pause(0.1)
        plt.clf()



base_directory ="/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Switching_Analysis/Selected_sessions/NXAK4.1B/2021_03_04_Switching_Imaging/"
view_cluster_activity_matrix(base_directory)