import numpy as np
import matplotlib.pyplot as plt
import cv2
import networkx as nx
import community as community_louvain
from sklearn.cluster import SpectralClustering, MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from skimage.morphology import  erosion, disk
from skimage import feature



def threshold_erosion(cluster, downsampled_indicies, downsampled_height, downsampled_width):

    # Create Original Image
    image = np.zeros((downsampled_height * downsampled_width))
    for pixel in cluster:
        pixel_index = downsampled_indicies[pixel]
        image[pixel_index] = 1

    image = np.ndarray.reshape(image, (downsampled_height, downsampled_width))

    # Erode
    footprint = disk(1)
    eroded = erosion(image, footprint)

    if np.max(eroded) > 0:
        return True
    else:
        return False


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



def visualise_clusters(base_directory, clusters):

    downsampled_indicies, downsampled_height, downsampled_width = downsample_mask(base_directory)

    #load_clusters
    number_of_clusters = len(clusters)
    print("numberf clusters", number_of_clusters)


    # View
    image = np.zeros((downsampled_height * downsampled_width))
    for cluster_index in range(number_of_clusters):
        #colour_value = float(cluster_index) / number_of_clusters
        colour_value = np.random.randint(low=0, high=10)
        cluster = clusters[cluster_index]
        for pixel in cluster:
            pixel_index = downsampled_indicies[pixel]

            #image[pixel_index] = colour_value

            image[pixel_index] = cluster_index

    image = np.ndarray.reshape(image, (downsampled_height, downsampled_width))
    plt.imshow(image, cmap='turbo') #gist_ncar
    plt.show()


def get_cluster_outlines(clusters, downsampled_indicies, downsampled_height, downsampled_width):


    outlines = np.zeros((downsampled_height, downsampled_width))

    for cluster in clusters:
        cluster_map = np.zeros(downsampled_height * downsampled_width)
        for pixel in cluster:
            pixel_index = downsampled_indicies[pixel]
            cluster_map[pixel_index] = 1
        cluster_map = np.ndarray.reshape(cluster_map, (downsampled_height, downsampled_width))

        edges = feature.canny(cluster_map, sigma=3)
        outlines = np.add(outlines, edges)

    binary_outlines = np.where(outlines>0, 1, 0)
    np.save("/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/clean_clusters_outline.npy", binary_outlines)
    plt.imshow(binary_outlines, cmap='binary', vmax=2, vmin=0)
    plt.show()



def plot_individual_clusters(clusters, output_directory, downsampled_indicies, downsampled_height, downsampled_width):

    count = 0
    for cluster in clusters:
        cluster_map = np.zeros(downsampled_height * downsampled_width)
        for pixel in cluster:
            pixel_index = downsampled_indicies[pixel]
            cluster_map[pixel_index] = 1
        cluster_map = np.ndarray.reshape(cluster_map, (downsampled_height, downsampled_width))

        plt.title(count)
        plt.imshow(cluster_map)
        plt.savefig(output_directory + str(count).zfill(3) + ".png")
        plt.close()
        count += 1



def clean_consensus_clusters(base_directory, clusters_file, save_directory):

    # Load Clusters
    clusters = np.load(clusters_file, allow_pickle=True)

    # Load Downsampled Mask
    downsampled_indicies, downsampled_height, downsampled_width = downsample_mask(base_directory)

    # Threshold Based on Size
    min_size = 20
    thresholded_clusters = []
    for cluster in clusters:
        size = len(cluster[0])
        if size > min_size:
            thresholded_clusters.append(cluster)

    # Threshold Based On Morpholgical Opening
    final_clusters = []
    for cluster in thresholded_clusters:
        verdict = threshold_erosion(cluster, downsampled_indicies, downsampled_height, downsampled_width)
        if verdict == True:
            final_clusters.append(cluster)

    # Save Clusters
    print("Unthresholded Clusters", len(clusters))
    print("Thresholded clusters", len(thresholded_clusters))
    print("Eroded Clusters", len(final_clusters))
    np.save(save_directory, final_clusters)

    # Visualise Clusters
    visualise_clusters(base_directory, final_clusters)

    # Visualise Individual Clusters
    output_directory = "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/Consensus_Clusters/"
    plot_individual_clusters(clusters, output_directory, downsampled_indicies, downsampled_height, downsampled_width)

    # Get Cluster Outlines
    get_cluster_outlines(clusters, downsampled_indicies, downsampled_height, downsampled_width)



base_directory = "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/"
clusters_file = "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/consensus_clusters.npy"
save_directory = "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/clean_clusters.npy"
