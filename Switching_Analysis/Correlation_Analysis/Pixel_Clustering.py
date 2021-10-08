import numpy as np
import h5py
import tables
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
import sys
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import TruncatedSVD
from matplotlib import cm

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")
import Widefield_General_Functions


def downsample_widefield_data(widefield_data_file, base_directory):

    # Load Widefield Data
    widefield_data_object = tables.open_file(base_directory + "/" + widefield_data_file, mode='r')
    widefield_data = widefield_data_object.root['Data']

    # Load Mask
    full_indicies, full_height, full_width = Widefield_General_Functions.load_mask(base_directory)

    # Downsample Mask
    downsampled_indicies, downsampled_height, downsampled_width = downsample_mask(base_directory)

    # Create Output File
    output_file = base_directory + "/Downsampled_Delta_F.hdf5"
    number_of_frames = np.shape(widefield_data)[0]
    number_of_downsampled_active_pixels = np.shape(downsampled_indicies)[0]
    print("Number Of Downsampled Active Pixels", number_of_downsampled_active_pixels)

    # Define Chunking Settings
    preferred_chunk_size = 20000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = Widefield_General_Functions.get_chunk_structure(preferred_chunk_size, number_of_frames)

    with h5py.File(output_file, "w") as f:
        dataset = f.create_dataset("Data", (number_of_downsampled_active_pixels, number_of_frames), dtype=np.float32, chunks=True, compression="gzip")

        for chunk_index in range(number_of_chunks):
            print("Chunk:", str(chunk_index).zfill(2), "of", number_of_chunks)
            chunk_start = int(chunk_starts[chunk_index])
            chunk_stop = int(chunk_stops[chunk_index])

            chunk_data = widefield_data[chunk_start:chunk_stop]
            chunk_data = process_chunk(chunk_data, full_indicies, full_height, full_width,  downsampled_indicies, downsampled_height, downsampled_width)
            chunk_data = np.transpose(chunk_data)
            dataset[:, chunk_start:chunk_stop] = chunk_data


def process_chunk(data, full_indicies, full_height, full_width, downsampled_indicies, downsampled_height, downsampled_width):

    processed_data = []
    for frame_data in data:

        # Recreate Full Image
        frame_data = Widefield_General_Functions.create_image_from_data(frame_data, full_indicies, full_height, full_width)

        # Smooth With 2D Gaussian
        frame_data = ndimage.gaussian_filter(frame_data, sigma=2)

        #  Downsample Image
        frame_data = cv2.resize(frame_data, dsize=(downsampled_width, downsampled_height))

        # Flatten Array
        frame_data = np.ndarray.flatten(frame_data)

        # Take Only Masked Indicies
        active_pixels = frame_data[downsampled_indicies]

        # Add To List
        processed_data.append(active_pixels)

    processed_data = np.array(processed_data)
    return processed_data



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



def compute_distance_matrix(base_directory):

    # Load Downsampled Data
    downsampled_data_file = base_directory + "Downsampled_Delta_F.hdf5"
    downsampled_file_object = h5py.File(downsampled_data_file, 'r')
    data_matrix = downsampled_file_object["Data"]
    data_matrix = np.array(data_matrix)

    #  Compute Distance Matrix
    number_of_pixels = np.shape(data_matrix)[0]
    distance_matrix = np.zeros((number_of_pixels, number_of_pixels))
    for pixel_1 in range(number_of_pixels):
        print("Pixel ", pixel_1, " of ", number_of_pixels)
        pixel_1_trace = data_matrix[pixel_1]
        for pixel_2 in range(pixel_1 + 1, number_of_pixels):
            pixel_2_trace = data_matrix[pixel_2]

            # Calculate Euclidean Distance Between Activity Vectors
            distance = np.linalg.norm(pixel_1_trace - pixel_2_trace)

            distance_matrix[pixel_1][pixel_2] = distance
            distance_matrix[pixel_2][pixel_1] = distance

    np.save(base_directory + "/Distance_matrix.npy", distance_matrix)


def convert_distancce_to_simmilarity_matrix(base_directory, rbf=0.1):

    # Load Distance Matrix
    distance_matrix_file = base_directory + "/Distance_matrix.npy"
    distance_matrix = np.load(distance_matrix_file)
    print("Max distance", np.max(distance_matrix))
    print("Min Distance", np.min(distance_matrix))

    # Create Simmilarity Matrix
    simmilarity_matrix = np.exp(-rbf * distance_matrix)
    print("Simmiliarty max", np.max(simmilarity_matrix))
    print("Simmilairty min", np.min(simmilarity_matrix))

    # Set Zero Diagonal
    np.fill_diagonal(simmilarity_matrix, 0)

    # Save Simmilairty Matrix
    np.save(base_directory + "/Simmilarity_matrix.npy", simmilarity_matrix)




def normalise_affinity_matrix(base_directory):

    # Load matrix
    matrix = np.load(base_directory + "/Simmilarity_matrix.npy")

    # Get Row Sum
    d = np.sum(matrix, axis=1)

    # Convert This Into A Diagonal Matrix
    d = np.diag(d)

    # Invert This
    d = np.linalg.inv(d)

    # Multiply By Original Matrix
    matrix = np.dot(d, matrix)

    # Save Output
    np.save(base_directory + "/Normalised_Affinity_Matrix.npy", matrix)




def perform_svd(base_directory, number_of_components=200):

    # Load Normalised Affinity Matrix
    matrix = np.load(base_directory + "/Normalised_Affinity_Matrix.npy")

    # Perform SVD
    model = TruncatedSVD(n_components=number_of_components)
    model.fit(matrix)

    u = model.components_
    s = model.singular_values_

    # Save Output
    np.save(base_directory + "/U.npy", u)
    np.save(base_directory + "/S.npy", s)


def visualise_svd_components(base_directory):

    components = np.load(base_directory + "/U.npy")
    print("Components shape", np.shape(components))

    # Downsample Mask
    downsampled_indicies, downsampled_height, downsampled_width = downsample_mask(base_directory)

    # View
    for x in range(10):
        image = np.zeros((downsampled_height * downsampled_width))
        image[downsampled_indicies] = components[x]
        image = np.ndarray.reshape(image, (downsampled_height, downsampled_width))

        plt.title("Component " + str(x))
        plt.imshow(image, cmap='plasma')
        plt.show()



def perform_clustering(base_directory):

    # Settings:
    stop_fraction = 0.05
    number_of_vectors_to_use = 50

    u = np.load(base_directory + "/U.npy")
    number_of_pixels = np.shape(u)[1]
    #print("Number of pixels", number_of_pixels)

    # Compute Spectral Embedding Norm of Each Pixel
    embedding_norm_list = []
    for pixel in range(number_of_pixels):
        embedding_vector = u[:, pixel]
        embedding_norm = np.linalg.norm(embedding_vector)
        embedding_norm_list.append(embedding_norm)
    #print("Embedding Norms", embedding_norm_list)

    # Create List Of Pixel Indicies Sorted By Embedding Norm
    sorted_pixel_list = []
    sorted_embedding_norm_list = sorted(embedding_norm_list, reverse=True)
    #print("Sorted embedding norm list", sorted_embedding_norm_list)
    for norm in sorted_embedding_norm_list:
        sorted_pixel_list.append(embedding_norm_list.index(norm))
    #print("Pixels Sorted By Embedding Norm", sorted_pixel_list)

    origin_vector = np.zeros(number_of_vectors_to_use)
    converged = False
    clusters = []
    print("Clustering")
    while not converged:

        # Take Remaning Pixel With Largest Embedding Norm
        key_pixel = sorted_pixel_list[0]
        #print("Key Pixel: ", key_pixel)

        # Take Top N Axes
        key_pixel_embedding = u[:, key_pixel]
        #print("Key Pixel Embeddings: ", key_pixel_embedding)

        sorted_k_pixel_embedding = sorted(key_pixel_embedding, reverse=True)
        #print("Key Pixel Embeddings Sorted: ", sorted_k_pixel_embedding)

        largest_n_axes_values = sorted_k_pixel_embedding[0:number_of_vectors_to_use]
        #print("LArgest N Values", largest_n_axes_values)
        selected_axes = []
        for axis_value in largest_n_axes_values:
            selected_axes.append(list(key_pixel_embedding).index(axis_value))
        #print("Selected Axes ", selected_axes)

        # Get Position Of Key Pixel In Selected Axes
        key_pixel_embedding_in_selected_axes = u[selected_axes, key_pixel]

        # Iterate Through Other Pixels and Get Distances
        pixels_in_this_cluster = []
        for pixel in sorted_pixel_list:
            pixel_embedding_in_selected_axes = u[selected_axes, pixel]
            distance_to_origin = np.linalg.norm(pixel_embedding_in_selected_axes - origin_vector)
            distance_to_key_pixel = np.linalg.norm(
                pixel_embedding_in_selected_axes - key_pixel_embedding_in_selected_axes)

            if distance_to_key_pixel < distance_to_origin:
                pixels_in_this_cluster.append(pixel)

        #print("Pixels in this cluster", pixels_in_this_cluster)
        clusters.append(pixels_in_this_cluster)

        # Remove These Pixels From Sorted Pixel List
        for clustered_pixel in pixels_in_this_cluster:
            sorted_pixel_list.remove(clustered_pixel)

        print("Percentage Remaining Unclustered: ", float(len(sorted_pixel_list))/number_of_pixels)
        if len(sorted_pixel_list) < number_of_pixels * stop_fraction:
            converged = True

        #print(clusters)

    np.save(base_directory + "/Clusters.npy", clusters)


def visualise_clusters(base_directory):

    #load_clusters
    clusters = np.load(base_directory + "/Clusters.npy", allow_pickle=True)
    print(clusters)
    number_of_clusters = len(clusters)
    print("numberf clusters", number_of_clusters)

    # Downsample Mask
    downsampled_indicies, downsampled_height, downsampled_width = downsample_mask(base_directory)

    # Get Colourmap
    colour_map = cm.get_cmap('gist_rainbow')

    # View
    image = np.zeros((downsampled_height * downsampled_width))
    for cluster_index in range(number_of_clusters):
        #colour_value = float(cluster_index) / number_of_clusters
        colour_value = np.random.randint(low=0, high=10)
        cluster = clusters[cluster_index]
        for pixel in cluster:
            pixel_index = downsampled_indicies[pixel]
            image[pixel_index] = colour_value

    image = np.ndarray.reshape(image, (downsampled_height, downsampled_width))
    plt.imshow(image, cmap='tab10')
    plt.show()


    for cluster_index in range(number_of_clusters):
        image = np.zeros((downsampled_height * downsampled_width))
        cluster = clusters[cluster_index]
        image[downsampled_indicies] = 1
        for pixel in cluster:
            pixel_index = downsampled_indicies[pixel]
            image[pixel_index] = 2
        image = np.ndarray.reshape(image, (downsampled_height, downsampled_width))
        plt.imshow(image,  cmap='tab10')
        plt.show()



def perform_pixel_clustering(base_directory):

    # Downsample Data Matrix
    downsample_widefield_data("Delta_F.h5", base_directory)

    # Compute Distance Matrix
    compute_distance_matrix(base_directory)

    # Convert Distance To Simmilairty Matrix
    convert_distancce_to_simmilarity_matrix(base_directory)

    # Normalise Simmilairty Matrix
    normalise_affinity_matrix(base_directory)

    #  Perform SVD
    perform_svd(base_directory)

    # Perform Clustering
    perform_clustering(base_directory)

    # Visualise Clusters
    visualise_clusters(base_directory)



base_directory ="/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Switching_Analysis/Selected_sessions/NXAK4.1B/2021_03_04_Switching_Imaging/"
perform_pixel_clustering(base_directory)
