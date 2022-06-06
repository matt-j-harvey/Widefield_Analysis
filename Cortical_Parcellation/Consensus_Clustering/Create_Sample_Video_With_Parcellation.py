import numpy as np
import h5py
from scipy import ndimage
import matplotlib.pyplot as plt
import os
from skimage.feature import canny
from datetime import datetime
from matplotlib import cm
from scipy import ndimage
import cv2
from sklearn.decomposition import PCA


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



def create_image_from_data(data, indicies, image_height, image_width):
    template = np.zeros((image_height, image_width))
    data = np.nan_to_num(data)
    np.put(template, indicies, data)
    image = np.ndarray.reshape(template, (image_height, image_width))

    return image



def create_parcellation_example_video(base_directory, sample_size=10000, smoothing_window=3):

    # Load Clusters
    clusters = np.load("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Final_Consensus_Clusters.npy", allow_pickle=True)
    clusters = np.ndarray.astype(clusters, int)

    # Load Delta F Data
    data_file = os.path.join(base_directory, "Delta_F.hdf5")
    file_object = h5py.File(data_file, 'r')
    data_matrix = file_object["Data"]
    data_sample = data_matrix[2000:2000 + sample_size + smoothing_window]

    # Perform PCA Denosing
    data_sample = np.nan_to_num(data_sample)
    model = PCA(n_components=150)
    transformed_data = model.fit_transform(data_sample)
    data_sample = model.inverse_transform(transformed_data)

    # Load Alignment Dictionary
    alignment_dictionary = np.load(os.path.join(base_directory, "Cluster_Alignment_Dictionary.npy"), allow_pickle=True)[()]

    # Align Clusters
    aligned_clusters = transform_clusters(clusters, alignment_dictionary, invert=True)

    # Get Contours
    contours = get_contours(aligned_clusters)
    contour_indexes = np.nonzero(contours)

    # Load Mask
    indicies, image_height, image_width = load_generous_mask(base_directory)

    # Create Colourmaps
    cm = plt.cm.ScalarMappable(norm=None, cmap='inferno')
    colour_max = 0.7
    colour_min = 0.1
    cm.set_clim(vmin=colour_min, vmax=colour_max)

    # Create Open CV Video Object
    video_name = os.path.join(base_directory, "Activity_Video_With_Contours.avi")
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, video_codec, frameSize=(image_width, image_height), fps=30)  # 0, 12

    for frame_index in range(sample_size):

        # Get Smoothed Average
        frame_data = data_sample[frame_index:frame_index + smoothing_window]
        frame_data = np.mean(frame_data, axis=0)

        # Spatially Smooth This
        image = create_image_from_data(frame_data, indicies, image_height, image_width)
        image = ndimage.gaussian_filter(image, sigma=1)

        # Convert To RGBA
        image = cm.to_rgba(image)

        # Insert Contours
        image[contour_indexes] = [1,1,1,1]

        # Write To Open CV Video Object
        image = image * 255
        image = np.ndarray.astype(image, np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video.write(image)

    cv2.destroyAllWindows()
    video.release()





base_directory = r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_19_Discrimination_Imaging"
create_parcellation_example_video(base_directory)