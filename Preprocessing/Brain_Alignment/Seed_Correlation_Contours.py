import os
import tables
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage
from sklearn.decomposition import PCA
from datetime import datetime
from skimage.transform import resize

import Registration_Utils


def correlate_one_with_many(one, many):
    one = np.expand_dims(one, axis=1)
    print("One Shape", np.shape(one))
    print("many shape", np.shape(many))
    one = np.transpose(one)
    many = np.transpose(many)
    c = 1 - cdist(one, many, metric='correlation')[0]
    return c


def check_point_indicies(point_indicies, indicies, image_height, image_width):

    # Check Point Indicies
    checking_image = np.zeros((image_height * image_width))
    selected_indicies = indicies[point_indicies]
    checking_image[selected_indicies] = 1
    checking_image = np.reshape(checking_image, (image_height, image_width))
    plt.title("Checking Image")
    plt.imshow(checking_image)
    plt.show()


def get_point_indicies(indicies, image_height, image_width):

    # Create 2D Mask
    mask = np.zeros(image_height * image_width)
    mask[indicies] = 1
    mask = np.reshape(mask, (image_height, image_width))

    # Create Grid of Points
    point_grid = np.zeros((image_height, image_width))
    for y in range(0, image_height, 20):
        for x in range(0, image_width, 20):
            point_grid[y, x] = 1

    # Mask Grid of Points
    point_grid = np.multiply(point_grid, mask)

    # Get Point Coords
    point_grid = np.reshape(point_grid, (image_height * image_width))
    point_coords = np.nonzero(point_grid)

    # Create Map Of Indicies
    index_map = np.zeros(image_height * image_width)
    index_map[indicies] = list(range(len(indicies)))

    # Get indicies at point coords
    point_indicies = index_map[point_coords]
    point_indicies = np.array(point_indicies, dtype=int)

    return point_indicies

def downsample_delta_f(delta_f_matrix, indicies, image_height, image_width):


    # Get downsample
    template = np.zeros(image_height * image_width)
    template[indicies] = 1
    template = np.reshape(template, (image_height, image_width))
    template = resize(template, (100, 100), preserve_range=True)
    template = np.reshape(template, 100*100)
    templae_indicies = np.nonzero(template)

    print("downsampling delta F")
    downsampled_list = []
    number_of_frames = np.shape(delta_f_matrix)[0]
    for frame_index in tqdm(range(3000,number_of_frames)):
        frame = delta_f_matrix[frame_index]
        frame = np.nan_to_num(frame)
        frame = Registration_Utils.create_image_from_data(frame, indicies, image_height, image_width)
        frame = resize(frame, (100, 100), preserve_range=True)
        frame = np.reshape(frame, (100 * 100))
        frame = frame[templae_indicies]
        downsampled_list.append(frame)

    downsampled_list = np.array(downsampled_list)
    #downsampled_list = downsampled_list[3000:]
    return downsampled_list, templae_indicies



def denoise_delta_f(delta_f_matrix):
    model = PCA(n_components=200)
    inverse_data = model.fit_transform(delta_f_matrix)
    delta_f_matrix = model.inverse_transform(inverse_data)
    return delta_f_matrix


def get_seed_correlation_contours(base_directory):

    # Load Delta F File
    delta_f_file = os.path.join(base_directory, "Downsampled_Delta_F.h5")
    delta_f_file_container = tables.open_file(delta_f_file)
    delta_f_matrix = delta_f_file_container.root["Data"]
    print("Delta F Matrix Shape", np.shape(delta_f_matrix))

    # Load Mask
    indicies, image_height, image_width = Registration_Utils.load_downsampled_mask(base_directory)

    # Downsample
    delta_f_matrix, downsample_indicies = downsample_delta_f(delta_f_matrix, indicies, image_height, image_width)
    print("Downsampled Delta F", np.shape(delta_f_matrix))

    delta_f_matrix = denoise_delta_f(delta_f_matrix)

    # Get Correlation Matrix
    correlation_matrix = np.corrcoef(delta_f_matrix, rowvar=False)
    plt.imshow(correlation_matrix)
    plt.show()


    colourmap = Registration_Utils.get_musall_cmap()
    edge_map_list = []
    plt.ion()
    for pixel in tqdm(correlation_matrix):
        correlation_map = Registration_Utils.create_image_from_data(pixel, downsample_indicies, 100, 100)
        #correlation_map = ndimage.gaussian_filter(correlation_map, sigma=1)
        smoothed_correlation_map = ndimage.gaussian_filter(correlation_map, sigma=1)
        edge_map = np.subtract(correlation_map, smoothed_correlation_map)

        #plt.imshow(edge_map, vmin=-0.02, vmax=0.02, cmap=colourmap)
        #plt.draw()
        #plt.pause(0.1)
        #plt.clf()

        edge_map_list.append(edge_map)

    plt.ioff()
    edge_map_list = np.array(edge_map_list)
    edge_map_list = np.nan_to_num(edge_map_list)
    mean_edge_map = np.mean(edge_map_list, axis=0)
    plt.title("Mean Edge Map")
    plt.imshow(mean_edge_map)
    plt.show()

base_directory = r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_01_Continous_Retinotopy_Left"
get_seed_correlation_contours(base_directory)