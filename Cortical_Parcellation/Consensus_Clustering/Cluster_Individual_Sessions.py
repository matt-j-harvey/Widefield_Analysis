from sklearn.decomposition import NMF, TruncatedSVD
from skimage.transform import resize
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from skimage import feature
from datetime import datetime

def sort_matrix(matrix):

    # Cluster Matrix
    Z = ward(pdist(matrix))

    # Get Dendogram Leaf Order
    new_order = leaves_list(Z)

    # Sorted Matrix
    sorted_matrix = matrix[:, new_order][new_order]

    return sorted_matrix



def load_mask(base_directory):

    mask = np.load(os.path.join(base_directory, "Generous_Mask.npy"))

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask > 0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width


def create_image_from_data(data, indicies, image_height, image_width):
    template = np.zeros((image_height, image_width))
    data = np.nan_to_num(data)
    np.put(template, indicies, data)
    image = np.ndarray.reshape(template, (image_height, image_width))

    return image

def downsize_components(session, components, new_size=(300, 304)):

    # Load Mask
    indicies, image_height, image_width = load_mask(session)

    downsized_components = []
    for component in components:
        component_image = create_image_from_data(component,  indicies, image_height, image_width)
        component_image = resize(component_image, new_size)
        component_image = np.ndarray.flatten(component_image)
        downsized_components.append(component_image)

    downsized_components = np.array(downsized_components)
    return downsized_components


def create_connectivity_matrix_from_components(components):

    number_of_components, number_of_pixels = np.shape(components)


    connectivity_matrix = np.zeros((number_of_pixels, number_of_pixels))
    for component_index in range(number_of_components):
        component_data = components[component_index]
        component_data = np.divide(component_data, np.max(np.abs(component_data))) # Normalize
        component_contribution = np.outer(component_data, component_data)
        component_contribution = np.nan_to_num(component_contribution)
        connectivity_matrix = np.add(connectivity_matrix, component_contribution)


    """
    plt.imshow(connectivity_matrix)
    plt.show()
    sorted_matrix = sort_matrix(connectivity_matrix)
    plt.imshow(sorted_matrix)
    plt.show()
    """

    return connectivity_matrix


def perform_clustering(session):

    # Load Components
    spatial_components = np.load(os.path.join(session, "Blockwise_SVD_Spatial_Components.npy"))
    temporal_components = np.load(os.path.join(session, "Blockwise_SVD_Temporal_Components.npy"))

    # Get Random Sample
    number_of_timepoints, number_of_components = np.shape(temporal_components)
    sample_size = 5000
    sample_indicies = np.random.randint(low=0, high=number_of_timepoints - 1, size=sample_size)
    sample_indicies = list(sample_indicies)
    sample_indicies.sort()
    temporal_sample = temporal_components[sample_indicies]
    data_sample = np.dot(temporal_sample, spatial_components)
    data_sample = np.clip(data_sample, a_min=0, a_max=None)

    # Fit Model
    model = TruncatedSVD(n_components=30)
    # model = NMF(n_components=30)
    model.fit(data_sample)
    nmf_components = model.components_

    # Downsample Components
    downsample_size = (150, 152)
    downsized_components = downsize_components(session, nmf_components, new_size=downsample_size)

    # Create Connectivity Matrix
    connectivity_matrix = create_connectivity_matrix_from_components(downsized_components)

    # Cluster Connectivity Matrix
    print("Clustering")
    model = SpectralClustering(n_clusters=50, affinity='precomputed', n_components=30, verbose=True)  # assign_labels='discretize'

    # connectivity_matrix = np.divide(1, connectivity_matrix)
    # connectivity_matrix = np.nan_to_num(connectivity_matrix)
    # model = AgglomerativeClustering(n_clusters=20, affinity='precomputed', linkage='average')

    pixel_assignments = model.fit_predict(connectivity_matrix)
    pixel_assignments = np.reshape(pixel_assignments, downsample_size)

    # Save These
    np.save(os.path.join(session, "Pixel_Assignments.npy"), pixel_assignments)



def view_clusters(session):

    pixel_assignments = np.load(os.path.join(session, "Pixel_Assignments.npy"))
    pixel_assignments = resize(pixel_assignments, (600, 608), preserve_range=True)
    pixel_assignments = np.ndarray.astype(pixel_assignments, 'float32')

    # Get Edges
    edges = feature.canny(pixel_assignments, sigma=2)

    # Save
    np.save(os.path.join(session, "Pixel_Assignments.npy"), pixel_assignments)
    np.save(os.path.join(session, "Cluster_Edges.npy"), edges)

    """
    # View Clustering
    plt.imshow(pixel_assignments, cmap='flag')
    plt.show()
    
    # View Edges
    plt.imshow(edges)
    plt.show()
    """





"""
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_01_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_03_Discrimina-tion_Imaging",
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
]
"""

session_list = [
"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging",
"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_16_Discrimination_Imaging",
"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_17_Discrimination_Imaging",
"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_19_Discrimination_Imaging",
"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_21_Discrimination_Imaging",
"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_24_Discrimination_Imaging",
]


# Load SVD Components
for session in session_list:
    print("Session: ", session, datetime.now())
    perform_clustering(session)
    view_clusters(session)