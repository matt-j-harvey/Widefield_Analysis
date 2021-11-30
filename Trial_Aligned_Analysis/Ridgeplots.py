import plotly.graph_objects as go
import scipy.ndimage
from plotly.colors import n_colors
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy import stats
np.random.seed(1)


def plot_ridgeplot(data):

    print(np.shape(data))
    number_of_rois = np.shape(data)[0]

    """
    # 12 sets of normal distributed random data, with increasing mean and standard deviation
    data = (np.linspace(1, 2, 12)[:, np.newaxis] * np.random.randn(12, 200) +
                (np.arange(12) + 2 * np.random.random(12))[:, np.newaxis])
    """

    colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', number_of_rois, colortype='rgb')

    fig = go.Figure()
    for data_line, color in zip(data, colors):
        fig.add_trace(go.Violin(x=data_line, line_color=color, opacity=0.1))

    fig.update_traces(orientation='h', side='positive', width=60, points=False)
    fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=True)
    fig.show()


def sort_modulation(modulation):

    mean_vector = np.mean(modulation, axis=1)
    sorted_mean_vector = np.copy(mean_vector)

    mean_vector = list(mean_vector)
    sorted_mean_vector = list(sorted_mean_vector)
    sorted_mean_vector.sort()

    new_order = []
    for item in sorted_mean_vector:
        index = mean_vector.index(item)
        new_order.append(index)

    sorted_modualtion = []
    for row in new_order:
        sorted_modualtion.append(modulation[row])
    sorted_modualtion = np.array(sorted_modualtion)

    return sorted_modualtion


def get_mean_modulation_plot(session_list):

    modulation_matricies = []

    for base_directory in session_list:
        #modulation_file_location = os.path.join(base_directory, "Pre_Stimulus", "Concatenated_Modulation.npy")
        modulation_file_location = os.path.join(base_directory, "Noise_Correlations", "Noise_Correlation_Delta_Matrix.npy")
        modulation_matrix = np.load(modulation_file_location)
        modulation_matrix = np.multiply(-1, modulation_matrix)
        modulation_matricies.append(modulation_matrix)

    modulation_matricies = np.stack(modulation_matricies)
    print("Modulation Matricies", np.shape(modulation_matricies))
    mean_modulation_matrix = np.mean(modulation_matricies, axis=0)

    return mean_modulation_matrix


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



def visualise_clusters(base_directory, vector):

    #load_clusters
    clusters = np.load("/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/clean_clusters.npy", allow_pickle=True)
    number_of_clusters = len(clusters)
    print(number_of_clusters)

    # Downsample Mask
    downsampled_indicies, downsampled_height, downsampled_width = downsample_mask(base_directory)

    # View
    image = np.zeros((downsampled_height * downsampled_width))
    for cluster_index in range(number_of_clusters):
        cluster = clusters[cluster_index]
        cluster_value = vector[cluster_index]
        for pixel in cluster:
            pixel_index = downsampled_indicies[pixel]
            image[pixel_index] = cluster_value

    image = np.ndarray.reshape(image, (downsampled_height, downsampled_width))
    print("Max", np.max(image))
    #image = scipy.ndimage.rotate(image, angle=2, reshape=False)
    plt.imshow(image, cmap='plasma') #gist_ncar
    plt.show()
    #plt.savefig(base_directory + "/Spectral_Clusters.png")
    #plt.close()






session_list = ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging/"]
                #"/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN78.1A/2020_12_09_Switching_Imaging",
                #"/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN78.1D/2020_11_29_Switching_Imaging"]


session_list = ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging/"]
                #"/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK12.1F/2021_09_22_Transition_Imaging"]



session_list = ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging/"]

modulation_matrix = get_mean_modulation_plot(session_list)

# Get Modulation Vector
modulation_vector = np.mean(np.abs(modulation_matrix), axis=1)
#modulation_vector = stats.mode(modulation_matrix, axis=1)[0]
base_directory = r"/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging"
visualise_clusters(base_directory, modulation_vector)

sorted_modualtion = sort_modulation(modulation_matrix)
plot_ridgeplot(sorted_modualtion)


