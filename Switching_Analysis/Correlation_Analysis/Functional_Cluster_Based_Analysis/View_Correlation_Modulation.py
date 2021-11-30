import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import h5py
import tables
from scipy import signal, ndimage, stats
from sklearn.linear_model import LinearRegression
from skimage.morphology import white_tophat
from sklearn.preprocessing import StandardScaler
from skimage.transform import rescale
from PIL import Image
import os
import cv2
import datetime

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph
import sys

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import FactorAnalysis, TruncatedSVD, FastICA, PCA, NMF
from sklearn.metrics import accuracy_score
from sklearn.cluster import MiniBatchKMeans, SpectralClustering, DBSCAN, AffinityPropagation
from sklearn.mixture import GaussianMixture
import networkx as nx
import cv2
from matplotlib.pyplot import cm
from matplotlib.colors import to_rgb
import os
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
import random


pyqtgraph.setConfigOptions(imageAxisOrder='row-major')

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



def factor_number(number_to_factor):
    factor_list = []
    for potential_factor in range(1, number_to_factor):
        if number_to_factor % potential_factor == 0:
            factor_pair = [potential_factor, int(number_to_factor/potential_factor)]
            factor_list.append(factor_pair)

    return factor_list


def get_best_grid(number_of_items):
    factors = factor_number(number_of_items)
    factor_difference_list = []

    #Get Difference Between All Factors
    for factor_pair in factors:
        factor_difference = abs(factor_pair[0] - factor_pair[1])
        factor_difference_list.append(factor_difference)

    #Select Smallest Factor difference
    smallest_difference = np.min(factor_difference_list)
    best_pair = factor_difference_list.index(smallest_difference)

    return factors[best_pair]


def sort_matrix(matrix):

    # Cluster Matrix
    Z = ward(pdist(matrix))

    # Get Dendogram Leaf Order
    new_order = leaves_list(Z)

    # Sorted Matrix
    sorted_matrix = matrix[:, new_order][new_order]

    return sorted_matrix


class correlation_explorer(QWidget):

    def __init__(self, session_list, file_path, parent=None):
        super(correlation_explorer, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("Correlation Modulation")
        self.setGeometry(0, 0, 1000, 500)

        # Create Variable Holders
        self.session_list = session_list
        self.number_of_sessions = len(session_list)
        self.modulation_maps = []
        self.display_view_list = []
        self.display_widgets_list = []


        # Load Modulation Maps
        for session in session_list:
            modulation_map = np.load(session + file_path)
            modulation_map = -1 * modulation_map
            print("Modulation map shape", np.shape(modulation_map))
            self.modulation_maps.append(modulation_map)

        # Load Clusters
        cluster_file = r"/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/clean_clusters.npy"
        self.clusters = np.load(cluster_file, allow_pickle=True)
        self.number_of_clusters = len(self.clusters)
        self.indicies, self.downsampled_height, self.downsampled_width = downsample_mask(session_list[0])

        # Create Cluster Map
        self.cluster_map = np.zeros((self.downsampled_height * self.downsampled_width))
        for cluster_index in range(self.number_of_clusters):
            cluster = self.clusters[cluster_index]
            for pixel in cluster:
                self.cluster_map[self.indicies[pixel]] = cluster_index
        self.cluster_map = np.ndarray.reshape(self.cluster_map, (self.downsampled_height, self.downsampled_width))

        # Create Display Views
        for session in range(self.number_of_sessions):
            display_view, display_widget = self.create_display_widget()
            self.display_view_list.append(display_view)
            self.display_widgets_list.append(display_widget)

        # Create Layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Add Display Widgets
        [rows, columns] = get_best_grid(self.number_of_sessions)

        count = 0
        for row in range(rows):
            for column in range(columns):
                if count < self.number_of_sessions:
                    print(row, column)
                    self.layout.addWidget(self.display_widgets_list[count], row, column, 1, 1)
                count += 1


    def view_modulation_map(self, selected_region):

        # Highlight Current Region
        number_of_clusters = len(self.clusters)
        for session in range(self.number_of_sessions):
            session_map = np.zeros((self.downsampled_height * self.downsampled_width, 3))

            print("Modulation map", np.shape(self.modulation_maps[session]))
            modulation_vector = self.modulation_maps[session][selected_region]
            for cluster_index in range(number_of_clusters):
                cluster_modulation = modulation_vector[cluster_index]
                cluster = self.clusters[cluster_index]
                for pixel in cluster:

                    if cluster_index == selected_region:
                        session_map[self.indicies[pixel]] = [255, 255, 0]

                    else:

                        cluster_modulation = cluster_modulation * 2
                        modulation_rgb_value = np.clip(np.abs(cluster_modulation * 255), a_min=None, a_max=255)

                        if cluster_modulation >= 0:
                            session_map[self.indicies[pixel]] = [255,  255 - modulation_rgb_value, 255 - modulation_rgb_value]

                        else:
                            session_map[self.indicies[pixel]] = [255 - modulation_rgb_value, 255 - modulation_rgb_value,255]



            session_image = np.reshape(session_map, (self.downsampled_height, self.downsampled_width, 3))
            self.display_view_list[session].setImage(session_image)


    def getPixel(self, pos, imageview):
        pos = imageview.getImageItem().mapFromScene(pos)
        y = np.clip(int(pos.y()), a_min=0, a_max=self.downsampled_height-1)
        x = np.clip(int(pos.x()), a_min=0, a_max=self.downsampled_width-1)
        selected_cluster = self.cluster_map[y, x]
        print("Selected Cluster", selected_cluster)
        self.view_modulation_map(int(selected_cluster))

        print(imageview.getImageItem().mapFromScene(pos))

    def create_display_widget(self):

        # Create Figures
        display_view_widget = QWidget()
        display_view_widget_layout = QGridLayout()
        display_view = pyqtgraph.ImageView()
        #display_view.setColorMap(self.colour_map)
        display_view.ui.histogram.hide()
        display_view.ui.roiBtn.hide()
        display_view.ui.menuBtn.hide()
        display_view_widget_layout.addWidget(display_view, 0, 0)
        display_view_widget.setLayout(display_view_widget_layout)
        #display_view_widget.setMinimumWidth(800)
        #display_view_widget.setMinimumHeight(800)


        display_view.getView().scene().sigMouseMoved.connect(lambda pos: self.getPixel(pos, display_view))

        return display_view, display_view_widget





def create_modulation_maps(session_list):

    for base_directory in session_list:

        # Load Correlation Tensors
        save_directory = base_directory + "/Pre_Stimulus/"
        visual_tensor = np.load(save_directory + "Visual_Context_Correlation_Tensor.npy")
        odour_tensor = np.load(save_directory + "Odour_Context_Correlation_Tensor.npy")

        # Get n-th discrete difference
        mean_visual_tensor = np.mean(visual_tensor, axis=0)
        mean_odour_tensor = np.mean(odour_tensor, axis=0)
        modulation = np.diff(np.array([mean_visual_tensor, mean_odour_tensor]), axis=0)

        number_of_clusters = np.shape(modulation)[1]
        modulation = np.reshape(modulation, (number_of_clusters, number_of_clusters))
        #modulation = sort_matrix(modulation)


        # Perform T Tests
        t_stats, p_values = stats.ttest_ind(visual_tensor, odour_tensor, axis=0)

        # Threshold t_stats
        thresholded_modulation_map = np.where(p_values < 0.01, modulation, 0)

        # View Map
        #plt.imshow(thresholded_modulation_map, vmax=0.5, vmin=-0.5, cmap='bwr')
        #plt.show()

        # Save Thresholded Modulation Map
        np.save(save_directory + "/Thresholded_Modulation_Map.npy", modulation)





if __name__ == '__main__':

    app = QApplication(sys.argv)
    #app.setQuitOnLastWindowClosed(False)

    session_list = [
        "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/",
        "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging/",
        "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging/",
        "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN78.1A/2020_12_09_Switching_Imaging",
        "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN78.1D/2020_11_29_Switching_Imaging",

        "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging",
        "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/",
        "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging/",
        "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK12.1F/2021_09_22_Transition_Imaging",
        "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN71.2A/2020_12_17_Switching_Imaging"]



    #create_modulation_maps(session_list)
    file_path = "/Noise_Correlations/Noise_Correlation_Delta_Matrix.npy"
    #file_path = "/Noise_Correlations/Mean_Correlation_Delta_Matrix.npy"
    #file_path = "/Pre_Stimulus/Thresholded_Modulation_Map.npy"
    #file_path = "/Pre_Stimulus/Concatenated_Modulation.npy"
    window = correlation_explorer(session_list, file_path)
    window.showMaximized()

    window.view_modulation_map(9)
    window.view_modulation_map(73)
    #window.view_modulation_map(39)
    #window.view_modulation_map(49)

    app.exec_()





