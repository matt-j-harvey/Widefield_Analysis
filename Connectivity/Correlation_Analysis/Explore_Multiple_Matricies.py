import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import h5py
import tables
from scipy import signal, ndimage, stats
from sklearn.linear_model import LinearRegression
from skimage.morphology import white_tophat
from sklearn.preprocessing import StandardScaler
from skimage.transform import rescale, resize
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
import pathlib
from tqdm import tqdm

pyqtgraph.setConfigOptions(imageAxisOrder='row-major')

from Widefield_Utils import widefield_utils


class correlation_explorer(QWidget):

    def __init__(self, correlation_maps, index_maps, parent=None):
        super(correlation_explorer, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("Correlation Modulation")
        self.setGeometry(0, 0, 1000, 500)

        # Create Variable Holders
        self.correlation_maps = correlation_maps
        self.index_maps = index_maps

        # Create Colourmaps
        colour_list = [[0, 0.87, 0.9, 1],
                       [0, 0, 1, 1],
                       [0, 0, 0, 1],
                       [1, 0, 0, 1],
                       [1, 1, 0, 1], ]
        colour_list = np.array(colour_list)
        colour_list = np.multiply(colour_list, 255)
        value_list = np.linspace(0, 1, num=len(colour_list))
        self.colourmap = pyqtgraph.ColorMap(pos=value_list, color=colour_list)

        # Create Display Views
        self.display_view_list = []
        self.display_widget_list = []

        for mouse in self.correlation_maps:
            mouse_display_views = []
            mouse_display_widgets = []

            for session in mouse:
                display_view, display_widget = self.create_display_widget()
                mouse_display_views.append(display_view)
                mouse_display_widgets.append(display_widget)

            self.display_view_list.append(mouse_display_views)
            self.display_widget_list.append(mouse_display_widgets)

        # Create Layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Add Display View Widgets
        for mouse_index in range(len(self.display_widget_list)):
            for session_index in range(len(self.display_widget_list[mouse_index])):
                self.layout.addWidget(self.display_widget_list[mouse_index][session_index], mouse_index, session_index, 1, 1)


    def getPixel(self, pos, imageview):
        pos = imageview.getImageItem().mapFromScene(pos)
        y = np.clip(int(pos.y()), a_min=0, a_max=100 - 1)
        x = np.clip(int(pos.x()), a_min=0, a_max=100 - 1)
        print(x, y)

        # Update Each Map
        for mouse_index in range(len(self.display_widget_list)):
            for session_index in range(len(self.display_widget_list[mouse_index])):

                # Get Pixel Index
                pixel_index = int(self.index_maps[mouse_index][session_index][y, x])
                print("pixel index", pixel_index)

                # Get Correlation Map
                modulation_image = self.correlation_maps[mouse_index][session_index][pixel_index]

                # Set Display View Image
                display_view = self.display_view_list[mouse_index][session_index]
                display_view.setImage(modulation_image)
                display_view.setLevels(-0.2, 0.2)
                display_view.setColorMap(self.colourmap)


    def create_display_widget(self):
        # Create Figures
        display_view_widget = QWidget()
        display_view_widget_layout = QGridLayout()
        display_view = pyqtgraph.ImageView()
        # display_view.setColorMap(self.colour_map)
        display_view.ui.histogram.hide()
        display_view.ui.roiBtn.hide()
        display_view.ui.menuBtn.hide()
        display_view_widget_layout.addWidget(display_view, 0, 0)
        display_view_widget.setLayout(display_view_widget_layout)
        # display_view_widget.setMinimumWidth(800)
        # display_view_widget.setMinimumHeight(800)
        display_view.getView().scene().sigMouseMoved.connect(lambda pos: self.getPixel(pos, display_view))

        return display_view, display_view_widget


def downsample_mask_further(indicies, image_height, image_width, downsample_size=100):
    template = np.zeros((image_height * image_width))
    template[indicies] = 1
    template = np.reshape(template, (image_height, image_width))
    template = resize(template, (downsample_size, downsample_size), anti_aliasing=True)
    template = np.reshape(template, (downsample_size * downsample_size))
    template = np.where(template > 0.5, 1, 0)
    template_indicies = np.nonzero(template)
    return template_indicies, downsample_size, downsample_size


def get_average_correlation_modulation(session_list, tensor_names, correlation_matrix_filename):

    modulation_matrix_list = []
    for base_directory in session_list:

        # Get File Structure
        split_base_directory = Path(base_directory).parts
        mouse_name = split_base_directory[-2]
        session_name = split_base_directory[-1]

        # Load Correlation Matricies
        condition_1_matrix = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, tensor_names[0] + correlation_matrix_filename))
        condition_2_matrix = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, tensor_names[1] + correlation_matrix_filename))
        modulation_matrix = np.subtract(condition_1_matrix, condition_2_matrix)

        # Add To List
        modulation_matrix_list.append(modulation_matrix)

    # Get Mean Modulation Matrix
    modulation_matrix_list = np.array(modulation_matrix_list)
    mean_modulation_matrix = np.mean(modulation_matrix_list, axis=0)

    return mean_modulation_matrix


def construct_correlation_maps(modulation_matrix, indicies, image_height, image_width):

    # Create Correlation Map
    correlation_map_matrix = []
    #plt.ion()
    for pixel in modulation_matrix:
        correlation_map = widefield_utils.create_image_from_data(pixel, indicies, image_height, image_width)
        #plt.imshow(correlation_map)
        #plt.draw()
        #plt.pause(0.1)
        #plt.clf()
        correlation_map_matrix.append(correlation_map)

    correlation_map_matrix = np.array(correlation_map_matrix)
    return correlation_map_matrix


def load_correlation_maps_and_index_maps(mouse_list, tensor_save_directory, tensor_names, correlation_matrix_filename):

    # Create Empty Lists To Hold Variables
    group_correlation_map_list = []
    group_index_map_list = []

    for mouse in tqdm(mouse_list):
        mouse_correlation_map_list = []
        mouse_index_map_list = []

        for base_directory in mouse:

            # Get File Structure
            split_base_directory = pathlib.Path(base_directory).parts
            mouse_name = split_base_directory[-2]
            session_name = split_base_directory[-1]

            # Load Correlation Matricies
            condition_1_matrix = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, tensor_names[0] + correlation_matrix_filename))
            condition_2_matrix = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, tensor_names[1] + correlation_matrix_filename))
            modulation_matrix = np.subtract(condition_1_matrix, condition_2_matrix)

            print("Modulation matrix", np.shape(modulation_matrix))
            #plt.imshow(modulation_matrix)
            #plt.show()

            # Load Mask
            #indicies, image_height, image_width = widefield_utils.load_downsampled_mask(base_directory)
            indicies, image_height, image_width = widefield_utils.load_tight_mask()
            print("Indicies", np.shape(indicies))

            # Downsample Further
            indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)
            print("Indicies", np.shape(indicies))

            # Reconstruct Correlation Maps
            modulation_maps = construct_correlation_maps(modulation_matrix, indicies, image_height, image_width)

            # Create Index Map
            index_map = np.zeros((image_height * image_width))
            index_map[indicies] = list(range(np.shape(indicies)[1]))
            index_map = np.reshape(index_map, (image_height, image_width))

            # Add To List
            mouse_correlation_map_list.append(modulation_maps)
            mouse_index_map_list.append(index_map)

        group_correlation_map_list.append(mouse_correlation_map_list)
        group_index_map_list.append(mouse_index_map_list)

    return group_correlation_map_list, group_index_map_list



def explore_multiple_correlation_matricies(mouse_list, tensor_names, correlation_matrix_filename):

    app = QApplication(sys.argv)

    # Create Correlation Maps
    correlation_maps, index_maps = load_correlation_maps_and_index_maps(mouse_list, tensor_save_directory, tensor_names, correlation_matrix_filename)

    window = correlation_explorer(correlation_maps, index_maps)
    window.showMaximized()

    app.exec_()



control_mouse_list = [

                ["/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_28_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_05_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_09_Switching_Imaging"],

                ["/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_29_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_05_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging"],

                ["/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_02_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_04_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_06_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_02_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_08_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_10_Transition_Imaging"],

                ["/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_26_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_28_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_02_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_23_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_31_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_04_02_Transition_Imaging"],

                ["/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_21_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_23_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_11_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_13_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_15_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_17_Transition_Imaging"],

                ["/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_14_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_20_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_22_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_29_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_03_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_05_Transition_Imaging"],
]




mutant_mouse_list = [

                ["/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_13_Switching_Imaging",
                 "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_15_Switching_Imaging",
                 "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_17_Switching_Imaging"],

                ["/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_20_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_22_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_24_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_14_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_16_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_18_Transition_Imaging"],

                ["/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_17_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_19_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_23_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_30_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_06_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_08_Transition_Imaging"],

                ["/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_15_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_17_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_19_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_22_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_24_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_26_Transition_Imaging"],

                ["/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_14_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_20_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_26_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_05_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_08_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_10_Transition_Imaging"],

                ["/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_31_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_02_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_04_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_08_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_10_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_12_Transition_Imaging"],


]


control_mouse_list = [
                ["/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_28_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_05_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_09_Switching_Imaging"],

                ["/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_29_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_05_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging"],

                ["/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_02_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_04_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_06_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_02_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_08_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_10_Transition_Imaging"],

                ["/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_26_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_28_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_02_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_23_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_31_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_04_02_Transition_Imaging"],

                ["/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_21_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_23_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_11_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_13_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_15_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_17_Transition_Imaging"],

                ["/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_14_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_20_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_22_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_29_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_03_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_05_Transition_Imaging"],

]



test_mouse_list = [


                ["/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_20_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_22_Switching_Imaging"],
                ["/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_24_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_14_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_16_Transition_Imaging"],
]

# Get Analysis Details
analysis_name = "Unrewarded_Contextual_Modulation"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)

# Set Tensor Directory
tensor_save_directory = r"/media/matthew/29D46574463D2856/Widefield_Analysis/Noise_Correlation_Analysis/Activity_Tensors"

# Explore Correlation Matricies
#correlation_matrix_filename = "_Signal_Correlation_Matrix_Aligned_Within_Mouse.npy"
#correlation_matrix_filename = "_Signal_Correlation_Matrix_Aligned_Across_Mice.npy"
correlation_matrix_filename = "_Noise_Correlation_Matrix_Aligned_Within_Mouse.npy"
explore_multiple_correlation_matricies(mutant_mouse_list, tensor_names, correlation_matrix_filename)



