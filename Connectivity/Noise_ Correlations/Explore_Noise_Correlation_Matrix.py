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

pyqtgraph.setConfigOptions(imageAxisOrder='row-major')

import Noise_Correlation_Utils

class correlation_explorer(QWidget):

    def __init__(self, correlation_matrix, indicies, image_height, image_width, parent=None):
        super(correlation_explorer, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("Correlation Modulation")
        self.setGeometry(0, 0, 1000, 500)

        # Create Variable Holders
        self.correlation_matrix = correlation_matrix
        self.indicies = indicies
        self.image_height = image_height
        self.image_width = image_width
        self.index_map = self.create_index_map(indicies, image_height, image_width)

        colour_list = [[0, 0.87, 0.9, 1],
                        [0, 0, 1, 1],
                        [0, 0, 0, 1],
                        [1, 0, 0, 1],
                        [1, 1, 0, 1],]
        colour_list = np.array(colour_list)
        colour_list = np.multiply(colour_list, 255)

        value_list = np.linspace(0, 1, num=len(colour_list))
        print("Valye list", value_list)
        self.colourmap = pyqtgraph.ColorMap(pos=value_list, color=colour_list)


        # Create Display Views
        self.correlation_map_display_view, self.correlation_map_display_widget = self.create_display_widget()
        self.correlation_map_display_view.setImage(Noise_Correlation_Utils.create_image_from_data(self.correlation_matrix[0], indicies, image_height, image_width))

        # Create Layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.correlation_map_display_widget, 0, 0, 1, 1)

    def create_index_map(self, indicies, image_height, image_width):
        index_map = np.zeros(image_height * image_width)
        index_list = list(range(np.shape(indicies)[1]))
        index_map[indicies] = index_list
        index_map = np.reshape(index_map, (image_height, image_width))
        return index_map

    def getPixel(self, pos, imageview):
        pos = imageview.getImageItem().mapFromScene(pos)
        y = np.clip(int(pos.y()), a_min=0, a_max=self.image_height - 1)
        x = np.clip(int(pos.x()), a_min=0, a_max=self.image_width - 1)
        pixel_index = int(self.index_map[y, x])
        modulation = self.correlation_matrix[pixel_index]
        modulation_image = Noise_Correlation_Utils.create_image_from_data(modulation, self.indicies, self.image_height, self.image_width)
        self.correlation_map_display_view.setImage(modulation_image)
        self.correlation_map_display_view.setLevels(-1, 1)
        self.correlation_map_display_view.setColorMap(self.colourmap)

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
    template = np.zeros((image_height*image_width))
    template[indicies] = 1
    template = np.reshape(template, (image_height, image_width))
    template = resize(template, (downsample_size, downsample_size), anti_aliasing=True)
    template = np.reshape(template, (downsample_size * downsample_size))
    template = np.where(template > 0.5, 1, 0)
    template_indicies = np.nonzero(template)
    return template_indicies, downsample_size, downsample_size



if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Load Combined Mask
    indicies, image_height, image_width = Noise_Correlation_Utils.load_tight_mask()

    # Downsample Mask Further
    indicies, image_height, image_width = downsample_mask_further(indicies, image_height, image_width)

    # Load Correlation Map
    """
    matrix_root_directory = r"/media/matthew/29D46574463D2856/Widefield_Analysis/Noise_Correlation_Analysis/Activity_Tensors"
    mouse = "NXAK16.1B"
    session = "2021_06_30_Transition_Imaging"
    context_1_correlation_matrix = np.load(os.path.join(matrix_root_directory, mouse, session, "visual_context_stable_vis_2_Correlation_Matrix.npy"))
    context_2_correlation_matrix = np.load(os.path.join(matrix_root_directory, mouse, session, "odour_context_stable_vis_2_Correlation_Matrix.npy"))
    correlation_matrix = np.subtract(context_1_correlation_matrix, context_2_correlation_matrix)
    
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 3, 1)
    axis_2 = figure_1.add_subplot(1, 3, 2)
    axis_3 = figure_1.add_subplot(1, 3, 3)
    axis_1.imshow(context_1_correlation_matrix)
    axis_2.imshow(context_2_correlation_matrix)
    axis_3.imshow(correlation_matrix)
    plt.show()
    
    """

    #mean_matrix_save_directory = r"/media/matthew/29D46574463D2856/Widefield_Analysis/Noise_Correlation_Analysis/Mean_Modulation_Matricies"
    #correlation_matrix = np.load(os.path.join(mean_matrix_save_directory, "Mean_Control_Signal_Modulation.npy"))
    #mean_correlation = np.mean(correlation_matrix, axis=0)

    #matrix_save_directory = "/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Signal_Modulation/Control_Learning_Signal_Modulation"
    #matrix_save_directory = "/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Signal_Modulation/Mutant_Learning_Signal_Modulation"
    matrix_save_directory = "/media/matthew/29D46574463D2856/Widefield_Analysis/Noise_Correlation_Analysis/Mean_Modulation_Matricies"

    correlation_matrix = np.load(os.path.join(matrix_save_directory, "Mean_Mutant_Signal_Modulation_Significant.npy"))
    #p_matrix = np.load(os.path.join(matrix_save_directory, "Signal_p_Matrix.npy"))
    #correlation_matrix = np.where(p_matrix < 0.05, correlation_matrix, 0)

    mean_correlation = np.mean(np.abs(correlation_matrix), axis=0)
    mean_correlation_image = Noise_Correlation_Utils.create_image_from_data(mean_correlation, indicies, image_height, image_width)
    plt.imshow(mean_correlation_image)
    plt.show()

    window = correlation_explorer(correlation_matrix, indicies, image_height, image_width)
    window.showMaximized()

    app.exec_()



