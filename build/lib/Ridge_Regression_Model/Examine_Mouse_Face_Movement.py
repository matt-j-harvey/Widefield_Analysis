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
from tqdm import tqdm

from sklearn.decomposition import PCA

pyqtgraph.setConfigOptions(imageAxisOrder='row-major')

import MVAR_Utils

class correlation_explorer(QWidget):

    def __init__(self, bodycam_data, motion_pc_data, parent=None):
        super(correlation_explorer, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("Examine Mouse Bodycam")
        self.setGeometry(0, 0, 1000, 500)

        # Create Variable Holders
        self.bodycam_data = bodycam_data
        self.number_of_frames = np.shape(bodycam_data)[0]

        # Create Camera Dspaly View
        self.bodycam_display_view, self.bodycam_display_view_widget = self.create_display_widget()

        # Create Frame Selection Slider
        self.frame_selection_slider = QSlider(Qt.Horizontal)
        self.frame_selection_slider.setMinimum(0)
        self.frame_selection_slider.setMaximum(self.number_of_frames)
        self.frame_selection_slider.valueChanged.connect(self.select_frame)

        #self.delta_f_pen = pyqtgraph.mkPen(color=(200, 0, 0), width=2)
        #self.design_matrix_pen = pyqtgraph.mkPen(color=(0, 0, 200), width=2)

        # Create Motion PC Graph
        self.motion_pc_graph = pyqtgraph.PlotWidget()
        for trace in motion_pc_data:
            trace_plot_item = pyqtgraph.PlotCurveItem()
            trace_plot_item.setData(trace)
            self.motion_pc_graph.addItem(trace_plot_item)
        #self.motion_pc_plot_item = pyqtgraph.PlotCurveItem()
        #self.graph_display_view.addItem(self.delta_f_plot_item)
        #self.graph_display_view.addItem(self.design_plot_item)

        self.position_indicator_line = pyqtgraph.InfiniteLine()
        self.motion_pc_graph.addItem(self.position_indicator_line)


        # Create Layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.bodycam_display_view, 0, 0, 1, 1)
        self.layout.addWidget(self.motion_pc_graph)
        self.layout.addWidget(self.frame_selection_slider, 2, 0, 1, 1)

        """
        self.layout.addWidget(self.column_display_view_label, 0, 1, 1, 1)

        self.layout.addWidget(self.row_correlation_map_display_widget,      1, 0, 1, 1)
        self.layout.addWidget(self.column_correlation_map_display_widget,   1, 1, 1, 1)
        self.layout.addWidget(self.graph_display_view, 2, 0, 1, 2)

        self.layout.addWidget(self.x_spin_box, 3, 0, 1, 1)
        self.layout.addWidget(self.y_spin_box, 3, 1, 1, 1)
        """

    def select_frame(self):
        # Get Selected Frame
        selected_frame = int(self.frame_selection_slider.value())
        self.bodycam_display_view.setImage(self.bodycam_data[selected_frame])
        self.position_indicator_line.setValue(selected_frame)
        self.update()

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

        #display_view.getView().scene().sigMouseMoved.connect(lambda pos: self.getPixel(pos, display_view))

        return display_view, display_view_widget



def get_video_data(video_file):

    # Open Video File
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Extract Selected Frames
    face_data = np.zeros((frameCount, frameHeight, frameWidth), dtype=np.uint8)
    for frame_index in tqdm(range(frameCount)):
        ret, frame = cap.read()
        frame = frame[:, :, 0]
        frame = np.ndarray.astype(frame, np.uint8)
        face_data[frame_index] = frame

    cap.release()
    return face_data



def get_matched_bodycam_frames(base_directory, mousecam_data):

    # Load Widefield To Mousecam Frame Dict
    widefield_to_mousecam_frame_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy"), allow_pickle=True)[()]
    widefield_frame_list = list(widefield_to_mousecam_frame_dict.keys())
    number_of_mousecam_frames = np.shape(mousecam_data)[0]

    print("Widefield Frames", len(widefield_frame_list))
    print("Mousecam Frames", number_of_mousecam_frames)
    print("Minimum Matched Mousecam Frame", np.min(list(widefield_to_mousecam_frame_dict.values())))
    print("Maximum Matched Mousecam Frame", np.max(list(widefield_to_mousecam_frame_dict.values())))
    print("Transformed Whisker Data Shape", np.shape(mousecam_data))

    # Match Whisker Activity To Widefield Frames
    matched_mousecam_frames = []
    for widefield_frame in widefield_frame_list:
        corresponding_mousecam_frame = widefield_to_mousecam_frame_dict[widefield_frame]
        if corresponding_mousecam_frame < number_of_mousecam_frames:
            matched_mousecam_frames.append(corresponding_mousecam_frame)
        else:
            print("unmatched, mousecam frame: ", corresponding_mousecam_frame)

    return matched_mousecam_frames


def get_matched_bodycam_data(bodycam_data, matched_frames):
    matched_data = []
    for frame in tqdm(matched_frames):
        matched_data.append(bodycam_data[frame])
    matched_data = np.array(matched_data, dtype=np.uint8)
    return matched_data


def get_bodycam_filename(base_directory):
    file_list = os.listdir(base_directory)
    for file_name in file_list:
        if "_cam_1.mp4" in file_name:
            return file_name



if __name__ == '__main__':

    app = QApplication(sys.argv)

    base_directory = r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_09_Switching_Imaging"

    # Load Mouse PC Traces
    pc_traces = np.load(os.path.join(base_directory, "Mousecam_Analysis", "matched_face_data.npy"))
    pc_traces = np.transpose(pc_traces)
    print("PC Traces", np.shape(pc_traces))

    # Get Bodycam Filename
    bodycam_filename = get_bodycam_filename(base_directory)
    bodycam_file = os.path.join(base_directory, bodycam_filename)
    bodycam_data = get_video_data(bodycam_file)

    # Get Matched Bodycam Data
    matched_mousecam_frames = get_matched_bodycam_frames(base_directory, bodycam_data)
    bodycam_data = get_matched_bodycam_data(bodycam_data, matched_mousecam_frames)

    window = correlation_explorer(bodycam_data, pc_traces)
    window.showMaximized()

    app.exec_()



