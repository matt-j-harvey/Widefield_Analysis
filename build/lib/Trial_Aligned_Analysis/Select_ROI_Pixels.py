import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from matplotlib.path import Path

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph

import os
import sys

from tqdm import tqdm

import Preprocessing_Utils

pyqtgraph.setConfigOptions(imageAxisOrder='row-major')


def get_video_name(base_directory):
    file_list = os.listdir(base_directory)
    for file in file_list:
        if "_cam_1" in file:
            return file


def load_image_still(base_directory):
    example_image = np.load(os.path.join(base_directory, "Blue_Example_Image.npy"))
    return example_image


class roi_selection_window(QWidget):

    def __init__(self, base_directory, parent=None):
        super(roi_selection_window, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("ROI Selector")
        self.setGeometry(0, 0, 1900, 500)

        # Setup Internal Variables
        self.base_directory = base_directory
        
        # Load Video Frame
        self.example_image = load_image_still(base_directory)

        # Regressor Display View
        self.brain_display_view_widget = QWidget()
        self.brain_display_view_widget_layout = QGridLayout()
        self.brain_display_view = pyqtgraph.ImageView()
        self.brain_display_view.ui.histogram.hide()
        self.brain_display_view.ui.roiBtn.hide()
        self.brain_display_view.ui.menuBtn.hide()
        self.brain_display_view_widget_layout.addWidget(self.brain_display_view, 0, 0)
        self.brain_display_view_widget.setLayout(self.brain_display_view_widget_layout)
        self.brain_display_view_widget.setMinimumWidth(604)
        self.brain_display_view_widget.setMinimumHeight(600)
        cm = pyqtgraph.colormap.get('CET-R4')
        self.brain_display_view.setColorMap(cm)
        self.brain_display_view.setImage(self.example_image)

        # Create Map Button
        self.map_button = QPushButton("Map Region")
        self.map_button.clicked.connect(self.map_region)

        # Create and Set Layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Add Display Views
        self.layout.addWidget(self.brain_display_view_widget, 0, 0, 1, 10)
        self.layout.addWidget(self.map_button, 1, 0, 1, 1)

        # Add ROI
        self.brain_roi = pyqtgraph.RectROI([1, 1], [27, 28])
        self.brain_display_view.addItem(self.brain_roi)


    def map_region(self):

        # Create Index Map
        indicies, image_height, image_width = Preprocessing_Utils.load_generous_mask(self.base_directory)
        index_map = np.zeros(image_height * image_width)
        index_map[indicies] = list(range(len(indicies)))
        index_map = np.reshape(index_map, (image_height, image_width))

        # Get ROI Pos
        pos_x, pos_y = self.brain_roi.pos()  # POs is Lower Left corner
        size_x, size_y = self.brain_roi.size()
        pos_x = int(np.around(pos_x, 0))
        pos_y = int(np.around(pos_y, 0))
        size_x = int(np.around(size_x, 0))
        size_y = int(np.around(size_y, 0))

        # Get Slice Of Index Map
        index_map_slice = index_map[pos_y:pos_y+size_y, pos_x:pos_x + size_x]
        roi_indicies = np.ndarray.flatten(index_map_slice)
        roi_indicies = np.ndarray.astype(roi_indicies, int)

        # Save These
        np.save(os.path.join(self.base_directory, "Selected_ROI.npy"), roi_indicies)
        print("Mapped")

if __name__ == '__main__':
    app = QApplication(sys.argv)

    base_directory = r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging/Heamocorrection_Visualisation"

    selection_window = roi_selection_window(base_directory)
    selection_window.show()

    app.exec_()


