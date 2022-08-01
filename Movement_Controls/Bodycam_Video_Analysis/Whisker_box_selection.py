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

pyqtgraph.setConfigOptions(imageAxisOrder='row-major')



def get_video_name(base_directory):

    file_list = os.listdir(base_directory)
    for file in file_list:
        if "_cam_1" in file:
            return file


def load_image_still(video_file):

    # Open Video File
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Extract Selected Frames
    frame_index = 0
    ret = True
    while (frame_index < 1 and ret):
        ret, frame = cap.read()
        frame_index += 1

    cap.release()

    return frame


class roi_selection_window(QWidget):

    def __init__(self, session_list, parent=None):
        super(roi_selection_window, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("ROI Selector")
        self.setGeometry(0, 0, 1900, 500)

        # Setup Internal Variables
        self.session_list = session_list
        self.number_of_sessions = len(self.session_list)
        self.current_session_index = 0

        # Get List Of Max Projections
        self.image_list = []
        for base_directory in session_list:

            # Get Video Name
            video_name = get_video_name(base_directory)
            
            # Load Video Frame
            frame = load_image_still(os.path.join(base_directory, video_name))

            # Add To List
            self.image_list.append(frame)
            
        # Set Current Images
        self.current_image = self.image_list[0]

        # Regressor Display View
        self.mousecam_display_view_widget = QWidget()
        self.mousecam_display_view_widget_layout = QGridLayout()
        self.mousecam_display_view = pyqtgraph.ImageView()
        self.mousecam_display_view.ui.histogram.hide()
        self.mousecam_display_view.ui.roiBtn.hide()
        self.mousecam_display_view.ui.menuBtn.hide()
        self.mousecam_display_view_widget_layout.addWidget(self.mousecam_display_view, 0, 0)
        self.mousecam_display_view_widget.setLayout(self.mousecam_display_view_widget_layout)
        self.mousecam_display_view_widget.setMinimumWidth(800)
        self.mousecam_display_view_widget.setMinimumHeight(800)
        cm = pyqtgraph.colormap.get('CET-R4')
        self.mousecam_display_view.setColorMap(cm)
        self.mousecam_display_view.setImage(self.current_image)

        # Create Session List Views
        self.session_list_widget = QListWidget()
        for session in self.session_list:
            session_name = session.split('/')[-1]
            self.session_list_widget.addItem(session_name)

        self.session_list_widget.setCurrentRow(self.current_session_index)

        # Create Map Button
        self.map_button = QPushButton("Map Region")
        self.map_button.clicked.connect(self.map_region)

        # Create and Set Layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Add Display Views
        self.layout.addWidget(self.mousecam_display_view_widget, 0, 0, 1, 1)
        self.layout.addWidget(self.map_button, 1, 0, 1, 1)
        # Add List Views
        self.layout.addWidget(self.session_list_widget, 0, 1, 2, 1)

        # Add ROI
        self.whisker_roi = pyqtgraph.PolyLineROI(positions=[[500, 100], [550, 100], [550, 150], [500, 150]], closed=True)
        self.mousecam_display_view.addItem(self.whisker_roi)


    def map_region(self):

        roi_handles = self.whisker_roi.getLocalHandlePositions()

        polygon_verticies = []
        for handle in roi_handles:
            handle = handle[1]

            if type(handle) == QPointF:
                handle_coords = [int(handle.x()), int(handle.y())]

            else:
                handle = np.array(handle)
                handle_coords = [int(handle[0]), int(handle[1])]
            polygon_verticies.append(handle_coords)

        image_height, image_width, rgb_depth = np.shape(self.current_image)
        x, y = np.meshgrid(np.arange(image_width), np.arange(image_height))  # make a canvas with coordinates
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T

        print("Polygon verticies", polygon_verticies)
        p = Path(polygon_verticies)  # make a polygon
        grid = p.contains_points(points)
        grid = np.reshape(grid, (image_height, image_width))

        whisker_coords = np.nonzero(grid)

        # Save These
        save_directory = os.path.join(self.session_list[self.current_session_index], "Mousecam_analysis")
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)

        np.save(os.path.join(save_directory, "Whisker_Pixels.npy"), whisker_coords)

def assign_alm_pixels(base_directory):
    # Load Mask
    indicies, image_height, image_width = load_mask(base_directory)

    # Get All Vis 1 Coefs
    all_vis_1_coefs = np.load(os.path.join(base_directory, "Simple_Regression", "All_Vis_1_v_All_Vis_2_Regression_Model.npy"), allow_pickle=True)[()]
    all_vis_1_coefs = all_vis_1_coefs["Coefficients_of_Partial_Determination"]
    all_vis_1_coefs = all_vis_1_coefs[0]
    print("All vis 1 coefs", all_vis_1_coefs)
    # all_vis_1_coefs = reconstruct_images(all_vis_1_coefs, indicies, image_height, image_width)

    for frame in all_vis_1_coefs:
        plt.imshow(frame)
        plt.show()


# Load Sessions
session_list = [r"/media/matthew/External_Harddrive_1/Processed_Widefield_Data/NRXN78.1D/2020_12_07_Switching_Imaging"]

app = QApplication(sys.argv)

window = roi_selection_window(session_list)
window.show()

app.exec_()