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

pyqtgraph.setConfigOptions(imageAxisOrder='row-major')



def get_video_name(base_directory):

    file_list = os.listdir(base_directory)
    for file in file_list:
        if "_cam_1.mp4" in file:
            return file


def load_image_still(video_file, n_frames=100):

    # Open Video File
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Frame count", frameCount)
    # Get Evenly Spread Frames
    selected_frames = np.linspace(start=0, stop=frameCount-1, num=n_frames, dtype=int)
    selected_frames_list = []
    """
    frame_index = 0
    ret = True
    while (frame_index < frameCount and ret):
        ret, frame = cap.read()
        if frame_index in selected_frames:
            selected_frames_list.append(frame)

        frame_index += 1
    """

    # Extract Selected Frames
    for frame in selected_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, frame = cap.read()
        selected_frames_list.append(frame)

    cap.release()

    return selected_frames_list


class face_selection_window(QWidget):

    def __init__(self, session_list, parent=None):
        super(face_selection_window, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("ROI Selector")
        self.setGeometry(0, 0, 1900, 500)

        # Setup Internal Variables
        self.session_list = session_list
        self.number_of_sessions = len(self.session_list)
        self.current_session_index = 0

        # Regressor Display View
        self.mousecam_display_view_widget = QWidget()
        self.mousecam_display_view_widget_layout = QGridLayout()
        self.mousecam_display_view = pyqtgraph.ImageView()
        self.mousecam_display_view.ui.histogram.hide()
        self.mousecam_display_view.ui.roiBtn.hide()
        self.mousecam_display_view.ui.menuBtn.hide()
        self.mousecam_display_view_widget_layout.addWidget(self.mousecam_display_view, 0, 0)
        self.mousecam_display_view_widget.setLayout(self.mousecam_display_view_widget_layout)
        self.mousecam_display_view_widget.setMinimumWidth(1500)
        self.mousecam_display_view_widget.setMinimumHeight(800)
        cm = pyqtgraph.colormap.get('CET-R4')
        self.mousecam_display_view.setColorMap(cm)

        # Create Session List Views
        self.session_list_widget = QListWidget()
        self.session_list_widget.currentItemChanged.connect(self.change_session)
        self.image_list = []

        for base_directory in tqdm(session_list, desc="Loading Sessions"):
            session_name = base_directory.split('/')[-1]
            self.session_list_widget.addItem(session_name)

            # Get Video Name
            print("Base Directory", base_directory)
            video_name = get_video_name(base_directory)
            print("video name", video_name)


            # Load Video Frame
            frame_list = load_image_still(os.path.join(base_directory, video_name))

            # Add To List
            self.image_list.append(frame_list)

        # Set Current Images
        self.current_image = self.image_list[0][0]
        self.mousecam_display_view.setImage(self.current_image)

        # Create Map Button
        self.map_button = QPushButton("Map Region")
        self.map_button.clicked.connect(self.map_region)

        # Create Frame Slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(len(self.image_list[self.current_session_index])-1)
        self.frame_slider_label = QLabel("Frame: 0")
        self.frame_slider.valueChanged.connect(self.change_frame)

        self.session_list_widget.setCurrentRow(self.current_session_index)

        # Create and Set Layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Add Display Views
        self.layout.addWidget(self.mousecam_display_view_widget, 0, 0, 1, 10)
        self.layout.addWidget(self.frame_slider, 1, 0, 1, 8)
        self.layout.addWidget(self.frame_slider_label, 1, 8, 1, 1)
        self.layout.addWidget(self.map_button, 1, 9, 1, 1)


        # Add List Views
        self.layout.addWidget(self.session_list_widget, 0, 11, 2, 1)

        # Add ROI
        self.whisker_roi = pyqtgraph.PolyLineROI(positions=[[500, 100], [550, 100], [550, 150], [500, 150]], closed=True)
        self.mousecam_display_view.addItem(self.whisker_roi)

    def change_frame(self):
        current_frame = int(self.frame_slider.value())
        self.current_image = self.image_list[self.current_session_index][current_frame]
        self.mousecam_display_view.setImage(self.current_image)

    def change_session(self):
        self.current_session_index = int(self.session_list_widget.currentRow())
        self.change_frame()

    def map_region(self):

        print("Current image Shaoe", self.current_image.shape)
        cols, rows, colours = self.current_image.shape
        m = np.mgrid[:cols, :rows]
        possx = m[0, :, :]  # make the x pos array
        possy = m[1, :, :]  # make the y pos array
        possx.shape = cols, rows
        possy.shape = cols, rows
        mpossx = self.whisker_roi.getArrayRegion(possx, self.mousecam_display_view.imageItem).astype(int)
        mpossx = mpossx[np.nonzero(mpossx)]  # get the x pos from ROI
        mpossy = self.whisker_roi.getArrayRegion(possy, self.mousecam_display_view.imageItem).astype(int)
        mpossy = mpossy[np.nonzero(mpossy)]  # get the y pos from ROI


        face_pixels = np.array([mpossx, mpossy])
        print("Face Pixels", np.shape(face_pixels))

        # Save These
        save_directory = os.path.join(self.session_list[self.current_session_index], "Mousecam_Analysis")
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)
      
        np.save(os.path.join(save_directory, "Face_Pixels.npy"), face_pixels)
        print("Saved!", os.path.join(save_directory, "Face_Pixels.npy"))

    """
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

        p = Path(polygon_verticies)  # make a polygon
        grid = p.contains_points(points)
        grid = np.reshape(grid, (image_height, image_width))

        whisker_coords = np.nonzero(grid)

        # Save These
        save_directory = os.path.join(self.session_list[self.current_session_index], "Mousecam_Analysis")
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)
  
        np.save(os.path.join(save_directory, "Face_Pixels.npy"), whisker_coords)
        """

if __name__ == '__main__':

    app = QApplication(sys.argv)

    session_list = [
    r"NRXN78.1A/2020_11_28_Switching_Imaging",
    r"NRXN78.1A/2020_12_05_Switching_Imaging",
    r"NRXN78.1A/2020_12_09_Switching_Imaging",

    r"NRXN78.1D/2020_12_07_Switching_Imaging",
    r"NRXN78.1D/2020_11_29_Switching_Imaging",
    r"NRXN78.1D/2020_12_05_Switching_Imaging",

    r"NXAK14.1A/2021_05_21_Switching_Imaging",
    r"NXAK14.1A/2021_05_23_Switching_Imaging",
    r"NXAK14.1A/2021_06_11_Switching_Imaging",
    r"NXAK14.1A/2021_06_13_Transition_Imaging",
    r"NXAK14.1A/2021_06_15_Transition_Imaging",
    r"NXAK14.1A/2021_06_17_Transition_Imaging",

    r"NXAK22.1A/2021_10_14_Switching_Imaging",
    r"NXAK22.1A/2021_10_20_Switching_Imaging",
    r"NXAK22.1A/2021_10_22_Switching_Imaging",
    r"NXAK22.1A/2021_10_29_Transition_Imaging",
    r"NXAK22.1A/2021_11_03_Transition_Imaging",
    r"NXAK22.1A/2021_11_05_Transition_Imaging",

    r"NXAK4.1B/2021_03_02_Switching_Imaging",
    r"NXAK4.1B/2021_03_04_Switching_Imaging",
    r"NXAK4.1B/2021_03_06_Switching_Imaging",
    r"NXAK4.1B/2021_04_02_Transition_Imaging",
    r"NXAK4.1B/2021_04_08_Transition_Imaging",
    r"NXAK4.1B/2021_04_10_Transition_Imaging",

    r"NXAK7.1B/2021_02_26_Switching_Imaging",
    r"NXAK7.1B/2021_02_28_Switching_Imaging",
    r"NXAK7.1B/2021_03_02_Switching_Imaging",
    r"NXAK7.1B/2021_03_23_Transition_Imaging",
    r"NXAK7.1B/2021_03_31_Transition_Imaging",
    r"NXAK7.1B/2021_04_02_Transition_Imaging",

    r"NRXN78.1A/2020_11_14_Discrimination_Imaging",
    r"NRXN78.1A/2020_11_15_Discrimination_Imaging",
    r"NRXN78.1A/2020_11_24_Discrimination_Imaging",
    r"NRXN78.1A/2020_11_21_Discrimination_Imaging",

    r"NRXN78.1D/2020_11_14_Discrimination_Imaging",
    r"NRXN78.1D/2020_11_15_Discrimination_Imaging",
    r"NRXN78.1D/2020_11_25_Discrimination_Imaging",
    r"NRXN78.1D/2020_11_23_Discrimination_Imaging",

    r"NXAK4.1B/2021_02_04_Discrimination_Imaging",
    r"NXAK4.1B/2021_02_06_Discrimination_Imaging",
    r"NXAK4.1B/2021_02_22_Discrimination_Imaging",
    r"NXAK4.1B/2021_02_14_Discrimination_Imaging",

    r"NXAK7.1B/2021_02_01_Discrimination_Imaging",
    r"NXAK7.1B/2021_02_03_Discrimination_Imaging",
    r"NXAK7.1B/2021_02_24_Discrimination_Imaging",
    r"NXAK7.1B/2021_02_22_Discrimination_Imaging",

    r"NXAK14.1A/2021_04_29_Discrimination_Imaging",
    r"NXAK14.1A/2021_05_01_Discrimination_Imaging",
    r"NXAK14.1A/2021_05_09_Discrimination_Imaging",
    r"NXAK14.1A/2021_05_07_Discrimination_Imaging",

    r"NXAK22.1A/2021_09_25_Discrimination_Imaging",
    r"NXAK22.1A/2021_09_29_Discrimination_Imaging",
    r"NXAK22.1A/2021_10_08_Discrimination_Imaging",
    r"NXAK22.1A/2021_10_07_Discrimination_Imaging",

     ]


    full_session_list = []
    for item in session_list:
        full_session_list.append(os.path.join("/media/matthew/Expansion/Control_Data", item))


    full_session_list = [

    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_13_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_15_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_17_Switching_Imaging",

    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_31_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_02_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_04_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_08_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_10_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_12_Transition_Imaging",

    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_20_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_22_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_24_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_14_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_16_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_18_Transition_Imaging",

    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_17_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_19_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_23_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_30_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_06_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_08_Transition_Imaging",

    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_15_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_17_Switching_Imaging", ## here on 28/01
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_19_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_22_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_24_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_26_Transition_Imaging",

    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_14_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_20_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_26_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_05_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_08_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_10_Transition_Imaging",
    ]

    full_session_list = [

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_14_Discrimination_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_09_Discrimination_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_04_Discrimination_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_02_Discrimination_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_14_Discrimination_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_02_Discrimination_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_15_Discrimination_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_30_Discrimination_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_22_Discrimination_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_08_Discrimination_Imaging",
    ]

    full_session_list = ["/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_15_Discrimination_Imaging"]


    full_session_list = [

        "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_17_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_19_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_21_Discrimination_Imaging",

        "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_21_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_23_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_25_Discrimination_Imaging",

        "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_14_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_22_Discrimination_Imaging",

        "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_24_Discrimination_Imaging",

        "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_05_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_07_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_09_Discrimination_Imaging",

        "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_07_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging",

        "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_05_Discrimination_Imaging",

        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_03_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging",

        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_12_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_14_Discrimination_Imaging",

        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_15_Discrimination_Imaging",

        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_11_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_13_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_15_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_17_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging",

        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_02_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_04_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_06_Discrimination_Imaging",
        ]



    full_session_list = [

        "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_15_Discrimination_Imaging",

        "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_04_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_06_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_08_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_10_Discrimination_Imaging",

        "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_01_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_03_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_05_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_07_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_09_Discrimination_Imaging",

        "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_04_29_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_01_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_03_Discrimination_Imaging",

        "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_25_Discrimination_Imaging",

        "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_13_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_14_Discrimination_Imaging",
        #"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_15_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_16_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_17_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_19_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_21_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_23_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_25_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_27_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_29_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_01_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_03_Discrimination_Imaging",

        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_02_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_04_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_06_Discrimination_Imaging",

        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_04_30_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_02_Discrimination_Imaging",

        #"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_04_30_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_02_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_04_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_06_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_08_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_10_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_12_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_14_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_16_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_18_Discrimination_Imaging",

        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_28_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_30_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_02_Discrimination_Imaging",

        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_20_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_22_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_24_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_26_Discrimination_Imaging",

    ]

    full_session_list = ["/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_09_Discrimination_Imaging"]

    selection_window = face_selection_window(full_session_list)
    selection_window.show()

    app.exec_()

