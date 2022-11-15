import numpy as np
import pyqtgraph

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import os

import cv2
import tables
import sys
from tqdm import tqdm

pyqtgraph.setConfigOptions(imageAxisOrder='row-major')


def get_eyecam_filename(base_directory):
    file_list = os.listdir(base_directory)
    for file_name in file_list:
        if "_cam_2" in file_name:
            return file_name

def load_eyecam_video(base_directory):

    # Get Eyecam Filepath
    eyecam_filename = get_eyecam_filename(base_directory)
    eyecam_file_path = os.path.join(base_directory, eyecam_filename)

    # Open Video File
    cap = cv2.VideoCapture(eyecam_file_path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Extract Selected Frames
    eyecam_data = []
    print("Extracting Eyecam Data")
    for frame_index in tqdm(range(frameCount)):
        ret, frame = cap.read()
        frame = frame[:, :, 0]
        eyecam_data.append(frame)

    cap.release()
    eyecam_data = np.array(eyecam_data)
    return eyecam_data


def load_eye_movement_data(base_directory):
    eye_movements = np.load(os.path.join(base_directory,    "Eyecam_Analysis", "eye_movements.npy"))
    movement_events = np.load(os.path.join(base_directory,    "Eyecam_Analysis", "eye_movement_events.npy"))
    blinks = np.load(os.path.join(base_directory,           "Eyecam_Analysis", "blinks.npy"))
    pupil_top_p = np.load(os.path.join(base_directory,      "Eyecam_Analysis", "pupil_top_p.npy"))
    pupil_bottom_p = np.load(os.path.join(base_directory,   "Eyecam_Analysis", "pupil_bottom_p.npy"))
    pupil_left_p = np.load(os.path.join(base_directory,     "Eyecam_Analysis", "pupil_left_p.npy"))
    pupil_right_p = np.load(os.path.join(base_directory,    "Eyecam_Analysis", "pupil_right_p.npy"))

    eye_movements = np.divide(eye_movements, np.max(eye_movements))
    return eye_movements, movement_events, blinks, pupil_top_p, pupil_bottom_p, pupil_left_p, pupil_right_p



class eye_movement_viewer(QWidget):

    def __init__(self, base_directory, parent=None):
        super(eye_movement_viewer, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("View Eye Movements")
        self.setGeometry(0, 0, 1500, 500)

        # Create Variable Holders
        self.base_directory = base_directory
        self.current_frame = 0
        self.eyecam_data = load_eyecam_video(self.base_directory)
        self.number_of_frames, self.video_height, self.video_width = np.shape(self.eyecam_data)
        self.eye_movements, self.movement_events, self.blinks, self.pupil_top_p, self.pupil_bottom_p, self.pupil_left_p, self.pupil_right_p = load_eye_movement_data(self.base_directory)
        # Create Widgets

        # Current Session Label
        self.session_label = QLabel("Session: " + str(self.base_directory))

        # Eye Threshold Display Widget
        self.eye_display_view_widget = QWidget()
        self.eye_display_view_widget_layout = QGridLayout()
        self.eye_display_view = pyqtgraph.ImageView()
        self.eye_display_view_widget_layout.addWidget(self.eye_display_view, 0, 0)
        self.eye_display_view_widget.setLayout(self.eye_display_view_widget_layout)
        self.eye_display_view_widget.setMinimumWidth(400)
        
        # Eye Movement Graph Display View
        self.graph_display_view_widget = QWidget()
        self.graph_display_view_widget_layout = QGridLayout()
        self.graph_display_view = pyqtgraph.PlotWidget()
        self.graph_display_view_widget_layout.addWidget(self.graph_display_view, 0, 0)
        self.graph_display_view_widget.setLayout(self.graph_display_view_widget_layout)
        self.graph_display_view_widget.setMinimumWidth(400)
        self.eye_movement_line = QGraphicsLineItem

        self.current_frame_line = pyqtgraph.InfiniteLine()
        self.current_frame_line.setAngle(90)

        self.eye_movement_curve = pyqtgraph.PlotCurveItem(pen='b')
        self.movement_event_curve = pyqtgraph.PlotCurveItem(pen='g')
        self.pupil_probability_curve = pyqtgraph.PlotCurveItem()
        self.blink_curve = pyqtgraph.PlotCurveItem(pen='r')

        self.graph_display_view.addItem(self.current_frame_line)
        self.graph_display_view.addItem(self.eye_movement_curve)
        #self.graph_display_view.addItem(self.movement_event_curve)
        self.graph_display_view.addItem(self.blink_curve)
        #self.graph_display_view.addItem(self.pupil_probability_curve)

        self.eye_movement_curve.setData(self.eye_movements)
        self.blink_curve.setData(self.blinks)
        self.pupil_probability_curve.setData(self.pupil_top_p)
        self.movement_event_curve.setData(self.movement_events)

        # Timing Slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.number_of_frames-1)
        self.frame_slider.valueChanged.connect(self.change_frame)

        self.next_frame_button = QPushButton("Next Frame")
        self.next_frame_button.clicked.connect(self.next_frame)

        self.previous_frame_buttom = QPushButton("Previous Frame")
        self.previous_frame_buttom.clicked.connect(self.previous_frame)

        # Create Layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Add Transformation Widgets
        self.layout.addWidget(self.session_label, 0, 0, 1, 2)
        self.layout.addWidget(self.eye_display_view_widget, 1, 0, 1, 1)
        self.layout.addWidget(self.graph_display_view_widget, 1, 1, 1, 1)
        self.layout.addWidget(self.frame_slider, 2, 0, 1, 2)
        self.layout.addWidget(self.next_frame_button, 3, 0, 1,1)
        self.layout.addWidget(self.previous_frame_buttom, 4, 0, 1, 1)

        # Plot First Item
        """
        self.lick_threshold_line = pyqtgraph.InfiniteLine()
        self.lick_threshold_line.setAngle(0)
        self.lick_threshold_line.setValue(self.lick_threshold)
        self.lick_trace_curve = pyqtgraph.PlotCurveItem()
        self.lick_display_view.addItem(self.lick_threshold_line)
        self.lick_display_view.addItem(self.lick_trace_curve)
        self.load_session()
        """
        self.show()

    def change_frame(self):
        window_size=1000
        self.current_frame = int(self.frame_slider.value())
        self.eye_display_view.setImage(self.eyecam_data[self.current_frame])
        self.current_frame_line.setValue(self.current_frame)
        self.graph_display_view.setXRange(self.current_frame-window_size, self.current_frame+window_size)

    def next_frame(self):
        self.current_frame = int(self.frame_slider.value())
        self.frame_slider.setValue(self.current_frame + 1)

    def previous_frame(self):
        self.current_frame = int(self.frame_slider.value())
        self.frame_slider.setValue(self.current_frame - 1)

def view_eye_movements(base_directory):
    app = QApplication(sys.argv)

    window = eye_movement_viewer(base_directory)
    window.show()

    app.exec_()


base_directory = r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_17_Switching_Imaging"
view_eye_movements(base_directory)
