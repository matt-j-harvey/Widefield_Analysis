import tables
import os
import matplotlib.pyplot as plt
from skimage.feature import canny
from scipy import ndimage
import numpy as np


import numpy as np
import matplotlib.pyplot as plt
import tables
from scipy import ndimage
from skimage.feature import canny
import math

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph

import os
import sys


pyqtgraph.setConfigOptions(imageAxisOrder='row-major')


def load_images(file_list):

    array_list = []
    for filename in file_list:
        datacontainer = tables.open_file(filename, mode="r")
        data = datacontainer.root["blue"][0]
        array_list.append(data)

    return array_list


def extract_holes(image_list):

    processed_image_list = []
    for image in image_list:
        edges = canny(image, sigma=2)
        processed_image_list.append(edges)
        #plt.imshow(edges)
        #plt.show()
    return processed_image_list

def view_images(image_list):

    rows = 1
    columns = 4
    count = 1
    figure_1 = plt.figure()
    for image in image_list:
        axis = figure_1.add_subplot(rows, columns, count)
        axis.imshow(image)
        count += 1

    plt.show()



def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    oy, ox = origin
    py, px = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qy, qx


class masking_window(QWidget):

    def __init__(self, session_list, edge_list, parent=None):
        super(masking_window, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("Align Different Zooms")
        self.setGeometry(0, 0, 1900, 500)

        # Setup Internal Variables
        self.session_list = session_list
        self.number_of_sessions = len(self.session_list)
        self.current_template_index = 0
        self.current_matching_index = 0

        # Get List Of Max Projections
        self.edges_list = edge_list

        # Get List Of Alignment Dictionaries
        self.variable_dictionary_list = []
        for session_index in range(self.number_of_sessions):
            variable_dictionary = {'x_shift': 0, 'y_shift': 0, 'rotation': 0}
            self.variable_dictionary_list.append(variable_dictionary)

        # Set Current Images
        self.template_edges = self.edges_list[0]
        self.matching_edges = self.edges_list[0]

        # Regressor Display View
        self.edges_display_view_widget = QWidget()
        self.edges_display_view_widget_layout = QGridLayout()
        self.edges_display_view = pyqtgraph.ImageView()
        self.edges_display_view.ui.histogram.hide()
        self.edges_display_view.ui.roiBtn.hide()
        self.edges_display_view.ui.menuBtn.hide()
        self.edges_display_view_widget_layout.addWidget(self.edges_display_view, 0, 0)
        self.edges_display_view_widget.setLayout(self.edges_display_view_widget_layout)
        self.edges_display_view_widget.setFixedWidth(608)
        self.edges_display_view_widget.setFixedHeight(600)

        # Create Session Labels
        self.template_session_label = QLabel("Template Session: ")
        self.matching_session_label = QLabel("Matching Session: ")

        # Create Session List Views
        self.template_session_list_widget = QListWidget()
        self.matching_session_list_widget = QListWidget()

        self.matching_session_list_widget.currentRowChanged.connect(self.load_matching_session)
        self.template_session_list_widget.currentRowChanged.connect(self.load_template_session)

        for session in self.session_list:
            split_session = session.split('/')
            session_name = split_session[-2] + "_" + split_session[-1]
            self.template_session_list_widget.addItem(session_name)
            self.matching_session_list_widget.addItem(session_name)

        self.matching_session_list_widget.setCurrentRow(self.current_matching_index)

        # Create Transformation Buttons
        self.left_button = QPushButton("Left")
        self.left_button.clicked.connect(self.move_left)

        self.right_button = QPushButton("Right")
        self.right_button.clicked.connect(self.move_right)

        self.up_button = QPushButton("Up")
        self.up_button.clicked.connect(self.move_up)

        self.down_button = QPushButton("Down")
        self.down_button.clicked.connect(self.move_down)

        self.rotate_clockwise_button = QPushButton("Rotate Clockwise")
        self.rotate_clockwise_button.clicked.connect(self.rotate_clockwise)

        self.rotate_counterclockwise_button = QPushButton("Rotate Counterclockwise")
        self.rotate_counterclockwise_button.clicked.connect(self.rotate_counterclockwise)

        self.map_button = QPushButton("Set Alignment")
        self.map_button.clicked.connect(self.set_alignment)

        # Add Labels
        self.x_label = QLabel("x: 0")
        self.y_label = QLabel("y: 0")
        self.angle_label = QLabel("angle: 0")

        # Create and Set Layout]
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Add Labels
        self.layout.addWidget(self.template_session_label, 0, 0, 1, 2)
        self.layout.addWidget(self.matching_session_label, 0, 1, 1, 2)

        # Add List Views
        self.layout.addWidget(self.template_session_list_widget, 1, 0, 25, 1)
        self.layout.addWidget(self.matching_session_list_widget, 1, 1, 25, 1)

        # Add Display Views
        self.layout.addWidget(self.edges_display_view_widget, 1, 2, 25, 1)

        # Add Transformation Controls
        control_column = 4
        self.layout.addWidget(self.left_button,                     2, control_column, 1, 1)
        self.layout.addWidget(self.right_button,                    3, control_column, 1, 1)
        self.layout.addWidget(self.up_button,                       4, control_column, 1, 1)
        self.layout.addWidget(self.down_button,                     5, control_column, 1, 1)
        self.layout.addWidget(self.rotate_clockwise_button,         6, control_column, 1, 1)
        self.layout.addWidget(self.rotate_counterclockwise_button,  7, control_column, 1, 1)
        self.layout.addWidget(self.x_label,                         8, control_column, 1, 1)
        self.layout.addWidget(self.y_label,                         9, control_column, 1, 1)
        self.layout.addWidget(self.angle_label,                     10, control_column, 1, 1)
        self.layout.addWidget(self.map_button,                      11, control_column, 1, 1)



    def draw_images(self):
        transformed_matching_edges = self.transform_image(self.matching_edges, self.variable_dictionary_list[self.current_matching_index])
        transformed_template_edges = self.transform_image(self.template_edges, self.variable_dictionary_list[self.current_template_index])

        combined_edges_template = np.zeros((600, 608, 3))
        combined_edges_template[:, :, 0] = transformed_matching_edges
        combined_edges_template[:, :, 1] = transformed_template_edges

        self.edges_display_view.setImage(combined_edges_template)

    def transform_image(self, image, variable_dictionary):

        # Rotate
        angle = variable_dictionary['rotation']
        x_shift = variable_dictionary['x_shift']
        y_shift = variable_dictionary['y_shift']

        transformed_image = np.copy(image)
        transformed_image = ndimage.rotate(transformed_image, angle, reshape=False, prefilter=True)
        transformed_image = np.roll(a=transformed_image, axis=0, shift=y_shift)
        transformed_image = np.roll(a=transformed_image, axis=1, shift=x_shift)

        return transformed_image

    def move_left(self):
        self.variable_dictionary_list[self.current_matching_index]['x_shift'] = self.variable_dictionary_list[self.current_matching_index]['x_shift'] - 1
        self.x_label.setText("x: " + str(self.variable_dictionary_list[self.current_matching_index]['x_shift']))
        self.draw_images()

    def move_right(self):
        self.variable_dictionary_list[self.current_matching_index]['x_shift'] = self.variable_dictionary_list[self.current_matching_index]['x_shift'] + 1
        self.x_label.setText("x: " + str(self.variable_dictionary_list[self.current_matching_index]['x_shift']))
        self.draw_images()

    def move_up(self):
        self.variable_dictionary_list[self.current_matching_index]['y_shift'] = self.variable_dictionary_list[self.current_matching_index]['y_shift'] - 1
        self.y_label.setText("y: " + str(self.variable_dictionary_list[self.current_matching_index]['y_shift']))
        self.draw_images()

    def move_down(self):
        self.variable_dictionary_list[self.current_matching_index]['y_shift'] = self.variable_dictionary_list[self.current_matching_index]['y_shift'] + 1
        self.y_label.setText("y: " + str(self.variable_dictionary_list[self.current_matching_index]['y_shift']))
        self.draw_images()

    def rotate_clockwise(self):
        self.variable_dictionary_list[self.current_matching_index]['rotation'] = self.variable_dictionary_list[self.current_matching_index]['rotation'] - 0.5
        self.angle_label.setText("Angle: " + str(self.variable_dictionary_list[self.current_matching_index]['rotation']))
        self.draw_images()

    def rotate_counterclockwise(self):
        self.variable_dictionary_list[self.current_matching_index]['rotation'] = self.variable_dictionary_list[self.current_matching_index]['rotation'] + 0.5
        self.angle_label.setText("Angle: " + str(self.variable_dictionary_list[self.current_matching_index]['rotation']))
        self.draw_images()

    def set_alignment(self):

        for session_index in range(self.number_of_sessions):
            # Get Save Directory
            session_directory = self.session_list[session_index]
            save_directory = os.path.join(session_directory, "Cluster_Alignment_Dictionary.npy")

            # Save Dictionary
            np.save(save_directory, self.variable_dictionary_list[session_index])

    def load_matching_session(self):
        self.current_matching_index = self.matching_session_list_widget.currentRow()
        self.matching_edges = self.edges_list[self.current_matching_index]
        self.draw_images()

    def load_template_session(self):
        self.current_template_index = self.template_session_list_widget.currentRow()
        self.template_edges = self.edges_list[self.current_template_index]
        self.draw_images()



def align_sessions(session_list, file_list):

    app = QApplication(sys.argv)

    image_list = load_images(file_list)
    edge_list = extract_holes(image_list)
    window = masking_window(session_list, edge_list)
    window.show()

    app.exec_()



file_list =  ["/media/matthew/External_Harddrive_1/Zoom_Calibration/1_Grid_Infinity/1/Grid_Infinity_2_20221021-165643_widefield.h5",
                    "/media/matthew/External_Harddrive_1/Zoom_Calibration/2_Grid_12_4/1/Grid_4_2_4_20221021-170208_widefield.h5",
                    "/media/matthew/External_Harddrive_1/Zoom_Calibration/3_Grid_7_2/1/Grid_7_2_20221021-170824_widefield.h5",
                    "/media/matthew/External_Harddrive_1/Zoom_Calibration/4_Grid_4_1_2/1/Grid_4_1_2_20221021-171202_widefield.h5"]

session_list = ["/media/matthew/External_Harddrive_1/Zoom_Calibration/1_Grid_Infinity/1",
                    "/media/matthew/External_Harddrive_1/Zoom_Calibration/2_Grid_12_4/1",
                    "/media/matthew/External_Harddrive_1/Zoom_Calibration/3_Grid_7_2/1",
                    "/media/matthew/External_Harddrive_1/Zoom_Calibration/4_Grid_4_1_2/1"]

align_sessions(session_list, file_list)