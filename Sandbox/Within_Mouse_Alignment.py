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
from skimage.transform import warp, resize

import os
import sys

import Registration_Utils

pyqtgraph.setConfigOptions(imageAxisOrder='row-major')


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

    def __init__(self, session_list, parent=None):
        super(masking_window, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("Align To Static Cross")
        self.setGeometry(0, 0, 1900, 500)

        # Setup Internal Variables
        self.session_list = session_list
        self.number_of_sessions = len(self.session_list)
        self.current_template_index = 0
        self.current_matching_index = 0
        self.zoom_vector_u = np.load(r"/home/matthew/Documents/Github_Code_Clean/Preprocessing/Brain_Alignment/zoom_optic_flow_u.npy")
        self.zoom_vector_v = np.load(r"/home/matthew/Documents/Github_Code_Clean/Preprocessing/Brain_Alignment/zoom_optic_flow_v.npy")

        # Get List Of Max Projections
        self.max_projection_list = []
        self.edges_list = []
        for session in session_list:

            max_projection = np.load(os.path.join(session, "Blue_Example_Image.npy"))
            print("Max projector shape", np.shape(max_projection))
            upper_bound = np.percentile(max_projection, 99)
            max_projection = np.divide(max_projection, upper_bound)

            edges = canny(max_projection, sigma=1)

            max_projection = np.clip(max_projection, a_min=0, a_max=1)
            self.max_projection_list.append(max_projection)
            self.edges_list.append(edges)


        # Get List Of Alignment Dictionaries
        self.variable_dictionary_list = []
        for session_index in range(self.number_of_sessions):


            variable_dictionary_directory = os.path.join(self.session_list[session_index], "Within_Mouse_Alignment_Dictionary.npy")
            if os.path.exists(variable_dictionary_directory):
                print("IT Exists")
                variable_dictionary = np.load(variable_dictionary_directory, allow_pickle=True)[()]
            else:
                variable_dictionary = {'x_shift': 0, 'y_shift': 0, 'rotation': 0, "zoom":0}

            self.variable_dictionary_list.append(variable_dictionary)

        # Set Current Images
        self.max_projection = self.max_projection_list[0]
        self.template_edges = self.edges_list[0]
        self.matching_edges = self.edges_list[0]

        # Cross Display View
        self.cross_display_view_widget = QWidget()
        self.cross_display_view_widget_layout = QGridLayout()
        self.cross_display_view = pyqtgraph.ImageView()
        self.cross_display_view.ui.histogram.hide()
        self.cross_display_view.ui.roiBtn.hide()
        self.cross_display_view.ui.menuBtn.hide()
        self.cross_display_view_widget_layout.addWidget(self.cross_display_view, 0, 0)
        self.cross_display_view_widget.setLayout(self.cross_display_view_widget_layout)
        self.cross_display_view_widget.setFixedWidth(608)
        self.cross_display_view_widget.setFixedHeight(600)
        self.cross_display_view.setLevels(0, 1)

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

        self.zoom_spinner = QDoubleSpinBox()
        self.zoom_spinner.setRange(-5, 5)
        self.zoom_spinner.setSingleStep(0.1)
        self.zoom_spinner.valueChanged.connect(self.set_zoom)

        self.map_button = QPushButton("Set Alignment")
        self.map_button.clicked.connect(self.set_alignment)

        # Add Labels
        self.x_label = QLabel("x: 0")
        self.y_label = QLabel("y: 0")
        self.angle_label = QLabel("angle: 0")

        # Create and Set Layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Add Labels
        self.layout.addWidget(self.template_session_label,          0, 0, 1, 2)
        self.layout.addWidget(self.matching_session_label,          0, 1, 1, 2)

        # Add List Views
        self.layout.addWidget(self.template_session_list_widget,    1, 0, 25, 1)
        self.layout.addWidget(self.matching_session_list_widget,    1, 1, 25, 1)

        # Add Display Views
        self.layout.addWidget(self.edges_display_view_widget,       1, 2, 25, 1)
        self.layout.addWidget(self.cross_display_view_widget,       1, 3, 25, 1)

        # Add Transformation Controls
        control_column = 4
        self.layout.addWidget(self.left_button,                     2,  control_column, 1, 1)
        self.layout.addWidget(self.right_button,                    3,  control_column, 1, 1)
        self.layout.addWidget(self.up_button,                       4,  control_column, 1, 1)
        self.layout.addWidget(self.down_button,                     5,  control_column, 1, 1)
        self.layout.addWidget(self.rotate_clockwise_button,         6,  control_column, 1, 1)
        self.layout.addWidget(self.rotate_counterclockwise_button,  7,  control_column, 1, 1)
        self.layout.addWidget(self.x_label,                         8,  control_column, 1, 1)
        self.layout.addWidget(self.y_label,                         9,  control_column, 1, 1)
        self.layout.addWidget(self.angle_label,                     10, control_column, 1, 1)
        self.layout.addWidget(self.zoom_spinner,                    11, control_column, 1, 1)
        self.layout.addWidget(self.map_button,                      12, control_column, 1, 1)

        # Add ROI
        horizontal_line_coords = [[152, 250], [152, 12]]
        vertical_line_coords = [[101, 50], [202, 50]]

        self.horizontal_roi = pyqtgraph.PolyLineROI(horizontal_line_coords, closed=False)
        self.vertical_roi = pyqtgraph.PolyLineROI(vertical_line_coords, closed=False)

        self.cross_display_view.addItem(self.horizontal_roi)
        self.cross_display_view.addItem(self.vertical_roi)


    def draw_images(self):
        
        transformed_max_projection = self.transform_image(self.max_projection, self.variable_dictionary_list[self.current_matching_index])
        transformed_matching_edges = self.transform_image(self.matching_edges, self.variable_dictionary_list[self.current_matching_index])
        transformed_template_edges = self.transform_image(self.template_edges, self.variable_dictionary_list[self.current_template_index])

        combined_edges_template = np.zeros((300, 304, 3))
        combined_edges_template[:, :, 0] = transformed_matching_edges
        combined_edges_template[:, :, 1] = transformed_template_edges

        self.cross_display_view.setImage(transformed_max_projection)
        self.edges_display_view.setImage(combined_edges_template)


    def zoom_transform(self, image, zoom_scale_factor):

        v = self.zoom_vector_v * zoom_scale_factor
        u = self.zoom_vector_u * zoom_scale_factor

        nr, nc = image.shape
        row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
        image_warp = warp(image, np.array([row_coords + v, col_coords + u]), mode='edge')
        return image_warp

    def transform_image(self, image, variable_dictionary):
        
        # Rotate
        angle = variable_dictionary['rotation']
        x_shift = variable_dictionary['x_shift']
        y_shift = variable_dictionary['y_shift']
        zoom_factor = variable_dictionary['zoom']

        transformed_image = np.copy(image)
        transformed_image = ndimage.rotate(transformed_image, angle, reshape=False, prefilter=True)
        transformed_image = np.roll(a=transformed_image, axis=0, shift=y_shift)
        transformed_image = np.roll(a=transformed_image, axis=1, shift=x_shift)

        # Zoom
        transformed_image = self.zoom_transform(transformed_image, zoom_factor)

        return transformed_image

    def move_left(self):
        self.variable_dictionary_list[self.current_matching_index]['x_shift'] = self.variable_dictionary_list[self.current_matching_index]['x_shift'] - 1
        self.x_label.setText("x: " + str(self.variable_dictionary_list[self.current_matching_index]['x_shift']))
        self.draw_images()

    def set_zoom(self):
        self.variable_dictionary_list[self.current_matching_index]['zoom'] = self.zoom_spinner.value()
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
        self.variable_dictionary_list[self.current_matching_index]['rotation'] = self.variable_dictionary_list[self.current_matching_index]['rotation'] - 0.1
        self.angle_label.setText("Angle: " + str(np.around(self.variable_dictionary_list[self.current_matching_index]['rotation'], 2)))
        self.draw_images()

    def rotate_counterclockwise(self):
        self.variable_dictionary_list[self.current_matching_index]['rotation'] = self.variable_dictionary_list[self.current_matching_index]['rotation'] + 0.1
        self.angle_label.setText("Angle: " + str(np.around(self.variable_dictionary_list[self.current_matching_index]['rotation'], 2)))
        self.draw_images()


    def set_alignment(self):

        for session_index in  range(self.number_of_sessions):

            # Get Save Directory
            session_directory = self.session_list[session_index]
            save_directory = os.path.join(session_directory, "Within_Mouse_Alignment_Dictionary.npy")

            # Save Dictionary
            np.save(save_directory, self.variable_dictionary_list[session_index])


    def load_matching_session(self):
        
        self.current_matching_index = self.matching_session_list_widget.currentRow()
        
        self.max_projection = self.max_projection_list[self.current_matching_index]
        self.matching_edges = self.edges_list[self.current_matching_index]
        self.draw_images()
        #self.cross_display_view.setImage(max_projection)


    def load_template_session(self):

        self.current_template_index = self.template_session_list_widget.currentRow()
        self.template_edges = self.edges_list[self.current_template_index]
        self.draw_images()



def align_sessions(session_list):

    app = QApplication(sys.argv)

    window = masking_window(session_list)
    window.show()

    app.exec_()


# Create Session List
mouse_base_directory = r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Mutant_Data/NXAK4.1A"
mouse_base_directory = r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A"
mouse_base_directory = r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B"
mouse_base_directory = r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B"


subfolder_list = os.listdir(mouse_base_directory)
session_list = []
for session in subfolder_list:
    session_list.append(os.path.join(mouse_base_directory, session))
print("Session List:")
for session in session_list:
    print(session)

align_sessions(session_list)