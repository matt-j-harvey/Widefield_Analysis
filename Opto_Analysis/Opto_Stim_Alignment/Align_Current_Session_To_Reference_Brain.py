import numpy as np
import matplotlib.pyplot as plt
import tables
from scipy import ndimage
from skimage.feature import canny
import math
import pkg_resources

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph
from skimage.transform import warp, resize, rescale

import os
import sys


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

    def __init__(self, template_session, matching_session, parent=None):
        super(masking_window, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("Align Session To Template")
        self.setGeometry(0, 0, 1900, 500)

        # Setup Internal Variables
        self.template_session = template_session
        self.matching_session = matching_session
        self.template_image = np.load(os.path.join(self.template_session, "Combined_Sign_map_Anatomy_Only.npy"))
        self.matching_image = np.load(os.path.join(self.matching_session, "Blue_Example_Image_Full_Size.npy"))

        # Get List Of Alignment Dictionaries
        self.variable_dictionary = {'x_shift': 0, 'y_shift': 0, 'rotation': 0, "zoom":0}

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

        # Get Session Names
        self.matching_session_name = self.get_session_name(self.matching_session)
        self.template_session_name = self.get_session_name(self.template_session)


        # Create Session List Views
        self.session_list_widget = QListWidget()
        self.session_list_widget.addItem("Template: " + str(self.template_session_name))
        self.session_list_widget.addItem("Matching_Session: " + str(self.matching_session_name))
        self.session_list_widget.currentRowChanged.connect(self.draw_images)


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
        self.zoom_spinner.setSingleStep(0.01)
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

        # Add List Views
        self.layout.addWidget(self.session_list_widget,    1, 0, 25, 1)

        # Add Display Views
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


    def get_session_name(self, session_directory):
        split_session = session_directory.split('/')
        session_name = split_session[-2] + "_" + split_session[-1]
        return session_name



    def draw_images(self):

        current_selected_session = int(self.session_list_widget.currentRow())
        print("Current Selected Session", current_selected_session)

        if current_selected_session == 0:
            self.cross_display_view.setImage(self.matching_image)

            #transformed_max_projection = self.transform_image(self.matching_image, self.variable_dictionary)
            #self.cross_display_view.setImage(transformed_max_projection)

        elif current_selected_session == 1:
            #self.cross_display_view.setImage(self.template_image)

            transformed_template = self.transform_image(self.template_image, self.variable_dictionary)
            self.cross_display_view.setImage(transformed_template)

    def zoom_transform(self, image, zoom_scale_factor):

        v = self.zoom_vector_v * zoom_scale_factor
        u = self.zoom_vector_u * zoom_scale_factor

        nr, nc = image.shape
        row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
        image_warp = warp(image, np.array([row_coords + v, col_coords + u]), mode='edge')

        return image_warp

    def transform_image(self, image, variable_dictionary):

        # Settings
        background_size = 1000
        background_offset = 200
        origional_height, origional_width = np.shape(image)
        window_y_start = background_offset
        window_y_stop = window_y_start + origional_height
        window_x_start = background_offset
        window_x_stop = window_x_start + origional_width

        # Unpack Transformation Details
        angle = variable_dictionary['rotation']
        x_shift = variable_dictionary['x_shift']
        y_shift = variable_dictionary['y_shift']
        scale_factor = variable_dictionary['zoom']

        # Copy
        transformed_image = np.copy(image)

        # Scale
        transformed_image = rescale(transformed_image, 1 + scale_factor, anti_aliasing=False, preserve_range=True)

        # Rotate
        transformed_image = ndimage.rotate(transformed_image, angle, reshape=False, prefilter=True)

        # Translate
        background = np.zeros((background_size, background_size))
        new_height, new_width = np.shape(transformed_image)

        y_start = background_offset + y_shift
        y_stop = y_start + new_height

        x_start = background_offset + x_shift
        x_stop = x_start + new_width

        background[y_start:y_stop, x_start:x_stop] = transformed_image

        # Get Normal Sized Window
        transformed_image = background[window_y_start:window_y_stop, window_x_start:window_x_stop]

        return transformed_image

    def move_left(self):
        self.variable_dictionary['x_shift'] = self.variable_dictionary['x_shift'] - 1
        self.x_label.setText("x: " + str(self.variable_dictionary['x_shift']))
        self.draw_images()

    def set_zoom(self):
        self.variable_dictionary['zoom'] = self.zoom_spinner.value()
        self.draw_images()

    def move_right(self):
        self.variable_dictionary['x_shift'] = self.variable_dictionary['x_shift'] + 1
        self.x_label.setText("x: " + str(self.variable_dictionary['x_shift']))
        self.draw_images()

    def move_up(self):
        self.variable_dictionary['y_shift'] = self.variable_dictionary['y_shift'] - 1
        self.y_label.setText("y: " + str(self.variable_dictionary['y_shift']))
        self.draw_images()

    def move_down(self):
        self.variable_dictionary['y_shift'] = self.variable_dictionary['y_shift'] + 1
        self.y_label.setText("y: " + str(self.variable_dictionary['y_shift']))
        self.draw_images()

    def rotate_clockwise(self):
        self.variable_dictionary['rotation'] = self.variable_dictionary['rotation'] - 0.1
        self.angle_label.setText("Angle: " + str(np.around(self.variable_dictionary['rotation'], 2)))
        self.draw_images()

    def rotate_counterclockwise(self):
        self.variable_dictionary['rotation'] = self.variable_dictionary['rotation'] + 0.1
        self.angle_label.setText("Angle: " + str(np.around(self.variable_dictionary['rotation'], 2)))
        self.draw_images()


    def set_alignment(self):
        save_directory = os.path.join(self.matching_session, "Opto_Atlas_Alignment_Dictionary.npy")
        np.save(save_directory, self.variable_dictionary)





def align_sessions(template_session, matching_session):

    app = QApplication(sys.argv)

    window = masking_window(template_session, matching_session)
    window.show()

    app.exec_()


template_session = r"/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.1D/2022_12_14_Retinotopy_Left"
matching_session = r"/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.1D/2023_02_27_Switching_v1_inhibition"
align_sessions(template_session, matching_session)