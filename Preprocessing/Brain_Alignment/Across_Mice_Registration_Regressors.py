import numpy as np
import matplotlib.pyplot as plt
import tables
from scipy import ndimage
from skimage.feature import canny
from skimage.filters import sobel
import math
from pathlib import Path

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph
from skimage.transform import warp, resize, rescale

import os
import sys

import Registration_Utils

pyqtgraph.setConfigOptions(imageAxisOrder='row-major')

def combine_elements_into_path(list):
    current_path = list[0]
    for item in list[1:]:
        current_path = os.path.join(current_path, item)
    return current_path

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


def get_mask_contours(sign_map):

    positive_sign = np.where(sign_map > 0, sign_map, 0)
    negative_sign = np.where(sign_map < 0, sign_map, 0)
    negative_sign = np.abs(negative_sign)

    smoothed_positive_image = ndimage.gaussian_filter(positive_sign, sigma=1)
    smoothed_negative_image = ndimage.gaussian_filter(negative_sign, sigma=1)

    positive_edges = np.subtract(positive_sign, smoothed_positive_image)
    negative_edges = np.subtract(negative_sign, smoothed_negative_image)

    negative_edges = np.abs(negative_edges)
    positive_edges = np.abs(positive_edges)

    return positive_sign, negative_sign




class masking_window(QWidget):

    def __init__(self, mouse_list, parent=None):
        super(masking_window, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("Across Mice Regressor Alignment")
        self.setGeometry(0, 0, 1900, 500)

        # Setup Internal Variables
        self.session_list = []
        self.image_height = 300
        self.image_width = 304
        self.number_of_sessions = len(mouse_list)
        self.current_template_index = 0
        self.current_matching_index = 0
        self.current_regressor_inde = 0

        # Get List Of Regressor Maps
        self.regressor_map_list = []
        for mouse in mouse_list:
            regressor_maps = load_mouse_regressors(mouse)
            self.regressor_map_list.append(regressor_maps)

        # Get List Of Root Directories
        self.root_directory_list = []
        for mouse in mouse_list:
            root_directory = Path(mouse[0]).parts[0:-1]
            root_directory = combine_elements_into_path(root_directory)
            self.root_directory_list.append(root_directory)

        # Get List Of Alignment Dictionaries
        self.variable_dictionary_list = []
        for mouse_index in range(self.number_of_sessions):

            variable_dictionary_directory = os.path.join(self.root_directory_list[mouse_index], "Across_Mouse_Alignment_Dictionary.npy")
            print("Varianble dictionary directory", variable_dictionary_directory)

            if os.path.exists(variable_dictionary_directory):
                print("IT Exists")
                variable_dictionary = np.load(variable_dictionary_directory, allow_pickle=True)[()]
            else:
                variable_dictionary = {'x_shift': 0, 'y_shift': 0, 'rotation': 0, "zoom": 0}

            self.variable_dictionary_list.append(variable_dictionary)

        # Set Current Images
        #self.max_projection = self.max_projection_list[0]
        #self.template_edges = self.edges_list[0]
        #self.matching_edges = self.edges_list[0]

        # Regressor Display View
        self.lick_display_view_widget = QWidget()
        self.lick_display_view_widget_layout = QGridLayout()
        self.lick_display_view = pyqtgraph.ImageView()
        self.lick_display_view.ui.histogram.hide()
        self.lick_display_view.ui.roiBtn.hide()
        self.lick_display_view.ui.menuBtn.hide()
        self.lick_display_view_widget_layout.addWidget(self.lick_display_view, 0, 0)
        self.lick_display_view_widget.setLayout(self.lick_display_view_widget_layout)
        self.lick_display_view_widget.setFixedWidth(self.image_width)
        self.lick_display_view_widget.setFixedHeight(self.image_height)

        self.running_display_view_widget = QWidget()
        self.running_display_view_widget_layout = QGridLayout()
        self.running_display_view = pyqtgraph.ImageView()
        self.running_display_view.ui.histogram.hide()
        self.running_display_view.ui.roiBtn.hide()
        self.running_display_view.ui.menuBtn.hide()
        self.running_display_view_widget_layout.addWidget(self.running_display_view, 0, 0)
        self.running_display_view_widget.setLayout(self.running_display_view_widget_layout)
        self.running_display_view_widget.setFixedWidth(self.image_width)
        self.running_display_view_widget.setFixedHeight(self.image_height)

        self.limbs_display_view_widget = QWidget()
        self.limbs_display_view_widget_layout = QGridLayout()
        self.limbs_display_view = pyqtgraph.ImageView()
        self.limbs_display_view.ui.histogram.hide()
        self.limbs_display_view.ui.roiBtn.hide()
        self.limbs_display_view.ui.menuBtn.hide()
        self.limbs_display_view_widget_layout.addWidget(self.limbs_display_view, 0, 0)
        self.limbs_display_view_widget.setLayout(self.limbs_display_view_widget_layout)
        self.limbs_display_view_widget.setFixedWidth(self.image_width)
        self.limbs_display_view_widget.setFixedHeight(self.image_height)

        # Create Session Labels
        self.template_session_label = QLabel("Template Session: ")
        self.matching_session_label = QLabel("Matching Session: ")

        # Create Session List Views
        self.template_session_list_widget = QListWidget()
        self.matching_session_list_widget = QListWidget()

        self.matching_session_list_widget.currentRowChanged.connect(self.load_matching_session)
        self.template_session_list_widget.currentRowChanged.connect(self.load_template_session)

        self.template_session_list_widget.setFixedWidth(200)
        self.matching_session_list_widget.setFixedWidth(200)

        # Get List Of Mice Names
        for mouse in mouse_list:
            mouse_name = Path(mouse[0]).parts[-2]
            self.session_list.append(mouse_name)
            self.template_session_list_widget.addItem(mouse_name)
            self.matching_session_list_widget.addItem(mouse_name)



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

        # Add Labels
        self.layout.addWidget(self.template_session_label, 0, 0, 1, 2)
        self.layout.addWidget(self.matching_session_label, 0, 1, 1, 2)

        # Add List Views
        self.layout.addWidget(self.template_session_list_widget, 1, 0, 25, 1)
        self.layout.addWidget(self.matching_session_list_widget, 1, 1, 25, 1)

        # Add Display Views
        self.layout.addWidget(self.lick_display_view_widget,    1, 2, 25, 1)
        self.layout.addWidget(self.running_display_view_widget, 1, 3, 25, 1)
        self.layout.addWidget(self.limbs_display_view,          1, 4, 25, 1)

        # Add Transformation Controls
        control_column = 5
        self.layout.addWidget(self.left_button, 2, control_column, 1, 1)
        self.layout.addWidget(self.right_button, 3, control_column, 1, 1)
        self.layout.addWidget(self.up_button, 4, control_column, 1, 1)
        self.layout.addWidget(self.down_button, 5, control_column, 1, 1)
        self.layout.addWidget(self.rotate_clockwise_button, 6, control_column, 1, 1)
        self.layout.addWidget(self.rotate_counterclockwise_button, 7, control_column, 1, 1)
        self.layout.addWidget(self.x_label, 8, control_column, 1, 1)
        self.layout.addWidget(self.y_label, 9, control_column, 1, 1)
        self.layout.addWidget(self.angle_label, 10, control_column, 1, 1)
        self.layout.addWidget(self.zoom_spinner, 11, control_column, 1, 1)
        self.layout.addWidget(self.map_button, 12, control_column, 1, 1)

        # Add ROI
        x_mindpoint = int(self.image_width / 2)
        print("X midpoint", x_mindpoint)
        y_roi_height = self.image_height * 0.2
        vertical_line_coords = [[x_mindpoint, 0], [x_mindpoint, self.image_height]]
        horizontal_line_coords = [[0, y_roi_height], [self.image_width, y_roi_height]]

        self.horizontal_roi = pyqtgraph.PolyLineROI(horizontal_line_coords, closed=False)
        self.vertical_roi = pyqtgraph.PolyLineROI(vertical_line_coords, closed=False)


        self.lick_display_view.addItem(self.horizontal_roi)
        self.running_display_view.addItem(self.horizontal_roi)
        self.limbs_display_view.addItem(self.horizontal_roi)

        self.lick_display_view.addItem(self.vertical_roi)
        self.running_display_view.addItem(self.vertical_roi)
        self.limbs_display_view.addItem(self.vertical_roi)

    def draw_images(self):

        # Transform Regressor Maps
        print("Regressor ma0 list liength", len(self.regressor_map_list))
        print("Individual item length", len(self.regressor_map_list[self.current_matching_index]))

        transformed_matching_lick_map       = self.transform_image(self.regressor_map_list[self.current_matching_index][0], self.variable_dictionary_list[self.current_matching_index])
        transformed_matching_running_map    = self.transform_image(self.regressor_map_list[self.current_matching_index][1], self.variable_dictionary_list[self.current_matching_index])
        transformed_matching_limb_map       = self.transform_image(self.regressor_map_list[self.current_matching_index][4], self.variable_dictionary_list[self.current_matching_index])

        transformed_template_lick_map       = self.transform_image(self.regressor_map_list[self.current_template_index][0], self.variable_dictionary_list[self.current_template_index])
        transformed_template_running_map    = self.transform_image(self.regressor_map_list[self.current_template_index][1], self.variable_dictionary_list[self.current_template_index])
        transformed_template_limb_map       = self.transform_image(self.regressor_map_list[self.current_template_index][4], self.variable_dictionary_list[self.current_template_index])

        # Combine Sign Maps
        combined_lick_map = np.zeros((300, 304, 3))
        combined_running_map = np.zeros((300, 304, 3))
        combined_limb_map = np.zeros((300, 304, 3))

        combined_lick_map[:, :, 0] = transformed_matching_lick_map
        combined_lick_map[:, :, 1] = transformed_template_lick_map

        combined_running_map[:, :, 0] = transformed_matching_running_map
        combined_running_map[:, :, 1] = transformed_template_running_map

        combined_limb_map[:, :, 0] = transformed_matching_limb_map
        combined_limb_map[:, :, 1] = transformed_template_limb_map

        # Set Images
        self.lick_display_view.setImage(combined_lick_map)
        self.running_display_view.setImage(combined_running_map)
        self.limbs_display_view.setImage(combined_limb_map)

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

        for session_index in range(self.number_of_sessions):
            # Get Save Directory
            session_directory = self.session_list[session_index]
            save_directory = os.path.join(session_directory, "Within_Mouse_Alignment_Dictionary.npy")

            # Save Dictionary
            np.save(save_directory, self.variable_dictionary_list[session_index])

    def load_matching_session(self):
        self.current_matching_index = self.matching_session_list_widget.currentRow()
        self.draw_images()

        # self.max_projection = self.max_projection_list[self.current_matching_index]
        # self.cross_display_view.setImage(max_projection)

    def load_template_session(self):
        self.current_template_index = self.template_session_list_widget.currentRow()
        self.draw_images()




def load_cpd_map(regression_dictionary, group_name, indicies, image_height, image_width, alignment_dict):

    # Load CPDs
    regression_group_names = regression_dictionary["Coef_Names"]
    regression_cpds = regression_dictionary["Coefficients_of_Partial_Determination"]
    regression_cpds = np.transpose(regression_cpds)

    # Create Image
    selected_cpd_index = regression_group_names.index(group_name)
    selected_cpd = regression_cpds[selected_cpd_index]
    selected_cpd = Registration_Utils.create_image_from_data(selected_cpd, indicies, image_height, image_width)

    # Align Image
    selected_cpd = Registration_Utils.transform_image(selected_cpd, alignment_dict)

    return selected_cpd

def get_map_edges(map):

    # Threshold Image
    map = ndimage.gaussian_filter(map, sigma=1)
    image_threshold = np.percentile(map, q=80)
    map = np.where(map > image_threshold, map, 0)
    map_edges = sobel(map)
    return map_edges


def load_mouse_regressors(session_list, visualise=True):

    mouse_lick_list = []
    mouse_running_list = []
    mouse_face_list = []
    mouse_whisk_list = []
    mouse_limbs_list = []

    for base_directory in session_list:

        # Load Regression Dict
        regression_dictionary = np.load(os.path.join(base_directory, "Regression_Coefs", "Regression_Dictionary_Simple.npy"), allow_pickle=True)[()]

        # Load Mask
        indicies, image_height, image_width = Registration_Utils.load_downsampled_mask(base_directory)

        # Load Within Mouse Alignment Dictionary
        within_mouse_alignment_dictionary = np.load(os.path.join(base_directory, "Within_Mouse_Alignment_Dictionary.npy"), allow_pickle=True)[()]

        # Load CPD Maps
        session_lick_map    = load_cpd_map(regression_dictionary, 'Lick',           indicies, image_height, image_width, within_mouse_alignment_dictionary)
        session_running_map = load_cpd_map(regression_dictionary, 'Running',        indicies, image_height, image_width, within_mouse_alignment_dictionary)
        session_face_map    = load_cpd_map(regression_dictionary, 'Face_Motion',    indicies, image_height, image_width, within_mouse_alignment_dictionary)
        session_whisk_map   = load_cpd_map(regression_dictionary, 'Whisking',       indicies, image_height, image_width, within_mouse_alignment_dictionary)
        session_limbs_map   = load_cpd_map(regression_dictionary, 'Limbs',          indicies, image_height, image_width, within_mouse_alignment_dictionary)

        # Add To List
        mouse_lick_list.append(session_lick_map)
        mouse_running_list.append(session_running_map)
        mouse_face_list.append(session_face_map)
        mouse_whisk_list.append(session_whisk_map)
        mouse_limbs_list.append(session_limbs_map)

    # Get Mean Maps
    mouse_lick_map = np.mean(mouse_lick_list, axis=0)
    mouse_running_map = np.mean(mouse_running_list, axis=0)
    mouse_face_map = np.mean(mouse_face_list, axis=0)
    mouse_whisk_map = np.mean(mouse_lick_list, axis=0)
    mouse_limbs_map = np.mean(mouse_limbs_list, axis=0)

    # Get edges
    """
    mouse_lick_map = get_map_edges(mouse_lick_map)
    mouse_running_map = get_map_edges(mouse_running_map)
    mouse_face_map = get_map_edges(mouse_face_map)
    mouse_whisk_map = get_map_edges(mouse_whisk_map)
    mouse_limbs_map = get_map_edges(mouse_limbs_map)
    """

    # Normalise
    mouse_lick_map = np.divide(mouse_lick_map, np.max(mouse_lick_map))
    mouse_running_map = np.divide(mouse_running_map, np.max(mouse_running_map))
    mouse_face_map = np.divide(mouse_face_map, np.max(mouse_face_map))
    mouse_whisk_map = np.divide(mouse_whisk_map, np.max(mouse_whisk_map))
    mouse_limbs_map = np.divide(mouse_limbs_map, np.max(mouse_limbs_map))

    # view These
    if visualise == True:
        figure_1 = plt.figure()
        rows = 1
        columns = 5
    
        lick_axis = figure_1.add_subplot(rows, columns, 1)
        running_axis = figure_1.add_subplot(rows, columns, 2)
        face_axis = figure_1.add_subplot(rows, columns, 3)
        whisk_axis = figure_1.add_subplot(rows, columns, 4)
        limbs_axis = figure_1.add_subplot(rows, columns, 5)
    
        lick_axis.imshow(mouse_lick_map)
        running_axis.imshow(mouse_running_map)
        face_axis.imshow(mouse_face_map)
        whisk_axis.imshow(mouse_whisk_map)
        limbs_axis.imshow(mouse_limbs_map)
    
        plt.show()
        
    regressor_map_list = [mouse_lick_map, mouse_running_map, mouse_face_map, mouse_whisk_map, mouse_limbs_map]
    
    return regressor_map_list

def load_atlas():

    # Load Atlas Boundaires
    allen_atlas_boundaries = np.load(r"/home/matthew/Documents/Github_Code_Clean/Transition_Analysis/Brain_Registration/Atlas_Outlines.npy")

    # Load Atlas ALignment Dict
    atlas_alignment_dictionary = np.load(r"/home/matthew/Documents/Github_Code_Clean/Transition_Analysis/Brain_Registration/Atlas_Alignment_Dictionary.npy", allow_pickle=True)[()]

    # Align Atlas
    allen_atlas_boundaries = Registration_Utils.transform_mask_or_atlas(allen_atlas_boundaries, atlas_alignment_dictionary)

    return allen_atlas_boundaries


def align_sessions(session_list):
    app = QApplication(sys.argv)

    window = masking_window(session_list)
    window.show()

    app.exec_()




mouse_list = [

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_13_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_15_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_17_Switching_Imaging"],

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_31_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_02_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_04_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_08_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_10_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_12_Transition_Imaging"],

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_20_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_22_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_24_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_14_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_16_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_18_Transition_Imaging"],

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_17_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_19_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_23_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_30_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_06_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_08_Transition_Imaging"]

]

#r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_26_Continous_Retinotopy_Left",
#r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_01_Continuous_Retinotopic_Mapping_Left",
#r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_01_Continous_Retinotopy_Left",
#r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_01_Continous_Retinotopy_Left",
#r"/media/matthew/Expansion/Control_Data/NXAK14.1A/Continous_Retinotopic_Mapping_Left"

#load_mouse_regressors(mouse_list[3])
load_atlas()
align_sessions(mouse_list)