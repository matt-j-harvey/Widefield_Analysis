import numpy as np
import matplotlib.pyplot as plt
import tables
from scipy import ndimage
from skimage.feature import canny
from skimage.filters import sobel
import math
from pathlib import Path
import pkg_resources

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


def create_display_view(image_width=304, image_height=300):
    display_view_widget = QWidget()
    display_view_widget_layout = QGridLayout()
    display_view = pyqtgraph.ImageView()
    display_view.ui.histogram.hide()
    display_view.ui.roiBtn.hide()
    display_view.ui.menuBtn.hide()
    display_view_widget_layout.addWidget(display_view, 0, 0)
    display_view_widget.setLayout(display_view_widget_layout)
    display_view_widget.setFixedWidth(image_width)
    display_view_widget.setFixedHeight(image_height)
    return display_view, display_view_widget


class masking_window(QWidget):

    def __init__(self, mouse_list, parent=None):
        super(masking_window, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("Across Mice Regressor Alignment")
        self.setGeometry(0, 0, 1200, 500)

        # Setup Internal Variables
        self.session_list = []
        self.image_height = 300
        self.image_width = 304
        self.display_view_scale = 1.5
        self.number_of_sessions = len(mouse_list)
        self.current_template_index = 0
        self.current_matching_index = 0
        self.current_regressor_inde = 0
        self.atlas_outlines = load_atlas()

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

        # Load Retinotopy
        for mouse_index in range(self.number_of_sessions):
            combined_retinotopy_directory = os.path.join(self.root_directory_list[mouse_index], "Combined_Retinotopy")

            if os.path.exists(combined_retinotopy_directory):
                combined_retinotopy = np.load(os.path.join(combined_retinotopy_directory, "Combined_Sign_map.npy"))
            else:
                combined_retinotopy = np.zeros((self.image_height, self.image_width))

            self.regressor_map_list[mouse_index].append(combined_retinotopy)


        # Load Foot Maps
        """
        for mouse_index in range(self.number_of_sessions):
            root_directory =  self.root_directory_list[mouse_index]
            foot_map = np.load(os.path.join(root_directory, "Limb_Map", "Foot_Map.npy"))
            self.regressor_map_list[mouse_index].append(foot_map)
        """
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

        # Regressor Display View
        self.display_view_height = int(self.image_height * self.display_view_scale)
        self.display_view_width = int(self.image_width * self.display_view_scale)
        self.lick_display_view, self.lick_display_view_widget = create_display_view(image_height=self.display_view_height, image_width=self.display_view_width)
        self.running_display_view, self.running_display_view_widget = create_display_view(image_height=self.display_view_height, image_width=self.display_view_width)
        self.limbs_display_view, self.limbs_display_view_widget = create_display_view(image_height=self.display_view_height, image_width=self.display_view_width)
        self.retinotopy_display_view, self.retinotopy_display_view_widget = create_display_view(image_height=self.display_view_height, image_width=self.display_view_width)

        # Create Session Labels
        self.matching_session_label = QLabel("Matching Session: ")

        # Create Session List View
        self.matching_session_list_widget = QListWidget()
        self.matching_session_list_widget.currentRowChanged.connect(self.load_matching_session)
        self.matching_session_list_widget.setFixedWidth(200)

        # Get List Of Mice Names
        for mouse in mouse_list:
            mouse_name = Path(mouse[0]).parts[-2]
            self.session_list.append(mouse_name)
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


        # Create Transformation Control Layput
        self.transformation_control_widget = QWidget()
        self.transformation_control_layout = QGridLayout()

        control_column = 0
        self.transformation_control_layout.addWidget(self.left_button,                      2, control_column, 1, 1)
        self.transformation_control_layout.addWidget(self.right_button,                     3, control_column, 1, 1)
        self.transformation_control_layout.addWidget(self.up_button,                        4, control_column, 1, 1)
        self.transformation_control_layout.addWidget(self.down_button,                      5, control_column, 1, 1)
        self.transformation_control_layout.addWidget(self.rotate_clockwise_button,          6, control_column, 1, 1)
        self.transformation_control_layout.addWidget(self.rotate_counterclockwise_button,   7, control_column, 1, 1)
        self.transformation_control_layout.addWidget(self.x_label,                          8, control_column, 1, 1)
        self.transformation_control_layout.addWidget(self.y_label,                          9, control_column, 1, 1)
        self.transformation_control_layout.addWidget(self.angle_label,                      10, control_column, 1, 1)
        self.transformation_control_layout.addWidget(self.zoom_spinner,                     11, control_column, 1, 1)
        self.transformation_control_layout.addWidget(self.map_button,                       12, control_column, 1, 1)

        self.transformation_control_widget.setLayout(self.transformation_control_layout)


        # Create and Set Layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)


        # Add List Views
        self.layout.addWidget(self.matching_session_label,          0, 0, 1, 1)
        self.layout.addWidget(self.matching_session_list_widget,    1, 0, 2, 1)

        # Add Display Views
        self.layout.addWidget(self.lick_display_view_widget,        1, 1, 1, 1)
        self.layout.addWidget(self.running_display_view_widget,     1, 2, 1, 1)
        self.layout.addWidget(self.limbs_display_view_widget,       2, 1, 1, 1)
        self.layout.addWidget(self.retinotopy_display_view_widget,  2, 2, 1, 1)

        # Add Transformation Widget
        self.layout.addWidget(self.transformation_control_widget,   1, 3, 1, 1)

    def draw_images(self):

        # Transform Regressor Maps
        print("Regressor ma0 list liength", len(self.regressor_map_list))
        print("Individual item length", len(self.regressor_map_list[self.current_matching_index]))

        transformed_matching_lick_map       = self.transform_image(self.regressor_map_list[self.current_matching_index][0], self.variable_dictionary_list[self.current_matching_index])
        transformed_matching_running_map    = self.transform_image(self.regressor_map_list[self.current_matching_index][3], self.variable_dictionary_list[self.current_matching_index])
        transformed_matching_limb_map       = self.transform_image(self.regressor_map_list[self.current_matching_index][-1], self.variable_dictionary_list[self.current_matching_index])
        transformed_matching_retinotopy     = self.transform_image(self.regressor_map_list[self.current_matching_index][-2], self.variable_dictionary_list[self.current_matching_index])

        # Combine Sign Maps
        combined_lick_map = np.zeros((300, 304, 3))
        combined_running_map = np.zeros((300, 304, 3))
        combined_limb_map = np.zeros((300, 304, 3))
        combined_retinotopy = np.zeros((300, 304, 3))

        combined_lick_map[:, :, 0] = transformed_matching_lick_map
        combined_lick_map[:, :, 1] = self.atlas_outlines

        combined_running_map[:, :, 0] = transformed_matching_running_map
        combined_running_map[:, :, 1] = self.atlas_outlines

        combined_limb_map[:, :, 0] = transformed_matching_limb_map
        combined_limb_map[:, :, 1] = self.atlas_outlines

        positive_retinotopy = np.where(transformed_matching_retinotopy > 0, transformed_matching_retinotopy, 0)
        negative_retinotopy = np.where(transformed_matching_retinotopy < 0, transformed_matching_retinotopy, 0)
        negative_retinotopy = np.abs(negative_retinotopy)

        combined_retinotopy[:, :, 0] = positive_retinotopy
        combined_retinotopy[:,:, 1] = self.atlas_outlines
        combined_retinotopy[:,:, 2] = negative_retinotopy

        # Set Images
        self.lick_display_view.setImage(combined_lick_map)
        self.running_display_view.setImage(combined_running_map)
        self.limbs_display_view.setImage(combined_limb_map)
        self.retinotopy_display_view.setImage(combined_retinotopy)

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
            root_directory = self.root_directory_list[session_index]
            save_directory = os.path.join(root_directory, "Across_Mouse_Alignment_Dictionary.npy")

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


def load_mouse_regressors(session_list, visualise=False):

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
        #session_limbs_map   = load_cpd_map(regression_dictionary, 'Limbs',          indicies, image_height, image_width, within_mouse_alignment_dictionary)

        # Add To List
        mouse_lick_list.append(session_lick_map)
        mouse_running_list.append(session_running_map)
        mouse_face_list.append(session_face_map)
        mouse_whisk_list.append(session_whisk_map)
        #mouse_limbs_list.append(session_limbs_map)

    # Get Mean Maps
    mouse_lick_map = np.mean(mouse_lick_list, axis=0)
    mouse_running_map = np.mean(mouse_running_list, axis=0)
    mouse_face_map = np.mean(mouse_face_list, axis=0)
    mouse_whisk_map = np.mean(mouse_whisk_list, axis=0)
    #mouse_limbs_map = np.mean(mouse_limbs_list, axis=0)

    # Get edges
    """
    mouse_lick_map = get_map_edges(mouse_lick_map)
    mouse_running_map = get_map_edges(mouse_running_map)
    mouse_face_map = get_map_edges(mouse_face_map)
    mouse_whisk_map = get_map_edges(mouse_whisk_map)
    mouse_limbs_map = get_map_edges(mouse_limbs_map)
    """

    # Normalise
    percentile_threshold = 99
    mouse_lick_map = np.divide(mouse_lick_map, np.percentile(mouse_lick_map, q=percentile_threshold))
    mouse_running_map = np.divide(mouse_running_map, np.percentile(mouse_running_map, q=percentile_threshold))
    mouse_face_map = np.divide(mouse_face_map, np.percentile(mouse_face_map, q=percentile_threshold))
    mouse_whisk_map = np.divide(mouse_whisk_map, np.percentile(mouse_whisk_map, q=percentile_threshold))
    #mouse_limbs_map = np.divide(mouse_limbs_map, np.percentile(mouse_limbs_map, q=percentile_threshold))

    mouse_lick_map = np.clip(mouse_lick_map,  a_min=0, a_max=1)
    mouse_running_map = np.clip(mouse_running_map,  a_min=0, a_max=1)
    mouse_face_map = np.clip(mouse_face_map,  a_min=0, a_max=1)
    mouse_whisk_map = np.clip(mouse_whisk_map,  a_min=0, a_max=1)
    #mouse_limbs_map = np.clip(mouse_limbs_map, a_min=0, a_max=1)

    # view These
    if visualise == True:
        figure_1 = plt.figure()
        rows = 1
        columns = 5
    
        lick_axis = figure_1.add_subplot(rows, columns, 1)
        running_axis = figure_1.add_subplot(rows, columns, 2)
        face_axis = figure_1.add_subplot(rows, columns, 3)
        whisk_axis = figure_1.add_subplot(rows, columns, 4)
        #limbs_axis = figure_1.add_subplot(rows, columns, 5)
    
        lick_axis.imshow(mouse_lick_map)
        running_axis.imshow(mouse_running_map)
        face_axis.imshow(mouse_face_map)
        whisk_axis.imshow(mouse_whisk_map)
        #limbs_axis.imshow(mouse_limbs_map)
    
        plt.show()
        
    regressor_map_list = [mouse_lick_map, mouse_running_map, mouse_face_map, mouse_whisk_map] #, mouse_limbs_map]
    
    return regressor_map_list


def load_atlas():

    # Load Atlas Boundaires
    allen_atlas_boundaries_file = pkg_resources.resource_stream('Files', 'Atlas_Outlines.npy')
    allen_atlas_boundaries = np.load(allen_atlas_boundaries_file)

    # Load Atlas ALignment Dict
    atlas_alignment_dictionary_file = pkg_resources.resource_stream('Files', 'Atlas_Alignment_Dictionary.npy')
    atlas_alignment_dictionary = np.load(atlas_alignment_dictionary_file, allow_pickle=True)[()]

    # Align Atlas
    allen_atlas_boundaries = Registration_Utils.transform_mask_or_atlas(allen_atlas_boundaries, atlas_alignment_dictionary)

    return allen_atlas_boundaries


def align_sessions(session_list):
    app = QApplication(sys.argv)

    window = masking_window(session_list)
    window.show()

    app.exec_()




mouse_list = [

    [r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Switching_Opto/KPGC2.2G/2022_12_02_Switching_Opto"],

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
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_08_Transition_Imaging"],

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_17_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_19_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_22_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_24_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_26_Transition_Imaging"],

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_14_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_20_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_26_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_05_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_08_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_10_Transition_Imaging"],

    [r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_21_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_23_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_11_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_13_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_15_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_17_Transition_Imaging"],

    [r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_14_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_20_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_22_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_29_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_03_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_05_Transition_Imaging"],

    [r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_05_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_09_Switching_Imaging"],

    [r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_29_Switching_Imaging",
     r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_05_Switching_Imaging"],

    [r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_02_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_04_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_06_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_02_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_08_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_10_Transition_Imaging"],

    [r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_26_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_28_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_02_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_23_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_31_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_04_02_Transition_Imaging"],

]

#r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_26_Continous_Retinotopy_Left",
#r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_01_Continuous_Retinotopic_Mapping_Left",
#r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_01_Continous_Retinotopy_Left",
#r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_01_Continous_Retinotopy_Left",
#r"/media/matthew/Expansion/Control_Data/NXAK14.1A/Continous_Retinotopic_Mapping_Left"



#load_mouse_regressors(mouse_list[3])
load_atlas()
align_sessions(mouse_list)