import numpy as np
import matplotlib.pyplot as plt
import h5py
import tables
from scipy import signal, ndimage, stats
from sklearn.linear_model import LinearRegression
from skimage.morphology import white_tophat
from sklearn.preprocessing import StandardScaler
from skimage.transform import rescale
from skimage.feature import canny
from PIL import Image
import os
import cv2
from datetime import datetime
import pyqtgraph

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import sys


pyqtgraph.setConfigOptions(imageAxisOrder='row-major')




class masking_window(QWidget):

    def __init__(self, parent=None):
        super(masking_window, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("Apply Mask")
        self.setGeometry(0, 0, 1500, 500)

        # Create Variable Holders
        self.variable_dictionary = self.create_variable_dictionary()

        pixel_assigments = np.load("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Curated_Clusters.npy")
        edges = canny(pixel_assigments.astype('float32'))
        self.edges = edges

        # Load Mask
        self.mask = np.load("/home/matthew/Documents/Allen_Atlas_Templates/Mask_Array.npy")
        self.mask_outline = np.load("/home/matthew/Documents/Allen_Atlas_Templates/Outline_array.npy")
        self.variable_dictionary['mask'] = self.mask
        self.variable_dictionary['mask_outline'] = self.mask_outline

        # Load Transformation Dictionary
        example_transformation_dictionary = np.load("/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_01_Discrimination_Imaging/Cluster_Alignment_Dictionary.npy", allow_pickle=True)[()]

        # Load Max Projection
        example_max_projection = np.load("/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_01_Discrimination_Imaging/max_projection.npy")

        # Transform Max Projection
        example_max_projection = transform_max_projection(example_max_projection, example_transformation_dictionary)
        self.max_projection = example_max_projection
        self.variable_dictionary['max_projection'] = self.max_projection


        # Load Parcellation Edges

        pixel_assigments = np.load("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Curated_Clusters.npy")
        edges = canny(pixel_assigments.astype('float32'))

        # Create Masked Figures
        self.masked_display_view_widget = QWidget()
        self.masked_display_view_widget_layout = QGridLayout()
        self.masked_display_view = pyqtgraph.ImageView()
        self.masked_display_view.ui.histogram.hide()
        self.masked_display_view.ui.roiBtn.hide()
        self.masked_display_view.ui.menuBtn.hide()
        self.masked_display_view_widget_layout.addWidget(self.masked_display_view, 0, 0)
        self.masked_display_view_widget.setLayout(self.masked_display_view_widget_layout)
        self.masked_display_view_widget.setMinimumWidth(1000)

        # Create Max_Projection_Display_View
        self.max_projection_display_view_widget = QWidget()
        self.max_projection_display_view_widget_layout = QGridLayout()
        self.max_projection_display_view = pyqtgraph.ImageView()
        self.max_projection_display_view.ui.histogram.hide()
        self.max_projection_display_view.ui.roiBtn.hide()
        self.max_projection_display_view.ui.menuBtn.hide()
        self.max_projection_display_view_widget_layout.addWidget(self.max_projection_display_view, 0, 0)
        self.max_projection_display_view_widget.setLayout(self.max_projection_display_view_widget_layout)
        self.max_projection_display_view_widget.setMinimumWidth(1000)


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

        self.shrink_button = QPushButton("Shrink")
        self.shrink_button.clicked.connect(self.shrink)

        self.enlarge_button = QPushButton("Enlarge")
        self.enlarge_button.clicked.connect(self.enlarge)

        self.map_button = QPushButton("Set Alignment")
        self.map_button.clicked.connect(self.set_alignment)

        # Add Labels
        self.x_label = QLabel()
        self.y_label = QLabel()
        self.height_label = QLabel()
        self.width_label = QLabel()
        self.angle_label = QLabel()
        self.scale_label = QLabel()


        # Create Layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Add Transformation Widgets
        #self.layout.addWidget(self.select_session_button,           1,  0,  1,  1)
        self.layout.addWidget(self.left_button,                     2,  0,  1,  1)
        self.layout.addWidget(self.right_button,                    3,  0,  1,  1)
        self.layout.addWidget(self.up_button,                       4,  0,  1,  1)
        self.layout.addWidget(self.down_button,                     5,  0,  1,  1)
        self.layout.addWidget(self.rotate_clockwise_button,         6,  0,  1,  1)
        self.layout.addWidget(self.rotate_counterclockwise_button,  7,  0,  1,  1)
        self.layout.addWidget(self.enlarge_button,                  8,  0,  1,  1)
        self.layout.addWidget(self.shrink_button,                   9,  0,  1,  1)
        self.layout.addWidget(self.x_label,                         10, 0,  1,  1)
        self.layout.addWidget(self.y_label,                         11, 0,  1,  1)
        self.layout.addWidget(self.height_label,                    12, 0,  1,  1)
        self.layout.addWidget(self.width_label,                     13, 0,  1,  1)
        self.layout.addWidget(self.angle_label,                     14, 0,  1,  1)
        self.layout.addWidget(self.scale_label,                     15, 0,  1,  1)
        self.layout.addWidget(self.map_button,                      16, 0,  1,  1)

        # Add Display Widgets
        self.layout.addWidget(self.masked_display_view_widget,      0, 1, 17, 17)
        self.layout.addWidget(self.max_projection_display_view_widget, 0, 18, 17, 17)
        self.show()


    def create_variable_dictionary(self):

        # Transformation Attributes
        x_shift = 0
        y_shift = 0
        rotation = 0
        scale = 0.83

        # Array Details
        background_size = 1000
        bounding_size = 400
        background_array = np.zeros((background_size, background_size, 3))
        bounding_array = np.zeros((bounding_size, bounding_size))

        # Template Details
        template_x_start = 100
        template_y_start = 100
        template_width = 608
        template_height = 600

        # Create Dictionary
        variable_dictionary = {

            # Affine Atributes
            'x_shift': x_shift,
            'y_shift': y_shift,
            'rotation': rotation,
            'scale':scale,

            # Template Details
            'template_x_start':template_x_start,
            'template_y_start':template_x_start,
            'template_width':template_width,
            'template_height':template_height,

            # Arrays
            'background_array': background_array,
            'bounding_array': bounding_array,
            'max_projection': None,
            'mask': None
        }

        return variable_dictionary


    def move_left(self):
        self.variable_dictionary['x_shift'] = self.variable_dictionary['x_shift'] + 2
        self.x_label.setText("x: " + str(self.variable_dictionary['x_shift']))
        self.draw_images()

    def move_right(self):
        self.variable_dictionary['x_shift'] = self.variable_dictionary['x_shift'] - 2
        self.x_label.setText("x: " + str(self.variable_dictionary['x_shift']))
        self.draw_images()

    def move_up(self):
        self.variable_dictionary['y_shift'] = self.variable_dictionary['y_shift'] - 2
        self.y_label.setText("y: " + str(self.variable_dictionary['y_shift']))
        self.draw_images()

    def move_down(self):
        self.variable_dictionary['y_shift'] = self.variable_dictionary['y_shift'] + 2
        self.y_label.setText("y: " + str(self.variable_dictionary['y_shift']))
        self.draw_images()

    def rotate_clockwise(self):
        self.variable_dictionary['rotation'] = self.variable_dictionary['rotation'] - 1
        self.angle_label.setText("Angle: " + str(self.variable_dictionary['rotation']))
        self.draw_images()

    def rotate_counterclockwise(self):
        self.variable_dictionary['rotation'] = self.variable_dictionary['rotation'] + 1
        self.angle_label.setText("Angle: " + str(self.variable_dictionary['rotation']))
        self.draw_images()

    def shrink(self):
        self.variable_dictionary['scale'] = np.around(self.variable_dictionary['scale'] - 0.01, 2)
        self.scale_label.setText("Scale: " + str(self.variable_dictionary['scale']))
        self.draw_images()

    def enlarge(self):
        self.variable_dictionary['scale'] = np.around(self.variable_dictionary['scale'] + 0.01, 2)
        self.scale_label.setText("Scale: " + str(self.variable_dictionary['scale']))
        self.draw_images()


    def transform_array(self, template_image, matching_image, variable_dictionary, normalise=False):

        # Load Data
        template_x_start = variable_dictionary['template_x_start']
        template_y_start = variable_dictionary['template_y_start']
        template_width   = variable_dictionary['template_width']
        template_height  = variable_dictionary['template_height']
        x_shift          = variable_dictionary['x_shift']
        y_shift          = variable_dictionary['y_shift']
        background_array = np.copy(variable_dictionary["background_array"])

        # Scale
        scale_factor = variable_dictionary['scale']
        matching_image = rescale(matching_image, scale=scale_factor, preserve_range=True)


        # Rotate
        angle = variable_dictionary['rotation']
        matching_image = ndimage.rotate(matching_image, angle, reshape=False)

        # Translate
        #matching_image = np.roll(a=matching_image, axis=0, shift=y_shift)
        #matching_image = np.roll(a=matching_image, axis=1, shift=x_shift)

        # Scale Images
        if normalise == True:
            template_image = np.divide(template_image.astype(float), np.percentile(template_image, 95))
            matching_image = np.divide(matching_image.astype(float), np.percentile(matching_image, 95))

        template_image = np.nan_to_num(template_image)
        matching_image = np.nan_to_num(matching_image)

        template_image = np.clip(template_image, a_min=0, a_max=1)
        matching_image = np.clip(matching_image, a_min=0, a_max=1)

        # Insert Images Into Background Array
        image_height = np.shape(matching_image)[0]
        image_width = np.shape(matching_image)[1]

        background_array[template_y_start:template_y_start + template_height, template_x_start:template_x_start + template_width, 2] = template_image
        background_array[template_y_start:template_y_start + template_height, template_x_start:template_x_start + template_width, 1] += 0.5 * template_image

        mask_y_start = template_y_start + y_shift
        mask_x_start = template_x_start + x_shift

        #background_array[template_y_start:template_y_start + image_height, template_x_start:template_x_start + image_width, 0] = matching_image
        background_array[mask_y_start:mask_y_start + image_height, mask_x_start:mask_x_start + image_width, 1] += 0.3 * matching_image

        return background_array



    def draw_images(self):

        # Transform Masked Matching Image
        transformed_mask = np.copy(self.mask_outline)

        background_array = self.transform_array(self.edges, transformed_mask, self.variable_dictionary)
        background_array_2 = self.transform_array(self.max_projection, transformed_mask, self.variable_dictionary, normalise=True)

        self.masked_display_view.setImage(background_array)
        self.max_projection_display_view.setImage(background_array_2)


    def set_alignment(self):

        matching_image = self.mask

        # Load Data
        template_x_start = self.variable_dictionary['template_x_start']
        template_y_start = self.variable_dictionary['template_y_start']
        template_width = self.variable_dictionary['template_width']
        template_height = self.variable_dictionary['template_height']
        x_shift = self.variable_dictionary['x_shift']
        y_shift = self.variable_dictionary['y_shift']
        background_array = np.copy(self.variable_dictionary["background_array"])

        # Scale
        scale_factor = self.variable_dictionary['scale']
        matching_image = rescale(matching_image, scale=scale_factor, preserve_range=True)

        # Rotate
        angle = self.variable_dictionary['rotation']
        matching_image = ndimage.rotate(matching_image, angle, reshape=False)

        # Binarise
        matching_image = np.nan_to_num(matching_image)
        matching_image = np.where(matching_image > 0.9, 1, 0)

        # Insert Images Into Background Array
        image_height = np.shape(matching_image)[0]
        image_width = np.shape(matching_image)[1]
        mask_y_start = template_y_start + y_shift
        mask_x_start = template_x_start + x_shift

        # background_array[template_y_start:template_y_start + image_height, template_x_start:template_x_start + image_width, 0] = matching_image
        background_array[mask_y_start:mask_y_start + image_height, mask_x_start:mask_x_start + image_width, 1] += matching_image

        # Final Mask
        final_mask = background_array[template_y_start:template_y_start + template_height, template_x_start:template_x_start + template_width, 1]

        np.save(r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/mask.npy", final_mask)
        plt.imshow(final_mask)
        plt.show()



def transform_max_projection(image, variable_dictionary):

    # Rotate
    angle = variable_dictionary['rotation']
    x_shift = variable_dictionary['x_shift']
    y_shift = variable_dictionary['y_shift']

    transformed_image = np.copy(image)
    transformed_image = ndimage.rotate(transformed_image, angle, reshape=False, prefilter=True)
    transformed_image = np.roll(a=transformed_image, axis=0, shift=y_shift)
    transformed_image = np.roll(a=transformed_image, axis=1, shift=x_shift)

    return transformed_image

def position_mask():

    app = QApplication(sys.argv)

    window = masking_window()
    window.show()

    app.exec_()

position_mask()