import numpy as np
from skimage.feature import canny
import matplotlib.pyplot as plt
from skimage import measure
from scipy import ndimage

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph

import os
import sys

pyqtgraph.setConfigOptions(imageAxisOrder='row-major')


class masking_window(QWidget):

    def __init__(self, smoothed_contours, pixel_assignments, parent=None):
        super(masking_window, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("Align To Static Cross")
        self.setGeometry(0, 0, 1900, 500)

        # Setup Internal Variables
        self.smoothed_contours = smoothed_contours
        self.pixel_assignments = pixel_assignments

        # Original Display View
        self.original_display_view_widget = QWidget()
        self.original_display_view_widget_layout = QGridLayout()
        self.original_display_view = pyqtgraph.ImageView()
        self.original_display_view.ui.histogram.hide()
        self.original_display_view.ui.roiBtn.hide()
        self.original_display_view.ui.menuBtn.hide()
        self.original_display_view_widget_layout.addWidget(self.original_display_view, 0, 0)
        self.original_display_view_widget.setLayout(self.original_display_view_widget_layout)
        self.original_display_view_widget.setFixedWidth(608)
        self.original_display_view_widget.setFixedHeight(600)
        self.original_display_view.setLevels(0, 1)

        # Regressor Display View
        self.left_reflected_display_view_widget = QWidget()
        self.left_reflected_display_view_widget_layout = QGridLayout()
        self.left_reflected_display_view = pyqtgraph.ImageView()
        self.left_reflected_display_view.ui.histogram.hide()
        self.left_reflected_display_view.ui.roiBtn.hide()
        self.left_reflected_display_view.ui.menuBtn.hide()
        self.left_reflected_display_view_widget_layout.addWidget(self.left_reflected_display_view, 0, 0)
        self.left_reflected_display_view_widget.setLayout(self.left_reflected_display_view_widget_layout)
        self.left_reflected_display_view_widget.setFixedWidth(608)
        self.left_reflected_display_view_widget.setFixedHeight(600)
        
        # Regressor Display View
        self.right_reflected_display_view_widget = QWidget()
        self.right_reflected_display_view_widget_layout = QGridLayout()
        self.right_reflected_display_view = pyqtgraph.ImageView()
        self.right_reflected_display_view.ui.histogram.hide()
        self.right_reflected_display_view.ui.roiBtn.hide()
        self.right_reflected_display_view.ui.menuBtn.hide()
        self.right_reflected_display_view_widget_layout.addWidget(self.right_reflected_display_view, 0, 0)
        self.right_reflected_display_view_widget.setLayout(self.right_reflected_display_view_widget_layout)
        self.right_reflected_display_view_widget.setFixedWidth(608)
        self.right_reflected_display_view_widget.setFixedHeight(600)

        self.variable_dictionary = {'rotation':0,
        'x_shift':0,
        'y_shift':0
        }

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

        # Add Display Views
        self.layout.addWidget(self.original_display_view_widget,        1, 1, 25, 1)
        self.layout.addWidget(self.left_reflected_display_view_widget,  1, 2, 25, 1)
        self.layout.addWidget(self.right_reflected_display_view_widget, 1, 3, 25, 1)

        # Add Transformation Controls
        self.layout.addWidget(self.left_button, 2, 6, 1, 1)
        self.layout.addWidget(self.right_button, 3, 6, 1, 1)
        self.layout.addWidget(self.up_button, 4, 6, 1, 1)
        self.layout.addWidget(self.down_button, 5, 6, 1, 1)
        self.layout.addWidget(self.rotate_clockwise_button, 6, 6, 1, 1)
        self.layout.addWidget(self.rotate_counterclockwise_button, 7, 6, 1, 1)
        self.layout.addWidget(self.x_label, 8, 6, 1, 1)
        self.layout.addWidget(self.y_label, 9, 6, 1, 1)
        self.layout.addWidget(self.angle_label, 10, 6, 1, 1)
        self.layout.addWidget(self.map_button, 11, 6, 1, 1)



    def draw_images(self):

        transformed_contours = self.transform_image(self.smoothed_contours, self.variable_dictionary)

        # Reflected_contours
        left_reflected_contours = np.copy(transformed_contours)
        right_reflected_contours = np.copy(transformed_contours)

        left_hand_side = transformed_contours[:, 0:304]
        right_hand_side = transformed_contours[:, 304:]

        left_hand_side = np.flip(left_hand_side, axis=1)
        right_hand_side = np.flip(right_hand_side, axis=1)

        left_reflected_contours[:, 304:] = left_hand_side
        right_reflected_contours[:, 0:304] = right_hand_side

        self.original_display_view.setImage(transformed_contours)
        self.left_reflected_display_view.setImage(left_reflected_contours)
        self.right_reflected_display_view.setImage(right_reflected_contours)

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
        self.variable_dictionary['x_shift'] = self.variable_dictionary['x_shift'] + 1
        self.x_label.setText("x: " + str(self.variable_dictionary['x_shift']))
        self.draw_images()

    def move_right(self):
        self.variable_dictionary['x_shift'] = self.variable_dictionary['x_shift'] - 1
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
        self.variable_dictionary['rotation'] = self.variable_dictionary['rotation'] - 1
        self.angle_label.setText("Angle: " + str(self.variable_dictionary['rotation']))
        self.draw_images()

    def rotate_counterclockwise(self):
        self.variable_dictionary['rotation'] = self.variable_dictionary['rotation'] + 1
        self.angle_label.setText("Angle: " + str(self.variable_dictionary['rotation']))
        self.draw_images()

    def set_alignment(self):

        # Transform_Clusters
        transformed_clusters = transform_clusters(self.pixel_assignments, self.variable_dictionary)

        # Reflect Clusters
        mirrored_clusters = np.copy(transformed_clusters)
        mirrored_clusters[:, 0:304] = -1 * np.flip(mirrored_clusters[:, 304:], axis=1)

        # Transform Back
        reverse_variable_dictionary = {
            'rotation':self.variable_dictionary['rotation'] * -1,
            'x_shift':self.variable_dictionary['x_shift'] * -1,
            'y_shift':self.variable_dictionary['y_shift'] * -1,
        }

        mirrored_clusters = transform_clusters(mirrored_clusters, reverse_variable_dictionary)
        plt.imshow(mirrored_clusters)
        plt.show()

        # Save Dictionary
        np.save("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Mirrored_Clusters.npy", mirrored_clusters)


    def load_matching_session(self):

        self.current_matching_index = self.matching_session_list_widget.currentRow()

        self.max_projection = self.max_projection_list[self.current_matching_index]
        self.matching_cluster_edges = self.cluster_edges_list[self.current_matching_index]
        self.draw_images()
        # self.cross_display_view.setImage(max_projection)

    def load_template_session(self):

        self.current_template_index = self.template_session_list_widget.currentRow()
        self.template_cluster_edges = self.cluster_edges_list[self.current_template_index]
        self.draw_images()


def transform_clusters(clusters, variable_dictionary):

    # Unpack Dict
    angle = variable_dictionary['rotation']
    x_shift = variable_dictionary['x_shift']
    y_shift = variable_dictionary['y_shift']

    transformed_clusters = np.zeros(np.shape(clusters))

    unique_clusters = list(np.unique(clusters))
    for cluster in unique_clusters:
        cluster_mask = np.where(clusters == cluster, 1, 0)
        cluster_mask = ndimage.rotate(cluster_mask, angle, reshape=False, prefilter=True)
        cluster_mask = np.roll(a=cluster_mask, axis=0, shift=y_shift)
        cluster_mask = np.roll(a=cluster_mask, axis=1, shift=x_shift)
        cluster_indicies = np.nonzero(cluster_mask)
        transformed_clusters[cluster_indicies] = cluster

    return transformed_clusters


def smooth_contours():

    mask = np.load(r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/mask.npy")

    pixel_assignments = np.load("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Curated_Clusters.npy")
    pixel_assignments = np.ndarray.astype(pixel_assignments, "float32")
    pixel_assignments = np.multiply(pixel_assignments, mask)
    plt.imshow(pixel_assignments)
    plt.show()

    unique_clusters = list(np.unique(pixel_assignments))
    smoothed_template = np.zeros(np.shape(pixel_assignments))

    # Load Mask
    mask = np.load(r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/mask.npy")
    mask_edge = canny(mask.astype('float32'), sigma=2)
    mask_edge_indexes = np.nonzero(mask_edge)
    smoothed_template[mask_edge_indexes] = 1

    for cluster in unique_clusters:

        cluster_mask = np.where(pixel_assignments == cluster, 1, 0)

        edges = canny(cluster_mask.astype('float32'), sigma=5)
        edge_indexes = np.nonzero(edges)
        smoothed_template[edge_indexes] = 1

    smoothed_template = np.multiply(smoothed_template, mask)

    return smoothed_template, pixel_assignments



def align_sessions():

    app = QApplication(sys.argv)

    smoothed_contours, pixel_assignments = smooth_contours()

    plt.imshow(smoothed_contours)
    plt.show()

    window = masking_window(smoothed_contours, pixel_assignments)
    window.show()

    app.exec_()


align_sessions()