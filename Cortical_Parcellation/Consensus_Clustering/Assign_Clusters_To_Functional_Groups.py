import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage import feature
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph

import os
import sys



pyqtgraph.setConfigOptions(imageAxisOrder='row-major')


class curation_window(QWidget):

    def __init__(self, pixel_assignments, parent=None):
        super(curation_window, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("Consensus Cluster Curation")
        self.setGeometry(0, 0, 1900, 500)

        # Setup Internal Variables
        self.pixel_assignments = pixel_assignments
        self.unique_clusters = list(np.unique(self.pixel_assignments))
        self.number_of_clusters = len(self.unique_clusters)
        self.cluster_group_assignments = list(np.zeros(self.number_of_clusters))
        self.number_of_groups = 10
        self.group_colours = []
        colourmap = cm.get_cmap('tab20')
        for group_index in range(self.number_of_groups):
            group_value = float(group_index) / self.number_of_groups
            colour = colourmap(group_value)
            self.group_colours.append(colour)


        # Edges Display View
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

        # Create Session List Views
        self.cluster_list_widget = QListWidget()
        self.cluster_list_widget.setCurrentRow(0)
        self.cluster_list_widget.currentRowChanged.connect(self.draw_group_colours)
        for cluster in self.unique_clusters:
            self.cluster_list_widget.addItem(str(cluster))

        # Create Group Combo Box
        self.group_combo_box = QComboBox()
        for group in range(self.number_of_groups):
            self.group_combo_box.addItem(str(group))
        #self.group_combo_box.currentIndexChanged.connect(self.set_cluster_group)

        # Create Save Button
        self.save_button = QPushButton("Save Clusters")
        self.save_button.clicked.connect(self.save_groups)

        # Create Set Current Button
        self.set_current_group_button = QPushButton("Set Current Group")
        self.set_current_group_button.clicked.connect(self.set_cluster_group)

        # Create and Set Layout]
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Add Display Views
        self.layout.addWidget(self.edges_display_view_widget, 0, 0, 3, 1)

        # Add List Views
        self.layout.addWidget(self.cluster_list_widget, 0, 1, 3, 1)

        # Add Combo Box
        self.layout.addWidget(self.group_combo_box, 0, 2, 3, 1)

        # Add Buttons
        self.layout.addWidget(self.set_current_group_button, 1, 3, 1, 1)
        self.layout.addWidget(self.save_button,              2, 3, 1, 1)


    def set_cluster_group(self):

        current_cluster = int(self.cluster_list_widget.currentRow())
        group_assigment = self.group_combo_box.currentIndex()
        self.cluster_group_assignments[current_cluster] = group_assigment
        self.draw_group_colours()


    def draw_group_colours(self):

        joint_image = np.zeros((600, 608, 4))

        for cluster_index in range(self.number_of_clusters):
            cluster = self.unique_clusters[cluster_index]
            cluster_mask = np.where(self.pixel_assignments == cluster, 1, 0)
            cluster_pixels = np.nonzero(cluster_mask)
            cluster_group = int(self.cluster_group_assignments[cluster_index])
            cluster_colour = self.group_colours[cluster_group]
            joint_image[cluster_pixels] = cluster_colour

        edges_pixels = feature.canny(np.ndarray.astype(self.pixel_assignments, "float32"))
        joint_image[edges_pixels] = [1, 1, 1, 1]

        # Highlight Current Cluster
        current_cluster = int(self.cluster_list_widget.currentRow())
        current_cluster_mask = np.where(self.pixel_assignments == current_cluster, 1, 0)
        current_cluster_edges = feature.canny(np.ndarray.astype(current_cluster_mask, "float32"))
        current_cluster_edge_pixels = np.nonzero(current_cluster_edges)
        joint_image[current_cluster_edge_pixels] = [0, 0, 0, 1]

        self.edges_display_view.setImage(joint_image)



    def update_selected_clusters(self):
        self.draw_group_colours()


    def save_groups(self):
        np.save(r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Cluster_Group_Assignments.npy", self.cluster_group_assignments)



app = QApplication(sys.argv)

# Load Consensus Clusters
pixel_assignments = np.load(r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Final_Consensus_Clusters.npy")
pixel_assignments = np.ndarray.astype(pixel_assignments, int)


window = curation_window(pixel_assignments)
window.show()

app.exec_()