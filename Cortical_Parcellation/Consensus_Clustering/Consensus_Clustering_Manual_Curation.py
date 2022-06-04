import numpy as np
import matplotlib.pyplot as plt
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

        # Create Session Labels
        self.cluster_1_label = QLabel("Cluster 1")
        self.cluster_2_label = QLabel("Cluster 2")

        # Create Session List Views
        self.cluster_1_list_widget = QListWidget()
        self.cluster_2_list_widget = QListWidget()

        self.cluster_1_list_widget.setCurrentRow(0)
        self.cluster_2_list_widget.setCurrentRow(0)

        self.cluster_1_list_widget.currentRowChanged.connect(self.update_selected_clusters)
        self.cluster_2_list_widget.currentRowChanged.connect(self.update_selected_clusters)
        self.update_list_widgets()

        # Create Merge Button
        self.merge_button = QPushButton("Merge Clusters")
        self.merge_button.clicked.connect(self.merge_clusters)

        # reate Save Button
        self.save_button = QPushButton("Save Clusters")
        self.save_button.clicked.connect(self.save_clusters)

        # Create and Set Layout]
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Add Display Views
        self.layout.addWidget(self.edges_display_view_widget, 0, 0, 1, 1)

        # Add Labels
        self.layout.addWidget(self.cluster_1_label, 0, 1, 1, 1)
        self.layout.addWidget(self.cluster_2_label, 0, 2, 1, 1)

        # Add List Views
        self.layout.addWidget(self.cluster_1_list_widget, 0, 1, 1, 2)
        self.layout.addWidget(self.cluster_2_list_widget, 0, 2, 1, 2)

        # Add Merge Button
        self.layout.addWidget(self.merge_button,        1, 0, 1, 1)
        self.layout.addWidget(self.save_button,        1, 1, 1, 1)

    def update_list_widgets(self):

        self.cluster_1_list_widget.clear()
        self.cluster_2_list_widget.clear()

        unique_clusters = list(np.unique(self.pixel_assignments))
        for cluster in unique_clusters:
            self.cluster_1_list_widget.addItem(str(cluster))
            self.cluster_2_list_widget.addItem(str(cluster))


    def update_selected_clusters(self):

        if self.cluster_1_list_widget.currentItem() != None and self.cluster_2_list_widget.currentItem() != None:
            cluster_1 = int(self.cluster_1_list_widget.currentItem().text())
            cluster_2 = int(self.cluster_2_list_widget.currentItem().text())

            # Get Cluster Masks
            cluster_1_mask = np.where(self.pixel_assignments == cluster_1, 1, 0)
            cluster_2_mask = np.where(self.pixel_assignments == cluster_2, 1, 0)

            # Get Cluster Edges
            edges = feature.canny(np.ndarray.astype(self.pixel_assignments, "float32"))

            # Get Selected Pixels
            edge_pixels = np.nonzero(edges)
            cluster_1_pixels = np.nonzero(cluster_1_mask)
            cluster_2_pixels = np.nonzero(cluster_2_mask)

            joint_image = np.zeros((600, 608, 3))
            joint_image[cluster_1_pixels] = [1, 0, 0]
            joint_image[cluster_2_pixels] = [0, 1, 0]
            joint_image[edge_pixels] = [1,1,1]
            self.edges_display_view.setImage(joint_image)


    def merge_clusters(self):

        cluster_1 = int(self.cluster_1_list_widget.currentItem().text())
        cluster_2 = int(self.cluster_2_list_widget.currentItem().text())

        self.pixel_assignments = np.where(self.pixel_assignments == cluster_2, cluster_1, self.pixel_assignments)

        self.update_list_widgets()

    def save_clusters(self):
        np.save(r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Mirrored_Curated_Clusters.npy", self.pixel_assignments)



app = QApplication(sys.argv)

# Load Consensus Clusters
pixel_assignments = np.load(r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/regrown_pixel_assignments.npy")
pixel_assignments = np.load(r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Curated_Clusters.npy")
pixel_assignments = np.load(r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Mirrored_Clusters.npy")
pixel_assignments = np.load(r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Mirrored_Curated_Clusters.npy")
pixel_assignments = np.ndarray.astype(pixel_assignments, int)


window = curation_window(pixel_assignments)
window.show()

app.exec_()