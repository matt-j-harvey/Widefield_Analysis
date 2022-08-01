from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph
import sys
import numpy as np
import matplotlib.pyplot as plt

pyqtgraph.setConfigOptions(imageAxisOrder='row-major')


class connectivity_matrix_explorer(QWidget):

    def __init__(self, connectivity_matrix, pixel_assignments, parent=None):
        super(connectivity_matrix_explorer, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("Adjacency Network Explorer")
        self.setGeometry(0, 0, 1000, 500)

        # Create Variable Holders
        self.connectivity_matrix = connectivity_matrix
        self.pixel_assignments = pixel_assignments
        self.image_height, self.image_width = np.shape(pixel_assignments)
        self.number_of_clusters = len(np.unique(pixel_assignments))

        # Create Brain Display View
        self.display_view_widget = QWidget()
        self.display_view_widget_layout = QGridLayout()
        self.display_view = pyqtgraph.ImageView()
        self.display_view.ui.histogram.hide()
        self.display_view.ui.roiBtn.hide()
        self.display_view.ui.menuBtn.hide()
        self.display_view_widget_layout.addWidget(self.display_view, 0, 0)
        self.display_view_widget.setLayout(self.display_view_widget_layout)
        self.display_view.getView().scene().sigMouseMoved.connect(lambda pos: self.getPixel(pos, self.display_view))
        self.display_view.setImage(self.pixel_assignments)
        self.display_view.setLevels(0, 3)
        colourmap = pyqtgraph.colormap.get('CET-L3')
        self.display_view.setColorMap(colourmap)

        # Create Layout
        self.layout = QGridLayout()
        self.layout.addWidget(self.display_view_widget)
        self.setLayout(self.layout)


    def view_modulation_map(self, selected_region):

        # Get Region Index
        region_index = selected_region - 1

        # Get Current Cluster Connectivity Vector
        cluster_connectivity_vector = self.connectivity_matrix[region_index]

        modulation_map = np.zeros(np.shape(self.pixel_assignments))

        for cluster_index in range(1, self.number_of_clusters):
            cluster_value = cluster_connectivity_vector[cluster_index - 1]
            cluster_pixels = np.where(self.pixel_assignments == cluster_index)
            modulation_map[cluster_pixels] = cluster_value


        self.display_view.setImage(modulation_map)


    def getPixel(self, pos, imageview):
        pos = imageview.getImageItem().mapFromScene(pos)

        y = np.clip(int(pos.y()), a_min=0, a_max=self.image_height-1)
        x = np.clip(int(pos.x()), a_min=0, a_max=self.image_width-1)

        print(y, x)


        selected_cluster = self.pixel_assignments[y, x]
        if selected_cluster != 0:
            print("Selected Cluster", selected_cluster)
            self.view_modulation_map(int(selected_cluster))
            #print(imageview.getImageItem().mapFromScene(pos))






if __name__ == '__main__':

    app = QApplication(sys.argv)

    # Load Connectivity Matrix
    connectivity_matrix = np.load(r"/media/matthew/Expansion/Widefield_Analysis/Switching_Analysis/Decoding/Decoder_weights.npy")

    # Load Consensus Clusters
    conensus_clusters = np.load(r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Final_Consensus_Clusters.npy")

    plt.imshow(conensus_clusters)
    plt.show()

    window = connectivity_matrix_explorer(connectivity_matrix, conensus_clusters)
    window.show()

    app.exec_()