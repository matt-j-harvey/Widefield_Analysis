import matplotlib.pyplot as plt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph
import sys
import sip

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

pyqtgraph.setConfigOptions(imageAxisOrder='row-major')


class roi_selector(QWidget):

    def __init__(self, atlas_edges, atlas_regions, parent=None):
        super(roi_selector, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("ROI Selector")
        self.setGeometry(0, 0, 1000, 500)

        # Create Variable Holders
        self.atlas_edges = atlas_edges
        self.atlas_regions = atlas_regions
        self.current_stim_pattern_list = []

        self.image_height, self.image_width = np.shape(atlas_edges)
        #self.image_height = 600
        #self.image_width = 608
        self.current_stim_pattern = np.zeros((self.image_height, self.image_width))
        self.current_region = 0

        # Create Display Views
        self.atlas_display_view_label = QLabel("Atlas Regions")
        self.atlas_display_view, self.atlas_display_widget = self.create_display_widget()
        self.atlas_display_view.getView().scene().sigMouseMoved.connect(lambda pos: self.getPixel(pos, self.atlas_display_view))
        self.atlas_display_view.getView().scene().sigMouseClicked.connect(self.add_stim_region)
        self.atlas_display_view.setImage(self.atlas_edges)

        self.current_stim_pattern_display_label = QLabel("Current Stim Pattern")
        self.current_stim_pattern_display_view, self.current_stim_pattern_display_widget = self.create_display_widget()
        #self.column_correlation_map_display_view.setImage(MVAR_Utils.create_image_from_data(self.correlation_matrix[:, 0], indicies, image_height, image_width))

        # Create Intensity Spinner
        self.stim_intensity_spinner = QDoubleSpinBox()
        self.stim_intensity_spinner.setMinimum(0)
        self.stim_intensity_spinner.setMaximum(1)
        self.stim_intensity_spinner.setSingleStep(0.01)
        self.stim_intensity_spinner.setValue(1)
        self.stim_intensity_spinner_label = QLabel("Stim Intensity")

        # Create ROI add and Clear Buttons
        self.clear_stim_pattern_button = QPushButton("Clear Stim Pattern")
        self.clear_stim_pattern_button.clicked.connect(self.clear_stim_pattern)

        self.add_stim_pattern_button = QPushButton("Add Stim Pattern")
        self.add_stim_pattern_button.clicked.connect(self.add_stim_pattern)

        # Create Current List
        self.current_roi_list_label = QLabel("Stim Pattern List")
        self.current_roi_list_widget = QWidget()
        self.stim_label_list = []

        # Create Layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.layout.addWidget(self.atlas_display_view_label,            0, 0, 1, 2)
        self.layout.addWidget(self.atlas_display_widget,                1, 0, 1, 2)
        self.layout.addWidget(self.stim_intensity_spinner_label,        2, 0, 1, 1)
        self.layout.addWidget(self.stim_intensity_spinner,              2, 1, 1, 1)

        self.layout.addWidget(self.current_stim_pattern_display_label,  0, 2, 1, 2)
        self.layout.addWidget(self.current_stim_pattern_display_widget, 1, 2, 1, 2)
        self.layout.addWidget(self.clear_stim_pattern_button,           2, 2, 1, 1)
        self.layout.addWidget(self.add_stim_pattern_button,             2, 3, 1, 1)

        self.layout.addWidget(self.current_roi_list_label,              0, 4, 1, 2)
        self.layout.addWidget(self.current_roi_list_widget,              1, 4, 1, 2)

        #self.layout.addWidget(self.column_display_view_label, 0, 1, 1, 1)

        #self.layout.addWidget(self.row_correlation_map_display_widget,      1, 0, 1, 1)
        #self.layout.addWidget(self.column_correlation_map_display_widget,   1, 1, 1, 1)



    def clear_stim_pattern(self):
        self.current_stim_pattern = np.zeros((self.image_height, self.image_width))
        self.current_stim_pattern_display_view.setImage(self.current_stim_pattern)


    def add_stim_pattern(self):
        self.current_stim_pattern_list.append(self.current_stim_pattern)
        self.clear_stim_pattern()
        self.update_stim_pattern_list()


    def convert_numpy_array_to_q_pixmap(self, numpy_array):
        #numpy_array = np.divide(numpy_array, np.max(numpy_array))
        numpy_array = np.multiply(numpy_array, 255)
        numpy_array = numpy_array.astype(np.uint8)
        height, width = numpy_array.shape
        three_channel_image = np.zeros((height, width, 3), dtype=np.uint8)
        three_channel_image[:, :, 0] = numpy_array
        three_channel_image[:, :, 1] = numpy_array
        three_channel_image[:, :, 2] = numpy_array
        bytesPerLine = 3 * width
        qImg = QImage(three_channel_image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap01 = QPixmap.fromImage(qImg)
        return pixmap01


    def update_stim_pattern_list(self):

        # Remove Current List
        self.layout.removeWidget(self.current_roi_list_widget)
        sip.delete(self.current_roi_list_widget)
        self.stim_label_list = []

        # Create Updated List
        self.current_roi_list_widget = QWidget()
        self.roi_list_widget_layout = QGridLayout()
        self.current_roi_list_widget.setLayout(self.roi_list_widget_layout)

        stim_count = 0
        for stim_pattern in self.current_stim_pattern_list:
            print(np.shape(stim_pattern))

            stim_label = QLabel(str(stim_count))

            stim_pattern = resize(stim_pattern,(100,100))
            stim_pattern = self.convert_numpy_array_to_q_pixmap(stim_pattern)
            stim_label.setPixmap(stim_pattern)

            self.stim_label_list.append(stim_label)
            self.roi_list_widget_layout.addWidget(stim_label)

            stim_count += 1

        self.layout.addWidget(self.current_roi_list_widget, 1, 4, 1, 2)


    def getPixel(self, pos, imageview):
        pos = imageview.getImageItem().mapFromScene(pos)
        y = np.clip(int(pos.y()), a_min=0, a_max=self.image_height - 1)
        x = np.clip(int(pos.x()), a_min=0, a_max=self.image_width - 1)
        selected_region = int(self.atlas_regions[y, x])

        if selected_region != 0:
            region_mask = np.where(self.atlas_regions == selected_region, 1, 0)
            combined_image = np.add(self.atlas_edges, region_mask)
            self.atlas_display_view.setImage(combined_image)
            self.current_region = selected_region


    def add_stim_region(self, mouseClickEvent):

        # Get Intensity
        stim_intensity = float(self.stim_intensity_spinner.value())

        region_mask = np.where(self.atlas_regions == self.current_region, stim_intensity, 0)
        self.current_stim_pattern = np.add(self.current_stim_pattern, region_mask)
        self.current_stim_pattern = np.clip(self.current_stim_pattern, a_min=0, a_max=1)
        self.current_stim_pattern_display_view.setImage(self.current_stim_pattern)
        print("Currnet Region", self.current_region)


    def create_display_widget(self):

        # Create Figures
        display_view_widget = QWidget()
        display_view_widget_layout = QGridLayout()
        display_view = pyqtgraph.ImageView()
        display_view.ui.histogram.hide()
        display_view.ui.roiBtn.hide()
        display_view.ui.menuBtn.hide()
        display_view_widget_layout.addWidget(display_view, 0, 0)
        display_view_widget.setLayout(display_view_widget_layout)

        return display_view, display_view_widget




if __name__ == '__main__':

    app = QApplication(sys.argv)

    atlas_region_file = np.load(r"/home/matthew/Documents/Allen_Atlas_Templates/churchland_atlas.npy")
    atlas_edges_file = np.load(r"/home/matthew/Documents/Allen_Atlas_Templates/churchland_outlines.npy")

    plt.imshow(atlas_edges_file)
    plt.show()

    window = roi_selector(atlas_edges_file, atlas_region_file)
    window.showMaximized()

    app.exec_()















