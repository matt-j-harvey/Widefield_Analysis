import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage import measure
from skimage.draw import polygon
import sys

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph

pyqtgraph.setConfigOptions(imageAxisOrder='row-major')




def get_roi_masks(binary_image):

    image_height, image_width = np.shape(binary_image)

    mask_list = []
    polygon_list = []

    # Find Contours
    contours = measure.find_contours(binary_image)
    for contour in contours:
        contour_polygon = polygon(contour[:, 0], contour[:, 1], shape=(image_height, image_width))

        mask = np.zeros((image_height, image_width))
        mask[contour_polygon] = 1
        mask_list.append(mask)
        polygon_list.append(contour_polygon)

    return mask_list, polygon_list

class roi_class():

    def __init__(self, index, mask, indicies, selected, sign):
        self.index = index
        self.mask = mask
        self.indicies = indicies
        self.selected = selected
        self.sign = sign


def segment_retinotopy(base_directory):

    # Load Combined Sign Map
    combined_sign_map = np.load(os.path.join(base_directory, "Combined_Sign_map.npy"))

    # Binarise Sign Map
    positive_sign_map = np.where(combined_sign_map < 0.2, 1, 0)
    negative_sign_map = np.where(combined_sign_map > -0.2, 1, 0)

    positive_roi_masks, positve_roi_indicies = get_roi_masks(positive_sign_map)
    negative_roi_masks, negative_roi_indicies = get_roi_masks(negative_sign_map)

    number_of_positive_rois = len(positive_roi_masks)
    number_of_negative_rois = len(negative_roi_masks)

    # Create ROI List
    roi_index = 1
    roi_list = []

    for roi in range(number_of_positive_rois):
        roi_object = roi_class(index=roi_index, mask=positive_roi_masks[roi], indicies=positve_roi_indicies[roi], selected=0, sign=1)
        roi_list.append(roi_object)
        roi_index += 1

    for roi in range(number_of_negative_rois):
        roi_object = roi_class(index=roi_index, mask=negative_roi_masks[roi], indicies=negative_roi_indicies[roi], selected=0, sign=-1)
        roi_list.append(roi_object)
        roi_index += 1

    # Create Index Map
    index_map = np.zeros(np.shape(combined_sign_map))
    for roi in roi_list:
        index_map[roi.indicies] = roi.index

    return combined_sign_map, roi_list, index_map





class curation_window(QWidget):

    def __init__(self, base_directory, combined_sign_map, roi_list, index_map, parent=None):
        super(curation_window, self).__init__(parent)


        # Setup Window
        self.setWindowTitle("Curate Retinotopic Mapping")
        self.setGeometry(0, 0, 1900, 500)
        self.base_directory = base_directory
        self.index_map = index_map
        self.combined_sign_map = combined_sign_map
        self.roi_list = roi_list
        self.image_height, self.image_width = np.shape(combined_sign_map)
        self.selected_sign_map = np.zeros((self.image_height, self.image_width))

        # Load Diagram Image
        self.diagram_pixmap = QPixmap(r"/home/matthew/Pictures/Retinotopy.png")
        print("Diagram pixmap", self.diagram_pixmap)
        self.diagram_label = QLabel("Diagram")
        self.diagram_label.setPixmap(self.diagram_pixmap.scaled(500, 1000, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # Cross Display View
        self.retinotopy_display_view_widget = QWidget()
        self.retinotopy_display_view_widget_layout = QGridLayout()
        self.retinotopy_display_view = pyqtgraph.ImageView()
        self.retinotopy_display_view.ui.histogram.hide()
        self.retinotopy_display_view.ui.roiBtn.hide()
        self.retinotopy_display_view.ui.menuBtn.hide()
        self.retinotopy_display_view_widget_layout.addWidget(self.retinotopy_display_view, 0, 0)
        self.retinotopy_display_view_widget.setLayout(self.retinotopy_display_view_widget_layout)
        self.retinotopy_display_view_widget.setFixedWidth(608)
        self.retinotopy_display_view_widget.setFixedHeight(600)
        self.retinotopy_display_view.getView().scene().sigMouseMoved.connect(lambda pos: self.mouse_move_all_rois(pos, self.retinotopy_display_view))
        self.retinotopy_display_view.getView().scene().sigMouseClicked.connect(lambda pos: self.click_on_unselected_display_view(pos, self.retinotopy_display_view))

        # Cross Display View
        self.selected_display_view_widget = QWidget()
        self.selected_display_view_widget_layout = QGridLayout()
        self.selected_display_view = pyqtgraph.ImageView()
        self.selected_display_view.ui.histogram.hide()
        self.selected_display_view.ui.roiBtn.hide()
        self.selected_display_view.ui.menuBtn.hide()
        self.selected_display_view_widget_layout.addWidget(self.selected_display_view, 0, 0)
        self.selected_display_view_widget.setLayout(self.selected_display_view_widget_layout)
        self.selected_display_view_widget.setFixedWidth(608)
        self.selected_display_view_widget.setFixedHeight(600)
        self.selected_display_view.getView().scene().sigMouseMoved.connect(lambda pos: self.mouse_move_selected_rois(pos, self.selected_display_view))
        self.selected_display_view.getView().scene().sigMouseClicked.connect(lambda pos: self.click_on_selected_display_view(pos, self.selected_display_view))

        # Create Labels
        self.all_roi_label = QLabel("All ROIs")
        self.selected_roi_label = QLabel("Selected ROIs")

        # Create Save Button
        self.save_button = QPushButton("Save Selected ROIs")
        self.save_button.clicked.connect(self.save_map)

        # Set Colourmap
        colour_list = [[0, 0.87, 0.9, 1],
                       [0, 0, 1, 1],
                       [0, 0, 0, 1],
                       [1, 0, 0, 1],
                       [1, 1, 0, 1], ]
        colour_list = np.array(colour_list)
        colour_list = np.multiply(colour_list, 255)
        value_list = np.linspace(0, 1, num=len(colour_list))
        print("Valye list", value_list)
        self.colourmap = pyqtgraph.ColorMap(pos=value_list, color=colour_list)

        self.retinotopy_display_view.setImage(combined_sign_map)
        self.retinotopy_display_view.setLevels(-1.5, 1.5)
        self.retinotopy_display_view.setColorMap(self.colourmap)
        self.selected_display_view.setLevels(-1.5, 1.5)
        self.selected_display_view.setColorMap(self.colourmap)

        # Create and Set Layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Add Labels
        self.layout.addWidget(self.diagram_label, 0, 0, 1, 2)
        self.layout.addWidget(self.all_roi_label, 1, 0, 1, 1)
        self.layout.addWidget(self.selected_roi_label, 1, 1, 1, 1)
        self.layout.addWidget(self.retinotopy_display_view_widget, 2, 0, 1, 1)
        self.layout.addWidget(self.selected_display_view_widget, 2, 1, 1, 1)
        self.layout.addWidget(self.save_button, 3, 0, 1, 2)
        self.show()



    def click_on_selected_display_view(self, event, imageview):
        pos = event.scenePos()
        pos = imageview.getImageItem().mapFromScene(pos)
        y = np.clip(int(pos.y()), a_min=0, a_max=self.image_height - 1)
        x = np.clip(int(pos.x()), a_min=0, a_max=self.image_width - 1)
        selected_roi = int(self.index_map[y, x])

        if selected_roi != 0:
            selected_roi = selected_roi - 1

            print("Clicked on: ", selected_roi)
            self.roi_list[selected_roi].selected = 0
            selected_roi_indicies = self.roi_list[selected_roi].indicies
            self.selected_sign_map[selected_roi_indicies] = 0

        self.selected_display_view.setImage(self.selected_sign_map)
        self.selected_display_view.setLevels(-1.5, 1.5)



    def click_on_unselected_display_view(self, event, imageview):
        pos = event.scenePos()
        pos = imageview.getImageItem().mapFromScene(pos)
        y = np.clip(int(pos.y()), a_min=0, a_max=self.image_height - 1)
        x = np.clip(int(pos.x()), a_min=0, a_max=self.image_width - 1)
        selected_roi = int(self.index_map[y, x])

        if selected_roi != 0:
            selected_roi = selected_roi - 1

            print("Clicked on: ", selected_roi)
            self.roi_list[selected_roi].selected = 1
            selected_roi_indicies = self.roi_list[selected_roi].indicies
            selected_roi_sign = self.roi_list[selected_roi].sign

            # Add To Selected
            self.selected_sign_map[selected_roi_indicies] = self.combined_sign_map[selected_roi_indicies]

        self.selected_display_view.setImage(self.selected_sign_map)
        self.selected_display_view.setLevels(-1.5, 1.5)


    def mouse_move_all_rois(self, pos, imageview):
        pos = imageview.getImageItem().mapFromScene(pos)
        y = np.clip(int(pos.y()), a_min=0, a_max=self.image_height - 1)
        x = np.clip(int(pos.x()), a_min=0, a_max=self.image_width - 1)
        selected_roi = int(self.index_map[y, x])

        if selected_roi != 0:
            selected_roi = selected_roi - 1
            selected_roi_indicies = self.roi_list[selected_roi].indicies
            selected_roi_sign = self.roi_list[selected_roi].sign

            new_combined_sign_map = np.copy(self.combined_sign_map)
            if selected_roi_sign == 1:
                new_combined_sign_map[selected_roi_indicies] = 1.5
            elif selected_roi_sign == -1:
                new_combined_sign_map[selected_roi_indicies] = -1.5

            self.retinotopy_display_view.setImage(new_combined_sign_map)
            self.retinotopy_display_view.setLevels(-1.5, 1.5)


    def mouse_move_selected_rois(self, pos, imageview):
        pos = imageview.getImageItem().mapFromScene(pos)
        y = np.clip(int(pos.y()), a_min=0, a_max=self.image_height - 1)
        x = np.clip(int(pos.x()), a_min=0, a_max=self.image_width - 1)
        selected_roi = int(self.index_map[y, x])

        if selected_roi != 0:
            selected_roi = selected_roi - 1
            roi_status = self.roi_list[selected_roi].selected

            if roi_status == 1:
                selected_roi_indicies = self.roi_list[selected_roi].indicies
                selected_roi_sign = self.roi_list[selected_roi].sign

                new_combined_sign_map = np.copy(self.selected_sign_map)
                if selected_roi_sign == 1:
                    new_combined_sign_map[selected_roi_indicies] = 1.5
                elif selected_roi_sign == -1:
                    new_combined_sign_map[selected_roi_indicies] = -1.5

                self.selected_display_view.setImage(new_combined_sign_map)
                self.selected_display_view.setLevels(-1.5, 1.5)

    def save_map(self):
        np.save(os.path.join(base_directory, "Curated_Combined_Retinotopy.npy"),  self.selected_sign_map)


session_list = [

    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_26_Continous_Retinotopy_Left",
    "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_01_Continuous_Retinotopic_Mapping_Left",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_01_Continous_Retinotopy_Left",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_01_Continous_Retinotopy_Left",
    "/media/matthew/Expansion/Control_Data/NXAK14.1A/Continous_Retinotopic_Mapping_Left",


    ]

for base_directory in session_list:

    # Segement Retinotopy
    combined_sign_map, roi_list, index_map = segment_retinotopy(base_directory)

    # Manually Curate This
    app = QApplication(sys.argv)
    window = curation_window(base_directory, combined_sign_map, roi_list, index_map)
    app.exec_()
