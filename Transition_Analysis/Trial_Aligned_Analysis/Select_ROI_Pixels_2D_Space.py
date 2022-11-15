import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from matplotlib.path import Path

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph

from scipy import ndimage
import os
import sys

from tqdm import tqdm

import Trial_Aligned_Utils

pyqtgraph.setConfigOptions(imageAxisOrder='row-major')





class roi_selection_window(QWidget):

    def __init__(self, session_list, image_list, parent=None):
        super(roi_selection_window, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("ROI Selector")
        self.setGeometry(0, 0, 1900, 500)

        # Setup Internal Variables
        self.session_list = session_list
        self.image_list = image_list

        # Regressor Display View
        self.brain_display_view_widget = QWidget()
        self.brain_display_view_widget_layout = QGridLayout()
        self.brain_display_view = pyqtgraph.ImageView()
        self.brain_display_view.ui.histogram.hide()
        self.brain_display_view.ui.roiBtn.hide()
        self.brain_display_view.ui.menuBtn.hide()
        self.brain_display_view_widget_layout.addWidget(self.brain_display_view, 0, 0)
        self.brain_display_view_widget.setLayout(self.brain_display_view_widget_layout)
        self.brain_display_view_widget.setMinimumWidth(604)
        self.brain_display_view_widget.setMinimumHeight(600)
        cm = pyqtgraph.colormap.get('CET-R4')
        self.brain_display_view.setColorMap(cm)
        self.brain_display_view.setImage(self.image_list[3])

        # Create Map Button
        self.map_button = QPushButton("Map Region")
        self.map_button.clicked.connect(self.map_region)

        # Create and Set Layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Add Display Views
        self.layout.addWidget(self.brain_display_view_widget, 0, 0, 1, 10)
        self.layout.addWidget(self.map_button, 1, 0, 1, 1)

        # Add ROI
        self.brain_roi_left = pyqtgraph.RectROI([1, 1], [15, 50])
        self.brain_roi_right = pyqtgraph.RectROI([200, 1], [15, 50])
        self.brain_display_view.addItem(self.brain_roi_left)
        self.brain_display_view.addItem(self.brain_roi_right)


    def map_region(self):

        # Create Index Map
        indicies, image_height, image_width = Trial_Aligned_Utils.load_tight_mask()
        number_of_indicies = np.shape(indicies)[1]

        index_map = np.zeros(image_height * image_width)
        index_map[indicies] = list(range(number_of_indicies))
        index_map = np.reshape(index_map, (image_height, image_width))

        plt.title("Index Map")
        plt.imshow(index_map)
        plt.show()

        # Get ROI Pos
        left_pos_x, left_pos_y = self.brain_roi_left.pos()  # POs is Lower Left corner
        left_size_x, left_size_y = self.brain_roi_left.size()
        left_pos_x = int(np.around(left_pos_x, 0))
        left_pos_y = int(np.around(left_pos_y, 0))
        left_size_x = int(np.around(left_size_x, 0))
        left_size_y = int(np.around(left_size_y, 0))

        # Get ROI Pos
        right_pos_x, right_pos_y = self.brain_roi_right.pos()  # POs is Lower Left corner
        right_size_x, right_size_y = self.brain_roi_right.size()
        right_pos_x = int(np.around(right_pos_x, 0))
        right_pos_y = int(np.around(right_pos_y, 0))
        right_size_x = int(np.around(right_size_x, 0))
        right_size_y = int(np.around(right_size_y, 0))

        # Get Slice Of Index Map
        left_index_map_slice  = index_map[left_pos_y: left_pos_y  + left_size_y,  left_pos_x: left_pos_x  + left_size_x]
        right_index_map_slice = index_map[right_pos_y:right_pos_y + right_size_y, right_pos_x:right_pos_x + right_size_x]
        left_roi_indicies = np.ndarray.flatten(left_index_map_slice)
        right_roi_indicies = np.ndarray.flatten(right_index_map_slice)
        roi_indicies = np.concatenate([left_roi_indicies, right_roi_indicies])
        roi_indicies = np.ndarray.astype(roi_indicies, int)


        # Save These
        for session in self.session_list:
            np.save(os.path.join(session, "Selected_ROI.npy"), roi_indicies)
        print("Mapped")


def transform_image(image, variable_dictionary):

    angle = variable_dictionary['rotation']
    x_shift = variable_dictionary['x_shift']
    y_shift = variable_dictionary['y_shift']

    transformed_image = np.copy(image)
    transformed_image = ndimage.rotate(transformed_image, angle, reshape=False, prefilter=True)
    transformed_image = np.roll(a=transformed_image, axis=0, shift=y_shift)
    transformed_image = np.roll(a=transformed_image, axis=1, shift=x_shift)

    return transformed_image


def load_brain_images(session_list):

    aligned_brain_image_list = []
    for session in session_list:

        # Load Example Image
        max_projection = np.load(os.path.join(session, "Blue_Example_Image.npy"))
        upper_bound = np.percentile(max_projection, 99)
        max_projection = np.divide(max_projection, upper_bound)
        max_projection = np.clip(max_projection, a_min=0, a_max=1)

        # Load Alignment Dictionary
        alignment_dictionary = np.load(os.path.join(session, "Brain_Alignment_Dictionary.npy"), allow_pickle=True)[()]

        # Transform Image
        max_projection = transform_image(max_projection, alignment_dictionary)

        aligned_brain_image_list.append(max_projection)

    return aligned_brain_image_list


if __name__ == '__main__':

    app = QApplication(sys.argv)

    session_list = [

        r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_23_Transition_Imaging",
        r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_31_Transition_Imaging",
        r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_04_02_Transition_Imaging",

        r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_02_Transition_Imaging",
        # r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_08_Transition_Imaging",
        r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_10_Transition_Imaging",

        r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_13_Transition_Imaging",
        r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_15_Transition_Imaging",
        r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_17_Transition_Imaging",

        r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_10_29_Transition_Imaging",
        r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_11_03_Transition_Imaging",
        r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_11_05_Transition_Imaging"

    ]

    # Get Aligned Images List
    aligned_images_list = load_brain_images(session_list)

    selection_window = roi_selection_window(session_list, aligned_images_list)
    selection_window.show()

    app.exec_()


