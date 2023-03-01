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
from datetime import datetime

from Widefield_Utils import widefield_utils
from Files import Session_List

pyqtgraph.setConfigOptions(imageAxisOrder='row-major')





class roi_selection_window(QWidget):

    def __init__(self, example_image, index_map, save_directory, parent=None):
        super(roi_selection_window, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("ROI Selector")
        self.setGeometry(0, 0, 1900, 500)

        # Setup Internal Variables
        self.example_image = example_image
        self.index_map = index_map
        self.save_directory = save_directory
        self.roi_name = datetime.now().strftime("%Y_%M_%d_%H_%m")
        print("ROI name", self.roi_name)


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
        self.brain_display_view.setImage(self.example_image)

        self.roi_name_input = QLineEdit(str(self.roi_name))

        # Create Map Button
        self.map_button = QPushButton("Map Region")
        self.map_button.clicked.connect(self.map_region)

        # Create and Set Layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Add Display Views
        self.layout.addWidget(self.brain_display_view_widget, 0, 0, 1, 10)
        self.layout.addWidget(self.map_button, 1, 0, 1, 1)
        self.layout.addWidget(self.roi_name_input, 1, 1, 1, 1)

        # Add ROI
        self.brain_roi_left = pyqtgraph.RectROI([1, 1], [15, 50])
        self.brain_roi_right = pyqtgraph.RectROI([200, 1], [15, 50])
        self.brain_display_view.addItem(self.brain_roi_left)
        self.brain_display_view.addItem(self.brain_roi_right)


    def map_region(self):


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
        left_index_map_slice  = self.index_map[left_pos_y: left_pos_y  + left_size_y,  left_pos_x: left_pos_x  + left_size_x]
        right_index_map_slice = self.index_map[right_pos_y:right_pos_y + right_size_y, right_pos_x:right_pos_x + right_size_x]

        left_roi_indicies = np.ndarray.flatten(left_index_map_slice)
        right_roi_indicies = np.ndarray.flatten(right_index_map_slice)

        roi_indicies = np.concatenate([left_roi_indicies, right_roi_indicies])
        roi_indicies = np.ndarray.astype(roi_indicies, int)

        """
        indicies, image_height, image_width = widefield_utils.load_tight_mask()
        indicies = np.array(indicies)
        selected_indicies = indicies[:, roi_indicies]
        template = np.zeros(300*304)
        template[selected_indicies] = 1
        template = np.reshape(template, (300, 304))
        plt.imshow(template)
        plt.show()
        """

        # Save These
        roi_filename = os.path.join(self.save_directory, str(self.roi_name_input.text() + ".npy"))
        np.save(roi_filename, roi_indicies)
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


def load_brain_images(session):

    # Load Example Image
    max_projection = np.load(os.path.join(session, "Blue_Example_Image.npy"))
    print("Max Projection Shape", np.shape(max_projection))
    upper_bound = np.percentile(max_projection, 99)
    max_projection = np.divide(max_projection, upper_bound)
    max_projection = np.clip(max_projection, a_min=0, a_max=1)

    # Load Alignment Dictionaries
    within_mouse_alignment_dictionary = np.load(os.path.join(session, "Within_Mouse_Alignment_Dictionary.npy"), allow_pickle=True)[()]
    across_mouse_alignment_dictionary = widefield_utils.load_across_mice_alignment_dictionary(session)

    # Transform Image
    max_projection = widefield_utils.transform_image(max_projection, within_mouse_alignment_dictionary)
    max_projection = widefield_utils.transform_image(max_projection, across_mouse_alignment_dictionary)

    return max_projection



def create_mask_index_map(indicies, image_height, image_width):
    number_of_indicies = np.shape(indicies)[1]
    index_map = np.zeros(image_height * image_width)
    index_map[indicies] = list(range(number_of_indicies))
    index_map = np.reshape(index_map, (image_height, image_width))
    return index_map


if __name__ == '__main__':

    app = QApplication(sys.argv)

    # ROI Save Directory
    roi_save_directory = r"/media/matthew/Expansion/Custom_ROIs"

    # Create Index MAp
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    index_map = create_mask_index_map(indicies, image_height, image_width)

    # Get Example Image
    example_directory = r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_10_Transition_Imaging"
    example_image = load_brain_images(example_directory)

    selection_window = roi_selection_window(example_image, index_map, roi_save_directory)
    selection_window.show()

    app.exec_()


