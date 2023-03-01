import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.transform import resize

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph

import os
import sys


from Widefield_Utils import widefield_utils

pyqtgraph.setConfigOptions(imageAxisOrder='row-major')


class masking_window(QWidget):

    def __init__(self, session_list, atlas_outline, parent=None):
        super(masking_window, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("Align Allen Atlas")
        self.setGeometry(0, 0, 1200, 500)

        # Setup Internal Variables
        self.session_list = session_list
        self.atlas_outline = atlas_outline
        self.number_of_sessions = len(self.session_list)
        self.example_image_list = []
        self.current_session_index = 0
        self.atlas_variable_dictionary = {'x_shift': 0, 'y_shift': 0, 'rotation': 0, 'x_scale': 0.5, 'y_scale': 0.5}
        self.image_height = 300
        self.image_width = 304
        self.growth_increment = 0.01

        # Load Combined Retinotopy Images
        for session_index in range(self.number_of_sessions):

            # Load Combined Retinotopy Image - These Are Aligned Within Mice
            combined_retinotopy_image = np.load(os.path.join(self.session_list[session_index], "Combined_Sign_map_with_anatomy.npy"))
            print(np.shape(combined_retinotopy_image))

            # Load Across Mouse Alignment Dictionary
            across_mouse_alignment_dict = widefield_utils.load_across_mice_alignment_dictionary(self.session_list[session_index])

            # Align Across Mouse
            image_height, image_width, colour_depth = np.shape(combined_retinotopy_image)
            combined_retinotopy_image_r = widefield_utils.transform_image(combined_retinotopy_image[:, :, 0], across_mouse_alignment_dict)
            combined_retinotopy_image_g = widefield_utils.transform_image(combined_retinotopy_image[:, :, 1], across_mouse_alignment_dict)
            combined_retinotopy_image_b = widefield_utils.transform_image(combined_retinotopy_image[:, :, 2], across_mouse_alignment_dict)

            across_mouse_aligned_combined_retinotopy_image = np.zeros((image_height, image_width, 3))
            across_mouse_aligned_combined_retinotopy_image[:, :, 0] = combined_retinotopy_image_r
            across_mouse_aligned_combined_retinotopy_image[:, :, 1] = combined_retinotopy_image_g
            across_mouse_aligned_combined_retinotopy_image[:, :, 2] = combined_retinotopy_image_b

            # Add To List
            self.example_image_list.append(across_mouse_aligned_combined_retinotopy_image)

        # Get List Of Alignment Dictionaries
        self.variable_dictionary_list = []
        for session_index in range(self.number_of_sessions):

            variable_dictionary_directory = os.path.join(self.session_list[session_index], "Churchland_Atlas_Alignment_Dict.npy")
            if os.path.exists(variable_dictionary_directory):
                print("IT Exists")
                variable_dictionary = np.load(variable_dictionary_directory, allow_pickle=True)[()]
            else:
                variable_dictionary = {'x_shift': 0, 'y_shift': 0, 'rotation': 0, 'x_scale': 1, 'y_scale':1}

            self.variable_dictionary_list.append(variable_dictionary)


        # Set Current Images
        self.max_projection = self.example_image_list[0]

        # Cross Display View
        self.atlas_display_view_widget = QWidget()
        self.atlas_display_view_widget_layout = QGridLayout()
        self.atlas_display_view = pyqtgraph.ImageView()
        self.atlas_display_view.ui.histogram.hide()
        self.atlas_display_view.ui.roiBtn.hide()
        self.atlas_display_view.ui.menuBtn.hide()
        self.atlas_display_view_widget_layout.addWidget(self.atlas_display_view, 0, 0)
        self.atlas_display_view_widget.setLayout(self.atlas_display_view_widget_layout)
        self.atlas_display_view_widget.setFixedWidth(608)
        self.atlas_display_view_widget.setFixedHeight(600)
        self.atlas_display_view.setLevels(0, 1)
        # self.cross_display_view.setImage(max_projection)

        # Create Session Labels
        self.session_label = QLabel("Session: ")

        # Create Session List Views
        self.session_list_widget = QListWidget()
        self.session_list_widget.currentRowChanged.connect(self.load_session)
        for session in self.session_list:
            session_name = session.split('/')[-1]
            self.session_list_widget.addItem(session_name)
        self.session_list_widget.setCurrentRow(self.current_session_index)


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

        self.grow_x_button = QPushButton("Grow X")
        self.grow_x_button.clicked.connect(self.grow_x)

        self.shrink_x_button = QPushButton("Shrink X")
        self.shrink_x_button.clicked.connect(self.shrink_x)

        self.grow_y_button = QPushButton("Grow Y")
        self.grow_y_button.clicked.connect(self.grow_y)

        self.shrink_y_button = QPushButton("Shrink Y")
        self.shrink_y_button.clicked.connect(self.shrink_y)

        self.map_button = QPushButton("Set Alignment")
        self.map_button.clicked.connect(self.set_alignment)

        # Add Labels
        self.x_label = QLabel("x: 0")
        self.y_label = QLabel("y: 0")
        self.angle_label = QLabel("angle: 0")
        self.x_scale_label = QLabel("x_scale: 1")
        self.y_scale_label = QLabel("y_scale: 1")

        # Create and Set Layout]
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Add Labels
        self.layout.addWidget(self.session_label, 0, 0, 1, 2)

        # Add Display Views
        self.layout.addWidget(self.atlas_display_view_widget, 1, 0, 25, 1)

        # Add List Views
        self.layout.addWidget(self.session_list_widget, 1, 1, 25, 1)

        # Add Transformation Controls
        self.layout.addWidget(self.left_button, 2, 6, 1, 1)
        self.layout.addWidget(self.right_button, 3, 6, 1, 1)
        self.layout.addWidget(self.up_button, 4, 6, 1, 1)
        self.layout.addWidget(self.down_button, 5, 6, 1, 1)
        self.layout.addWidget(self.rotate_clockwise_button, 6, 6, 1, 1)
        self.layout.addWidget(self.rotate_counterclockwise_button, 7, 6, 1, 1)
        self.layout.addWidget(self.grow_x_button, 8, 6, 1, 1)
        self.layout.addWidget(self.shrink_x_button, 9, 6, 1, 1)
        self.layout.addWidget(self.grow_y_button, 10, 6, 1, 1)
        self.layout.addWidget(self.shrink_y_button, 11, 6, 1, 1)

        self.layout.addWidget(self.x_label, 12, 6, 1, 1)
        self.layout.addWidget(self.y_label, 13, 6, 1, 1)
        self.layout.addWidget(self.angle_label, 14, 6, 1, 1)
        self.layout.addWidget(self.x_scale_label, 15, 6, 1, 1)
        self.layout.addWidget(self.y_scale_label, 16, 6, 1, 1)
        self.layout.addWidget(self.map_button, 17, 6, 1, 1)


    def draw_images(self):

        # Get Current Brain Image
        current_brain_image = np.copy(self.example_image_list[self.current_session_index])

        # Transform Atlas
        transformed_atlas = self.transform_mask_or_atlas(self.atlas_outline, self.atlas_variable_dictionary)

        # Add Atlas To Brain Image
        atlas_indicies = np.nonzero(transformed_atlas)
        current_brain_image[atlas_indicies] = [0, 0, 0]

        # Set Image
        self.atlas_display_view.setImage(current_brain_image)



    def transform_mask_or_atlas(self, image, variable_dictionary):

        # Unpack Dictionary
        angle = variable_dictionary['rotation']
        x_shift = variable_dictionary['x_shift']
        y_shift = variable_dictionary['y_shift']
        x_scale = variable_dictionary['x_scale']
        y_scale = variable_dictionary['y_scale']

        # Copy
        transformed_image = np.copy(image)

        # Scale
        original_height, original_width = np.shape(transformed_image)
        new_height = int(original_height * y_scale)
        new_width = int(original_width * x_scale)
        transformed_image = resize(transformed_image, (new_height, new_width), preserve_range=True)
        print("new image height", np.shape(transformed_image))

        # Rotate
        transformed_image = ndimage.rotate(transformed_image, angle, reshape=False, prefilter=True)

        # Insert Into Background
        mask_height, mask_width = np.shape(transformed_image)
        centre_x = 200
        centre_y = 200
        background_array = np.zeros((1000, 1000))
        x_start = centre_x + x_shift
        x_stop = x_start + mask_width

        y_start = centre_y + y_shift
        y_stop = y_start + mask_height

        background_array[y_start:y_stop, x_start:x_stop] = transformed_image

        # Take Chunk
        transformed_image = background_array[centre_y:centre_y + self.image_height, centre_x:centre_x + self.image_width]

        # Rebinarize
        transformed_image = np.where(transformed_image > 0.5, 1, 0)

        return transformed_image


    def move_left(self):
        self.atlas_variable_dictionary['x_shift'] = self.atlas_variable_dictionary['x_shift'] + 1
        self.x_label.setText("x: " + str(self.atlas_variable_dictionary['x_shift']))
        self.draw_images()

    def move_right(self):
        self.atlas_variable_dictionary['x_shift'] = self.atlas_variable_dictionary['x_shift'] - 1
        self.x_label.setText("x: " + str(self.atlas_variable_dictionary['x_shift']))
        self.draw_images()

    def move_up(self):
        self.atlas_variable_dictionary['y_shift'] = self.atlas_variable_dictionary['y_shift'] - 1
        self.y_label.setText("y: " + str(self.atlas_variable_dictionary['y_shift']))
        self.draw_images()

    def move_down(self):
        self.atlas_variable_dictionary['y_shift'] = self.atlas_variable_dictionary['y_shift'] + 1
        self.y_label.setText("y: " + str(self.atlas_variable_dictionary['y_shift']))
        self.draw_images()

    def rotate_clockwise(self):
        self.atlas_variable_dictionary['rotation'] = self.atlas_variable_dictionary['rotation'] - 1
        self.angle_label.setText("x: " + str(self.atlas_variable_dictionary['rotation']))
        self.draw_images()

    def rotate_counterclockwise(self):
        self.atlas_variable_dictionary['rotation'] = self.atlas_variable_dictionary['rotation'] + 1
        self.angle_label.setText("x: " + str(self.atlas_variable_dictionary['rotation']))
        self.draw_images()

    def grow_x(self):
        self.atlas_variable_dictionary['x_scale'] = self.atlas_variable_dictionary['x_scale'] + self.growth_increment
        self.x_scale_label.setText("x: " + str(self.atlas_variable_dictionary['x_scale']))
        self.draw_images()

    def grow_y(self):
        self.atlas_variable_dictionary['y_scale'] = self.atlas_variable_dictionary['y_scale'] + self.growth_increment
        self.y_scale_label.setText("y: " + str(self.atlas_variable_dictionary['y_scale']))
        self.draw_images()

    def shrink_x(self):
        self.atlas_variable_dictionary['x_scale'] = self.atlas_variable_dictionary['x_scale'] - self.growth_increment
        self.x_scale_label.setText("x: " + str(self.atlas_variable_dictionary['x_scale']))
        self.draw_images()

    def shrink_y(self):
        self.atlas_variable_dictionary['y_scale'] = self.atlas_variable_dictionary['y_scale'] - self.growth_increment
        self.y_scale_label.setText("y: " + str(self.atlas_variable_dictionary['y_scale']))
        self.draw_images()


    def set_alignment(self):

        # Save Dictionaries
        np.save(os.path.join(session_list[self.current_session_index], "Churchland_Atlas_Alignment_Dictionary.npy"), self.atlas_variable_dictionary)


    def load_session(self):

        # Put Stuff In Here To Load Mask Or Load Allen Atlas
        self.current_template_index = self.template_session_list_widget.currentRow()
        self.draw_images()



def align_sessions(session_list, atlas):
    app = QApplication(sys.argv)

    window = masking_window(session_list, atlas)
    window.show()

    app.exec_()


#session_list = [r"/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.1D/2022_12_14_Retinotopy_Left"]
session_list = ["/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_01_Continuous_Retinotopic_Mapping_Left"]
#session_list = ["/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.1F/2022_12_15_Continuous_Retinotopic_Mapping_Left"]
atlas_outline_location = r"/home/matthew/Documents/Allen_Atlas_Templates/churchland_outlines.npy"
atlas_outline = np.load(atlas_outline_location)

align_sessions(session_list, atlas_outline)
