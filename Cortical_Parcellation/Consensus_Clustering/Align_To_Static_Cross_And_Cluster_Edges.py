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


class masking_window(QWidget):

    def __init__(self, session_list, parent=None):
        super(masking_window, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("Align To Static Cross")
        self.setGeometry(0, 0, 1900, 500)

        # Setup Internal Variables
        self.session_list = session_list
        self.number_of_sessions = len(self.session_list)
        self.current_template_index = 0
        self.current_matching_index = 0

        # Get List Of Max Projections
        self.max_projection_list = []
        for session in session_list:
            max_projection = np.load(os.path.join(session, "max_projection.npy"))
            upper_bound = np.percentile(max_projection, 99)
            max_projection = np.divide(max_projection, upper_bound)
            max_projection = np.clip(max_projection, a_min=0, a_max=1)
            self.max_projection_list.append(max_projection)

        # Get List Of Cluster Edges
        self.cluster_edges_list = []
        for session in session_list:

            if os.path.exists(os.path.join(session, "Cluster_Edges.npy")):
                cluster_edges = np.load(os.path.join(session, "Cluster_Edges.npy"))
            else:
                cluster_edges = np.zeros(np.shape(self.max_projection_list[0]))
            self.cluster_edges_list.append(cluster_edges)
            
        # Get List Of Alignment Dictionaries
        self.variable_dictionary_list = []
        for session_index in range(self.number_of_sessions):

            variable_dictionary_directory = os.path.join(self.session_list[session_index],"Cluster_Alignment_Dictionary.npy")
            if os.path.exists(variable_dictionary_directory):
                print("IT Exists")
                variable_dictionary = np.load(variable_dictionary_directory, allow_pickle=True)[()]
            else:
                variable_dictionary = {'x_shift': 0, 'y_shift': 0, 'rotation': 0}

            self.variable_dictionary_list.append(variable_dictionary)

        # Set Current Images
        self.max_projection = self.max_projection_list[0]
        self.template_cluster_edges = self.cluster_edges_list[0]
        self.matching_cluster_edges = self.cluster_edges_list[0]

        # Cross Display View
        self.cross_display_view_widget = QWidget()
        self.cross_display_view_widget_layout = QGridLayout()
        self.cross_display_view = pyqtgraph.ImageView()
        self.cross_display_view.ui.histogram.hide()
        self.cross_display_view.ui.roiBtn.hide()
        self.cross_display_view.ui.menuBtn.hide()
        self.cross_display_view_widget_layout.addWidget(self.cross_display_view, 0, 0)
        self.cross_display_view_widget.setLayout(self.cross_display_view_widget_layout)
        self.cross_display_view_widget.setFixedWidth(608)
        self.cross_display_view_widget.setFixedHeight(600)
        self.cross_display_view.setLevels(0, 1)
        #self.cross_display_view.setImage(max_projection)

        # Regressor Display View
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
        #self.regressor_display_view.setImage(max_projection)

        # Create Session Labels
        self.template_session_label = QLabel("Template Session: ")
        self.matching_session_label = QLabel("Matching Session: ")

        # Create Session List Views
        self.template_session_list_widget = QListWidget()
        self.matching_session_list_widget = QListWidget()

        self.matching_session_list_widget.currentRowChanged.connect(self.load_matching_session)
        self.template_session_list_widget.currentRowChanged.connect(self.load_template_session)

        for session in self.session_list:
            session_name = session.split('/')[-1]
            self.template_session_list_widget.addItem(session_name)
            self.matching_session_list_widget.addItem(session_name)

        self.matching_session_list_widget.setCurrentRow(self.current_matching_index)

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

        # Add Labels
        self.layout.addWidget(self.template_session_label,          0, 0, 1, 2)
        self.layout.addWidget(self.matching_session_label,          0, 2, 1, 2)

        # Add Display Views
        self.layout.addWidget(self.edges_display_view_widget,   1, 1, 25, 1)
        self.layout.addWidget(self.cross_display_view_widget,       1, 3, 25, 1)

        # Add List Views
        self.layout.addWidget(self.template_session_list_widget,   1, 0, 25, 1)
        self.layout.addWidget(self.matching_session_list_widget, 1, 2, 25, 1)


        # Add Transformation Controls
        self.layout.addWidget(self.left_button,                     2,  6, 1, 1)
        self.layout.addWidget(self.right_button,                    3,  6, 1, 1)
        self.layout.addWidget(self.up_button,                       4,  6, 1, 1)
        self.layout.addWidget(self.down_button,                     5,  6, 1, 1)
        self.layout.addWidget(self.rotate_clockwise_button,         6,  6, 1, 1)
        self.layout.addWidget(self.rotate_counterclockwise_button,  7,  6, 1, 1)
        self.layout.addWidget(self.x_label,                         8,  6, 1, 1)
        self.layout.addWidget(self.y_label,                         9,  6, 1, 1)
        self.layout.addWidget(self.angle_label,                     10, 6, 1, 1)
        self.layout.addWidget(self.map_button,                      11, 6, 1, 1)


        # Add ROI
        horizontal_line_coords = [[304, 500], [304, 25]]
        vertical_line_coords = [[202, 100], [402, 100]]

        self.horizontal_roi = pyqtgraph.PolyLineROI(horizontal_line_coords, closed=False)
        self.vertical_roi = pyqtgraph.PolyLineROI(vertical_line_coords, closed=False)

        self.cross_display_view.addItem(self.horizontal_roi)
        self.cross_display_view.addItem(self.vertical_roi)


    def draw_images(self):
        
        transformed_max_projection = self.transform_image(self.max_projection, self.variable_dictionary_list[self.current_matching_index])
        transformed_matching_edges = self.transform_image(self.matching_cluster_edges, self.variable_dictionary_list[self.current_matching_index])
        transformed_template_edges = self.transform_image(self.template_cluster_edges, self.variable_dictionary_list[self.current_template_index])

        combined_edges_template = np.zeros((600, 608, 3))
        combined_edges_template[:, :, 0] = transformed_matching_edges
        combined_edges_template[:, :, 1] = transformed_template_edges

        self.cross_display_view.setImage(transformed_max_projection)
        self.edges_display_view.setImage(combined_edges_template)



        
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
        self.variable_dictionary_list[self.current_matching_index]['x_shift'] = self.variable_dictionary_list[self.current_matching_index]['x_shift'] + 1
        self.x_label.setText("x: " + str(self.variable_dictionary_list[self.current_matching_index]['x_shift']))
        self.draw_images()

    def move_right(self):
        self.variable_dictionary_list[self.current_matching_index]['x_shift'] = self.variable_dictionary_list[self.current_matching_index]['x_shift'] - 1
        self.x_label.setText("x: " + str(self.variable_dictionary_list[self.current_matching_index]['x_shift']))
        self.draw_images()

    def move_up(self):
        self.variable_dictionary_list[self.current_matching_index]['y_shift'] = self.variable_dictionary_list[self.current_matching_index]['y_shift'] - 1
        self.y_label.setText("y: " + str(self.variable_dictionary_list[self.current_matching_index]['y_shift']))
        self.draw_images()

    def move_down(self):
        self.variable_dictionary_list[self.current_matching_index]['y_shift'] = self.variable_dictionary_list[self.current_matching_index]['y_shift'] + 1
        self.y_label.setText("y: " + str(self.variable_dictionary_list[self.current_matching_index]['y_shift']))
        self.draw_images()

    def rotate_clockwise(self):
        self.variable_dictionary_list[self.current_matching_index]['rotation'] = self.variable_dictionary_list[self.current_matching_index]['rotation'] - 1
        self.angle_label.setText("Angle: " + str(self.variable_dictionary_list[self.current_matching_index]['rotation']))
        self.draw_images()

    def rotate_counterclockwise(self):
        self.variable_dictionary_list[self.current_matching_index]['rotation'] = self.variable_dictionary_list[self.current_matching_index]['rotation'] + 1
        self.angle_label.setText("Angle: " + str(self.variable_dictionary_list[self.current_matching_index]['rotation']))
        self.draw_images()


    def set_alignment(self):

        for session_index in  range(self.number_of_sessions):

            # Get Save Directory
            session_directory = self.session_list[session_index]
            save_directory = os.path.join(session_directory, "Cluster_Alignment_Dictionary.npy")

            # Save Dictionary
            np.save(save_directory, self.variable_dictionary_list[session_index])




    def load_matching_session(self):
        
        self.current_matching_index = self.matching_session_list_widget.currentRow()
        
        self.max_projection = self.max_projection_list[self.current_matching_index]
        self.matching_cluster_edges = self.cluster_edges_list[self.current_matching_index]
        self.draw_images()
        #self.cross_display_view.setImage(max_projection)


    def load_template_session(self):

        self.current_template_index = self.template_session_list_widget.currentRow()
        self.template_cluster_edges = self.cluster_edges_list[self.current_template_index]
        self.draw_images()


def align_sessions(session_list):

    app = QApplication(sys.argv)

    window = masking_window(session_list)
    window.show()

    app.exec_()


session_list = [

    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_01_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_03_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_05_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_07_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_09_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_11_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_13_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_15_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_17_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_19_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_22_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_24_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging",

    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging",

    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_04_29_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_01_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_03_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_05_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_07_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_09_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",

    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_09_25_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_09_29_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A//2021_10_01_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_10_03_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_10_05_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_10_07_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK22.1A/2021_11_05_Transition_Imaging",



    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_15_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_16_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_17_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_19_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_23_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_25_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1D/2020_11_14_Discrimination_Imaging",

    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_16_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_17_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_19_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_21_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_24_Discrimination_Imaging",

    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_08_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_10_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_12_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_14_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_16_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_18_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_23_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_25_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_27_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_03_01_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_03_03_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging",

    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_09_28_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_09_30_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_09_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_11_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_13_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_15_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_17_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging",

    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK16.1B/2021_04_30_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK16.1B/2021_05_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK16.1B/2021_05_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK16.1B/2021_05_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK16.1B/2021_05_08_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK16.1B/2021_05_10_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK16.1B/2021_05_12_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK16.1B/2021_05_14_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK16.1B/2021_05_16_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK16.1B/2021_05_18_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK16.1B/2021_05_20_Discrimination_Imaging",


]


align_sessions(session_list)
