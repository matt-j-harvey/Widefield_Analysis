import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph
import pandas as pd

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import os
from tqdm import tqdm

import tables
import sys

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Behaviour_Utils



class lick_threshold_window(QWidget):

    def __init__(self, session_directory_list, parent=None):
        super(lick_threshold_window, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("Set Lick Baseline")
        self.setGeometry(0, 0, 1500, 500)

        # Create Variable Holders
        self.session_directory_list = session_directory_list
        self.channel_dictionary = Behaviour_Utils.create_stimuli_dictionary()
        self.default_lick_threshold = 0.05
        self.current_lick_threshold = 0.05
        self.current_session_index = 0
        self.current_lick_trace = None

        # Create Widgets

        # Current Session Label
        self.session_label = QLabel("Session: " + str(session_directory_list[self.current_session_index]))

        # Lick Threshold Display Widget
        self.lick_display_view_widget = QWidget()
        self.lick_display_view_widget_layout = QGridLayout()
        self.lick_display_view = pyqtgraph.PlotWidget()
        self.lick_display_view_widget_layout.addWidget(self.lick_display_view, 0, 0)
        self.lick_display_view_widget.setLayout(self.lick_display_view_widget_layout)
        self.lick_display_view_widget.setMinimumWidth(1000)

        # Session List Widget
        self.session_list_widget = QListWidget()
        self.session_list_widget.setCurrentRow(0)
        self.session_list_widget.setFixedWidth(250)
        self.session_list_widget.currentRowChanged.connect(self.set_session)

        self.lick_trace_list = []
        self.lick_threshold_list = []
        self.threshold_status_list = []

        # load Data
        self.load_all_data()

        # Lick Threshold Spinner
        self.lick_threshold_spinner = QDoubleSpinBox()
        self.lick_threshold_spinner.setValue(self.current_lick_threshold)
        self.lick_threshold_spinner.setMinimum(0)
        self.lick_threshold_spinner.setMaximum(5)
        self.lick_threshold_spinner.valueChanged.connect(self.change_lick_threshold)
        self.lick_threshold_spinner.setSingleStep(0.01)

        # Set Lick Threshold Button
        self.set_lick_threshold_button = QPushButton("Set Lick Baseline")
        self.set_lick_threshold_button.clicked.connect(self.set_lick_threshold)

        # Create Layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Add Transformation Widgets
        self.layout.addWidget(self.session_label,                   0, 0, 1, 3)

        self.layout.addWidget(self.lick_display_view_widget,        1, 0, 1, 2)
        self.layout.addWidget(self.set_lick_threshold_button,       2, 0, 1, 1)
        self.layout.addWidget(self.lick_threshold_spinner,          2, 1, 1, 1)

        self.layout.addWidget(self.session_list_widget,             1, 2, 2, 1)


        # Plot First Item
        self.lick_threshold_line = pyqtgraph.InfiniteLine()
        self.lick_threshold_line.setAngle(0)
        self.lick_threshold_line.setValue(self.current_lick_threshold)
        self.lick_trace_curve = pyqtgraph.PlotCurveItem()
        self.lick_display_view.addItem(self.lick_threshold_line)
        self.lick_display_view.addItem(self.lick_trace_curve)


        self.show()

    def load_all_data(self):

        print("Loading Lick Data, Please Wait: ")
        # Iterate Through All Sessions
        number_of_sessions = len(self.session_directory_list)
        for session_index in tqdm(range(number_of_sessions)):

            # Get Current Session
            current_session = self.session_directory_list[session_index]

            # Get Session Name
            session_name = current_session.split('/')[-1]
            self.session_list_widget.addItem(session_name)

            # Load Lick Trace
            ai_data = Behaviour_Utils.load_ai_recorder_file(current_session)
            lick_trace = ai_data[self.channel_dictionary["Lick"]]
            self.lick_trace_list.append(lick_trace)

            #plt.plot(lick_trace)
            #plt.show()

            # See If We Already Have a Lick Threshold Set
            if os.path.exists(os.path.join(current_session, "Lick_Baseline.npy")):
                lick_threshold = np.load(os.path.join(current_session, "Lick_Baseline.npy"))
                self.lick_threshold_list.append(lick_threshold)
                self.threshold_status_list.append(1)
                already_set = True
            else:
                self.lick_threshold_list.append(self.default_lick_threshold)
                self.threshold_status_list.append(0)
                already_set = False

            # Update List Widget
            if already_set == True:
                self.session_list_widget.item(session_index).setBackground(QColor("#00c957"))
            else:
                self.session_list_widget.item(session_index).setBackground(QColor("#fc0303"))


    def change_lick_threshold(self):
        self.current_lick_threshold = self.lick_threshold_spinner.value()
        self.lick_threshold_line.setValue(self.current_lick_threshold)

    def set_lick_threshold(self):

         # Get Output Path
        file_save_directory = os.path.join(self.session_directory_list[self.current_session_index], "Lick_Baseline.npy")

        # Save File
        np.save(file_save_directory, self.current_lick_threshold)

        self.lick_threshold_list[self.current_session_index] = self.current_lick_threshold
        self.session_list_widget.item(self.current_session_index).setBackground(QColor("#00c957"))


    def set_session(self):

        self.current_session_index = int(self.session_list_widget.currentRow())
        print("current session index")

        # Plot Lick Trace
        self.lick_trace_curve.setData(self.lick_trace_list[self.current_session_index])

        # Set Current Threshold
        self.current_lick_threshold = self.lick_threshold_list[self.current_session_index]
        self.lick_threshold_spinner.setValue(self.current_lick_threshold)
        self.lick_threshold_line.setValue(self.current_lick_threshold)




def zero_lick_traces(session_directory):

    app = QApplication(sys.argv)

    window = lick_threshold_window(session_directory)
    window.show()

    app.exec_()

session_list = [

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_13_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_15_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_17_Switching_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_31_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_02_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_04_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_08_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_10_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_12_Transition_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_20_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_22_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_24_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_14_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_16_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_18_Transition_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_17_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_19_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_23_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_30_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_06_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_08_Transition_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_15_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_17_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_19_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_22_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_24_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_26_Transition_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_14_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_20_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_26_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_05_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_08_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_10_Transition_Imaging",

        r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_28_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_05_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_09_Switching_Imaging",

        r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_29_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_05_Switching_Imaging",

        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_02_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_04_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_06_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_02_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_08_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_10_Transition_Imaging",

        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_26_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_28_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_02_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_23_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_31_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_04_02_Transition_Imaging",

        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_21_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_23_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_11_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_13_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_15_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_17_Transition_Imaging",

        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_14_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_20_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_22_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_29_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_03_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_05_Transition_Imaging",
    ]

zero_lick_traces(session_list)