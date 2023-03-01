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

import Widefield_General_Functions
import Behaviour_Utils

def load_ai_recorder_file(ai_recorder_file_location):
    table = tables.open_file(ai_recorder_file_location, mode='r')
    data = table.root.Data

    number_of_seconds = np.shape(data)[0]
    number_of_channels = np.shape(data)[1]
    sampling_rate = np.shape(data)[2]

    data_matrix = np.zeros((number_of_channels, number_of_seconds * sampling_rate))

    for second in range(number_of_seconds):
        data_window = data[second]
        start_point = second * sampling_rate

        for channel in range(number_of_channels):
            data_matrix[channel, start_point:start_point + sampling_rate] = data_window[channel]

    data_matrix = np.clip(data_matrix, a_min=0, a_max=None)

    table.close()
    return data_matrix



class lick_threshold_window(QWidget):

    def __init__(self, session_directory_list, parent=None):
        super(lick_threshold_window, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("Set Lick Thresholds")
        self.setGeometry(0, 0, 1500, 500)

        # Create Variable Holders
        self.session_directory_list = session_directory_list
        self.channel_dictionary = Widefield_General_Functions.create_stimuli_dictionary()
        self.default_lick_threshold = 0.3
        self.current_lick_threshold = 0.3
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
        self.set_lick_threshold_button = QPushButton("Set Lick Threshold")
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
            ai_file_name = Widefield_General_Functions.get_ai_filename(current_session)
            ai_data = load_ai_recorder_file(current_session + ai_file_name)
            lick_trace = ai_data[self.channel_dictionary["Lick"]]
            self.lick_trace_list.append(lick_trace)

            # See If We Already Have a Lick Threshold Set
            if os.path.exists(os.path.join(current_session, "Lick_Threshold.npy")):
                lick_threshold = np.load(os.path.join(current_session, "Lick_Threshold.npy"))
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
        file_save_directory = os.path.join(self.session_directory_list[self.current_session_index], "Lick_Threshold.npy")

        # Save File
        np.save(file_save_directory, self.current_lick_threshold)

        self.lick_threshold_list[self.current_session_index] = 1
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




def set_lick_thresholds(session_directory):

    app = QApplication(sys.argv)

    window = lick_threshold_window(session_directory)
    window.show()

    app.exec_()


# Load Sessions
mouse_list = ["NRXN78.1A", "NRXN78.1D", "NXAK4.1B", "NXAK7.1B", "NXAK14.1A", "NXAK22.1A"]
session_type = "Switching"
session_list = []
for mouse_name in mouse_list:
    session_list = session_list + Behaviour_Utils.load_mouse_sessions(mouse_name, session_type)

# This Is The Location of My Experimental Logbook
logbook_file_location = r"/home/matthew/Documents/Experiment_Logbook.ods"

#  Read Logbook As A Dataframe
logbook_dataframe = pd.read_excel(logbook_file_location, engine="odf")

# Extract Session List
#session_list = list(logbook_dataframe["Filepath"].values)

#session_list = [r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Switching_Opto/KPVB17.1E/2022_11_21_Opto_Switching"]
#session_list = ["/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Switching_Opto/KPGC2.2G/2022_12_02_Switching_Opto"]
session_list = [r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Switching_Opto/KPGC2.2G/2022_12_08_Switching_Opto"]

set_lick_thresholds(session_list)