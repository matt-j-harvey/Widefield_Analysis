import os
from datetime import datetime

import Position_Mask
import Get_Max_Projection
import Motion_Correction

import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *



def get_file_names(base_directory):

    file_list = os.listdir(base_directory)
    blue_file = None
    violet_file = None

    for file in file_list:
        if "Blue_Data" in file:
            blue_file = file
        elif "Violet_Data" in file:
            violet_file = file

    return blue_file, violet_file


def check_led_colours(base_directory):

    blue_file_name, violet_file_name = get_file_names(base_directory)

    # Load Delta F File
    blue_filepath = os.path.join(base_directory, blue_file_name)
    violet_filepath = os.path.join(base_directory, violet_file_name)

    blue_data_container = h5py.File(blue_filepath, 'r')
    violet_data_container = h5py.File(violet_filepath, 'r')

    blue_array = blue_data_container["Data"]
    violet_array = violet_data_container["Data"]

    figure_1 = plt.figure()
    axes_1 = figure_1.subplots(1, 2)

    blue_image = blue_array[:, 0]
    blue_image = np.reshape(blue_image, (600,608))
    axes_1[0].set_title("Blue?")
    axes_1[0].imshow(blue_image)

    violet_image = violet_array[:, 0]
    violet_image = np.reshape(violet_image, (600,608))
    axes_1[1].set_title("Violet?")
    axes_1[1].imshow(violet_image)
    plt.show()



def get_output_directory(base_directory, output_stem):

    split_base_directory = base_directory.split("/")

    # Check Mouse Directory
    mouse_directory = os.path.join(output_stem, split_base_directory[-2])
    if not os.path.exists(mouse_directory):
        os.mkdir(mouse_directory)

    # Check Session Directory
    session_directory = os.path.join(mouse_directory, split_base_directory[-1])
    if not os.path.exists(session_directory):
        os.mkdir(session_directory)

    return session_directory

def get_motion_corrected_data_filename(base_directory):

    file_list = os.listdir(base_directory)
    for file in file_list:
        if "Motion_Corrected_Mask_Data" in file:
            return file




def run_motion_correction_pipeline(session_list):

    """
    1.) Get Max Projection
    2.) Assign Generous Mask
    3.) Motion Correction
    """


    # Create QApplication
    app = QApplication(sys.argv)

    # Get Number Of Session
    number_of_sessions = len(session_list)

    """
    # Check LED Colors
    for base_directory in session_list:
        check_led_colours(base_directory)


    # Get Max Projections
    for session_index in range(number_of_sessions):
        base_directory = session_list[session_index]
        Get_Max_Projection.check_max_projection(base_directory, base_directory)

    # Assign Masks
    window = Position_Mask.masking_window(session_list, session_list)
    window.show()
    app.exec_()
    """
    # Process Data
    for session_index in range(number_of_sessions):
        base_directory = session_list[session_index]
        print("Session ", session_index, " of ", number_of_sessions, base_directory)

        # Perform Motion Correction
        print("Performing Motion Correction", datetime.now())
        Motion_Correction.perform_motion_correction(base_directory, base_directory)





session_list = [
                #r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_02_Spontaneous",
                r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_01_25_Spontaneous",
                r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_01_25_Spontaneous",
                r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_04_21_Spontaneous",
                r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_02_Spontaneous",
                ]

run_motion_correction_pipeline(session_list)

