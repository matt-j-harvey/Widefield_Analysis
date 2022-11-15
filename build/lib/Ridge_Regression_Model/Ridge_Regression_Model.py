import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import tables
from tqdm import tqdm

import Regression_Utils

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def create_event_kernel_from_event_list(event_list, number_of_widefield_frames, preceeding_window=-14, following_window=28):

    kernel_size = following_window - preceeding_window
    design_matrix = np.zeros((number_of_widefield_frames, kernel_size))

    for timepoint_index in range(number_of_widefield_frames):

        if event_list[timepoint_index] == 1:

            # Get Start and Stop Times Of Kernel
            start_time = timepoint_index + preceeding_window
            stop_time = timepoint_index + following_window

            # Ensure Start and Stop Times Dont Fall Below Zero Or Above Number Of Frames
            start_time = np.max([0, start_time])
            stop_time = np.min([number_of_widefield_frames-1, stop_time])

            # Fill In Design Matrix
            number_of_regressor_timepoints = stop_time - start_time
            for regressor_index in range(number_of_regressor_timepoints):
                design_matrix[start_time + regressor_index, regressor_index] = 1

    return design_matrix



def fit_ridge_model(base_directory, early_cutoff=3000):

    # Load Delta F Matrix
    delta_f_file = os.path.join(base_directory, "Downsampled_Delta_F.h5")
    delta_f_file_container = tables.open_file(delta_f_file, "r")
    delta_f_matrix = delta_f_file_container.root.Data
    number_of_widefield_frames, number_of_pixels = np.shape(delta_f_matrix)

    # Load Downsampled AI
    downsampled_ai_file = os.path.join(base_directory, "Downsampled_AI_Matrix_Framewise.npy")
    downsampled_ai_matrix = np.load(downsampled_ai_file)
    number_of_widefield_frames = np.shape(downsampled_ai_matrix)[1]
    print("Number of widefield frames", number_of_widefield_frames)

    # Load Blink and Eye Movement Event Lists
    eye_movement_event_list = np.load(os.path.join(base_directory, "Eyecam_Analysis", "Matched_Eye_Movement_Events.npy"))
    blink_event_list = np.load(os.path.join(base_directory, "Eyecam_Analysis", "Matched_Blink_Events.npy"))

    # Create Regressor Kernels
    eye_movement_event_kernel = create_event_kernel_from_event_list(eye_movement_event_list, number_of_widefield_frames)
    blink_event_kernel = create_event_kernel_from_event_list(blink_event_list, number_of_widefield_frames)

    # Load Whisker Pad Motion
    whisker_pad_motion_components = np.load(os.path.join(base_directory, "Mousecam_Analysis", "matched_whisker_data.npy"))
    print("Whisker Pad Motion Components", np.shape(whisker_pad_motion_components))

    # Create Stimuli Dictionary
    stimuli_dictionary = Regression_Utils.create_stimuli_dictionary()

    # Extract Lick and Running Traces
    lick_trace = downsampled_ai_matrix[stimuli_dictionary["Lick"]]
    running_trace = downsampled_ai_matrix[stimuli_dictionary["Running"]]

    print("Eye Movement Event Kernel", np.shape(eye_movement_event_kernel))
    print("Blink Event Kernel", np.shape(blink_event_kernel))

    # Create Design Matrix
    design_matrix = np.hstack([
        #lick_trace,
        #running_trace,
        eye_movement_event_kernel,
        blink_event_kernel,
        whisker_pad_motion_components,
    ])

    print("Design Matrix Shape", np.shape(design_matrix))

    #design_matrix = np.transpose(design_matrix)
    design_matrix = design_matrix[early_cutoff:]

    # Iterate Through Pixels
    model = Ridge()

    # Get Chunk Structure
    chunk_size = 10000
    number_of_frames, number_of_pixels = np.shape(delta_f_matrix)
    number_of_chunks, chunk_sizes, chunk_start_list, chunk_stop_list = Regression_Utils.get_chunk_structure(chunk_size, number_of_pixels)

    # Fit Model For Each Chunk
    regression_coefs_list = []
    regression_intercepts_list = []
    for chunk_index in tqdm(range(number_of_chunks)):

        # Get Chunk Data
        chunk_start = chunk_start_list[chunk_index]
        chunk_stop = chunk_stop_list[chunk_index]
        chunk_data = delta_f_matrix[early_cutoff:, chunk_start:chunk_stop]
        chunk_data = np.nan_to_num(chunk_data)

        # Fit Model
        model.fit(y=chunk_data, X=design_matrix)

        # Get Coefs
        model_coefs = model.coef_
        model_intercept = model.intercept_

        # Add To Lists
        for coef in model_coefs:
            regression_coefs_list.append(coef)

        for intercept in model_intercept:
            regression_intercepts_list.append(intercept)

    # Save These
    save_directory = os.path.join(base_directory, "Regression_Coefs")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    regression_dict = {
        "Coefs":regression_coefs_list,
        "Intercepts":regression_intercepts_list
    }

    np.save(os.path.join(save_directory, "Regression_Dicionary.npy"), regression_dict)

    # Close Delta F File
    delta_f_file_container.close()


base_directory = r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_17_Switching_Imaging"
fit_ridge_model(base_directory)

"""
# Fit Models For Controls
mouse_list = ["NXAK14.1A", "NRXN78.1D", "NXAK4.1B", "NXAK7.1B", "NXAK22.1A"]
session_type = "Transition"
session_list = []
for mouse_name in mouse_list:
    session_list = session_list + Regression_Utils.load_mouse_sessions(mouse_name, session_type)

for session in session_list:
    fit_ridge_model(session)


# Fit Model For Mutants
mouse_list = ["NRXN71.2A", "NXAK4.1A", "NXAK10.1A",  "NXAK16.1B", "NXAK24.1C", "NXAK20.1B"]
session_type = "Transition"
session_list = []
for mouse_name in mouse_list:
    session_list = session_list + Regression_Utils.load_mouse_sessions(mouse_name, session_type)

for session in session_list:
    fit_ridge_model(session)
"""