import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import tables
import sys

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions


def ResampleLinear1D(original, targetLen):

    original = np.array(original, dtype=np.float)
    index_arr = np.linspace(0, len(original)-1, num=targetLen, dtype=np.float)
    index_floor = np.array(index_arr, dtype=np.int) #Round down
    index_ceil = index_floor + 1
    index_rem = index_arr - index_floor #Remain

    val1 = original[index_floor]
    val2 = original[index_ceil % len(original)]
    interp = val1 * (1.0-index_rem) + val2 * index_rem
    assert(len(interp) == targetLen)
    return interp


def downsample_running_trace(base_directory, sanity_check):
    print(base_directory)

    # Load Frame Times
    frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    frame_times = Widefield_General_Functions.invert_dictionary(frame_times)

    # Load AI Recorder File
    ai_filename = Widefield_General_Functions.get_ai_filename(base_directory)
    ai_data = Widefield_General_Functions.load_ai_recorder_file(base_directory + "/" + ai_filename)
    running_trace = ai_data[8]

    # Load Delta F Matrix
    delta_f_matrix_filepath = os.path.join(base_directory, "Delta_F.h5")
    delta_f_matrix_container = tables.open_file(delta_f_matrix_filepath, mode='r')
    delta_f_matrix = delta_f_matrix_container.root['Data']

    # Get Data Structure
    number_of_timepoints = np.shape(delta_f_matrix)[0]
    imaging_start = frame_times[0]
    imaging_stop = frame_times[number_of_timepoints - 1]

    # Downsample Running Traces
    imaging_running_trace = running_trace[imaging_start:imaging_stop]
    downsampled_running_trace = ResampleLinear1D(imaging_running_trace, number_of_timepoints)

    # Check Movement Controls Directory Exists
    movement_controls_directory = os.path.join(base_directory, "Movement_Controls")
    if not os.path.exists(movement_controls_directory):
        os.mkdir(movement_controls_directory)

    # Save Downsampled Running Trace
    np.save(os.path.join(movement_controls_directory, "Downsampled_Running_Trace.npy"), downsampled_running_trace)

    # Sanity Check
    if sanity_check == True:
        figure_1 = plt.figure()
        real_axis = figure_1.add_subplot(2, 1, 1)
        down_axis = figure_1.add_subplot(2, 1, 2)

        real_stop = int(len(imaging_running_trace)/100)
        down_stop = int(len(downsampled_running_trace)/100)

        real_axis.plot(imaging_running_trace[0:real_stop])
        down_axis.plot(downsampled_running_trace[0:down_stop])

        real_axis.set_title("Real Running Trace")
        down_axis.set_title("Downsampled Running Trace")
        plt.show()

