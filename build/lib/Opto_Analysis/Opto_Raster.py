import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.pyplot import GridSpec
import os
import sys
from scipy.io import loadmat
import tables


sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions



def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def get_stim_log_file(base_directory):

    file_list = os.listdir(base_directory)

    for file in file_list:
        if file[0:10] == 'opto_stim_':
            return file


def get_trial_activity(delta_f_matrix, onset, start_window, stop_window, indicies, image_height, image_width):

    trial_start = onset + start_window
    trial_stop = onset + stop_window
    trial_data = delta_f_matrix[trial_start:trial_stop]

    reconstruced_data = []
    for frame in trial_data:
        frame = Widefield_General_Functions.create_image_from_data(frame, indicies, image_height, image_width)
        reconstruced_data.append(frame)


    return reconstruced_data



def plot_full_brain_raster(base_directory):

    # Create Save Directory
    save_directory = os.path.join(base_directory, "Raster_Plots")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Settings
    start_window = -20
    stop_window = 100
    x_values = np.multiply(list(range(start_window, stop_window)), 36)

    # Get Number Of Stimuli
    stim_log_file = get_stim_log_file(base_directory)
    stim_log = loadmat(os.path.join(base_directory, stim_log_file))
    stim_log = stim_log['opto_session_data']
    stimuli = stim_log[0][0][0]
    number_of_stimuli = len(np.unique(stimuli))
    print("Number of stimuli", number_of_stimuli)

    # Get Mask Details
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    # Load Delta F File
    delta_f_file = os.path.join(base_directory, "Delta_F.h5")
    delta_f_file_object = tables.open_file(delta_f_file)
    delta_f_matrix = delta_f_file_object.root.Data

    # Iterate Through Each Stimuli
    figure_1 = plt.figure(figsize=(8,8))
    plt.tight_layout()
    rows = 3
    columns = 3

    extent = [x_values[0], x_values[-1], 0, image_height]
    vmin=0
    vmax= 40000

    for stimuli_index in range(number_of_stimuli):
       print("Stimuli: ", stimuli_index, " of ", number_of_stimuli)

       # Create Empty List To Hold Trial Data
       stimuli_data = []

       # Load onsets
       stimuli_directory = os.path.join(base_directory, "Stimuli_" + str(int(stimuli_index + 1)))
       onset_list = np.load(os.path.join(stimuli_directory, "Stimuli_Onsets.npy"))

       # Get Data For Each Onset
       for onset in onset_list:
           trial_data = get_trial_activity(delta_f_matrix, onset, start_window, stop_window, indicies, image_height, image_width)
           stimuli_data.append(trial_data)

       # Convert TO Array
       stimuli_data = np.array(stimuli_data)

       # Convert To Raster
       stimuli_data = np.mean(stimuli_data, axis=0)
       stimuli_data = np.mean(stimuli_data, axis=2)

        # Plot
       stimuli_axis = figure_1.add_subplot(rows, columns, stimuli_index + 1)
       stimuli_axis.set_title("Stimuli: " + str(stimuli_index))
       stimuli_axis.imshow(np.transpose(stimuli_data), extent=extent, vmin=vmin, vmax=vmax, cmap='jet')
       stimuli_axis.axvline(0, color='k', linestyle='--')
       forceAspect(stimuli_axis)

    # Adjust Plot Spacing
    left = 0.125  # the left side of the subplots of the figure
    right = 0.9  # the right side of the subplots of the figure
    bottom = 0.1  # the bottom of the subplots of the figure
    top = 0.9  # the top of the subplots of the figure
    wspace = 0.2  # the amount of width reserved for blank space between subplots
    hspace = 0.5  # the amount of height reserved for white space between subplots

    figure_1.suptitle("Whole Brain Raster")
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    #plt.show()

    plt.savefig(os.path.join(save_directory, "Whole_Brain_Raster.png"))
    plt.close()


def get_roi_trace(mean_tensor, roi_indicies):

    roi_tensor = []
    for frame in mean_tensor:
        roi_data = frame[roi_indicies]
        roi_tensor.append(roi_data)
    roi_tensor = np.array(roi_tensor)
    return roi_tensor


def plot_roi_raster(base_directory):

    # Settings
    start_window = -20
    stop_window = 100
    x_values = np.multiply(list(range(start_window, stop_window)), 36)

    # Create Save Directory
    save_directory = os.path.join(base_directory, "Raster_Plots")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Get Number Of Stimuli
    stim_log_file = get_stim_log_file(base_directory)
    stim_log = loadmat(os.path.join(base_directory, stim_log_file))
    stim_log = stim_log['opto_session_data']
    stimuli = stim_log[0][0][0]
    number_of_stimuli = len(np.unique(stimuli))
    print("Number of stimuli", number_of_stimuli)

    # Load ROI Masks
    roi_masks = stim_log[1][0][0]

    # Get ROI Indicies
    roi_indicies = []
    for mask in roi_masks:
        mask = np.flip(mask, axis=0)
        indicies = np.nonzero(mask)
        roi_indicies.append(indicies)

    # Get Mask Details
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    # Load Delta F File
    delta_f_file = os.path.join(base_directory, "Delta_F.h5")
    delta_f_file_object = tables.open_file(delta_f_file)
    delta_f_matrix = delta_f_file_object.root.Data

    # Iterate Through Each Stimuli
    figure_1 = plt.figure(figsize=(8,8))
    plt.tight_layout()
    rows = 3
    columns = 3

    extent = [x_values[0], x_values[-1], 0, image_height]
    vmin=0
    vmax= 40000

    for stimuli_index in range(number_of_stimuli):
       print("Stimuli: ", stimuli_index, " of ", number_of_stimuli)

       # Create Empty List To Hold Trial Data
       stimuli_data = []

       # Load onsets
       stimuli_directory = os.path.join(base_directory, "Stimuli_" + str(int(stimuli_index + 1)))
       onset_list = np.load(os.path.join(stimuli_directory, "Stimuli_Onsets.npy"))

       # Get Data For Each Onset
       for onset in onset_list:
           trial_data = get_trial_activity(delta_f_matrix, onset, start_window, stop_window, indicies, image_height, image_width)
           stimuli_data.append(trial_data)

       # Convert TO Array
       stimuli_data = np.array(stimuli_data)

       # Convert To Raster
       stimuli_data = np.mean(stimuli_data, axis=0)

       current_stimuli_roi_indicis = roi_indicies[stimuli_index]
       stimuli_data = get_roi_trace(stimuli_data, current_stimuli_roi_indicis)

        # Plot
       stimuli_axis = figure_1.add_subplot(rows, columns, stimuli_index + 1)
       stimuli_axis.set_title("Stimuli: " + str(stimuli_index))
       stimuli_axis.imshow(np.transpose(stimuli_data), extent=extent, vmin=vmin, vmax=vmax, cmap='jet')
       stimuli_axis.axvline(0, color='k', linestyle='--')
       forceAspect(stimuli_axis)

    # Adjust Plot Spacing
    left = 0.125  # the left side of the subplots of the figure
    right = 0.9  # the right side of the subplots of the figure
    bottom = 0.1  # the bottom of the subplots of the figure
    top = 0.9  # the top of the subplots of the figure
    wspace = 0.2  # the amount of width reserved for blank space between subplots
    hspace = 0.5  # the amount of height reserved for white space between subplots

    figure_1.suptitle("ROI Raster")
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    #plt.show()

    plt.savefig(os.path.join(save_directory, "ROI_Raster.png"))
    plt.close()





base_directory = r"/media/matthew/External_Harddrive_1/Opto_Test/KVIP25.5H/2022_07_27_Opto_Test_Grid"
#base_directory = r"/media/matthew/External_Harddrive_1/Opto_Test/KVIP25.5H/2022_07_26_Opto_Test_No_Filter"
#plot_full_brain_raster(base_directory)
plot_roi_raster(base_directory)