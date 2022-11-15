import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import os
import numpy as np
from scipy.io import loadmat
from scipy import ndimage
from skimage.measure import find_contours
import cv2
import sys
sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions



def get_stim_log_file(base_directory):

    file_list = os.listdir(base_directory)

    for file in file_list:
        if file[0:10] == 'opto_stim_':
            return file



def transform_image(transformation_details, image):

    # Load Variables From Dictionary
    rotation = transformation_details['rotation']
    x_shift = transformation_details['x_shift']
    y_shift = transformation_details['y_shift']

    # Rotate Mask
    image = ndimage.rotate(image, rotation, reshape=False)

    # Translate
    image = np.roll(a=image, axis=0, shift=y_shift)
    image = np.roll(a=image, axis=1, shift=x_shift)

    # Re-Binarise
    #image = np.where(image > 0.1, 1, 0)
    #image = np.ndarray.astype(image, int)

    return image


def get_roi_trace(stimuli_index, number_of_timepoints, stimuli_means, stimuli_stds, indicies, image_height, image_width, roi_indicies_list):

    roi_mean_list = []
    roi_std_list = []

    for timepoint_index in range(number_of_timepoints):

        # Get Brain Activity
        mean_brain_activity = stimuli_means[stimuli_index][timepoint_index]
        std_brain_activity = stimuli_stds[stimuli_index][timepoint_index]

        # Create Images
        mean_brain_image = Widefield_General_Functions.create_image_from_data(mean_brain_activity, indicies, image_height, image_width)
        std_brain_image = Widefield_General_Functions.create_image_from_data(std_brain_activity, indicies, image_height, image_width)

        # Get ROI indicies
        roi_indicies = roi_indicies_list[stimuli_index]

        # Get ROI Activity
        roi_mean = mean_brain_image[roi_indicies]
        roi_std = std_brain_image[roi_indicies]

        roi_mean = np.mean(roi_mean)
        roi_std = np.mean(roi_std)

        roi_mean_list.append(roi_mean)
        roi_std_list.append(roi_std)

    return roi_mean_list, roi_std_list


def get_mean_stimuli_responses(base_directory, number_of_stimuli):

    average_responses = []
    for stimuli_index in range(number_of_stimuli):
        data_file = os.path.join(base_directory, "Stimuli_" + str(stimuli_index + 1), "mean_response.npy")
        data = np.load(data_file)
        average_responses.append(data)

    return average_responses


def get_stimuli_responses(base_directory, number_of_stimuli):

    stimuli_means = []
    stimuli_stds = []
    for stimuli_index in range(number_of_stimuli):
        data_file = os.path.join(base_directory, "Stimuli_" + str(stimuli_index + 1), "Activity_Tensor.npy")
        data = np.load(data_file)

        # Get Mean
        response_mean = np.mean(data, axis=0)
        response_std = np.std(data, axis=0)

        # Add To List
        stimuli_means.append(response_mean)
        stimuli_stds.append(response_std)

    # Return Lists
    return stimuli_means, stimuli_stds




def get_roi_indicies(roi_masks, number_of_stimuli):

    roi_indicies_list = []
    for roi_index in range(number_of_stimuli):
        roi_mask = roi_masks[roi_index]

        roi_mask = np.flip(roi_mask, axis=0)

        roi_mask = np.where(roi_mask > 0.5, 1, 0)

        indicies = np.nonzero(roi_mask)

        roi_indicies_list.append(indicies)

    return roi_indicies_list


def plot_roi_traces(base_directory):

    # Get Mask Details
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    # Load Stim Log
    stim_log_file = get_stim_log_file(base_directory)
    stim_log = loadmat(os.path.join(base_directory, stim_log_file))
    stim_log = stim_log['opto_session_data']

    # Load Number Of Stimuli
    stimuli = stim_log[0][0][0]
    number_of_stimuli = len(np.unique(stimuli))
    print("Number of stimuli", number_of_stimuli)

    # Load ROI Masks
    roi_masks = stim_log[1][0][0]

    # Get ROI Indicies
    roi_indicies_list = get_roi_indicies(roi_masks, number_of_stimuli)

    # Load Average Responses
    stimuli_means, stimuli_stds = get_stimuli_responses(base_directory, number_of_stimuli)
    print("Average Response", np.shape(stimuli_means))
    print("Stimuli STDS", np.shape(stimuli_stds))
    number_of_timepoints = np.shape(stimuli_means)[1]


    # Get ROI Traces
    roi_mean_list = []
    roi_std_list = []
    for stimuli_index in range(number_of_stimuli):
        roi_mean, roi_std = get_roi_trace(stimuli_index, number_of_timepoints, stimuli_means, stimuli_stds, indicies, image_height, image_width, roi_indicies_list)
        roi_mean_list.append(roi_mean)
        roi_std_list.append(roi_std)

    # Plot In Figure
    vmin = np.min(roi_mean_list)
    vmax = np.max(roi_mean_list) * 1.1
    figure_1 = plt.figure()
    rows = 3
    columns = 3

    x_values = list(range(number_of_timepoints))
    for stimuli_index in range(number_of_stimuli):
        trace = roi_mean_list[stimuli_index]
        std_dev = roi_std_list[stimuli_index]

        lower_bound = np.subtract(trace, std_dev)
        upper_bound = np.add(trace, std_dev)

        axis_1 = figure_1.add_subplot(rows, columns, stimuli_index + 1)
        axis_1.plot(x_values, trace)
        axis_1.fill_between(x=x_values, y1=lower_bound, y2=upper_bound, alpha=0.2)

        axis_1.set_ylim([vmin, vmax])
    plt.show()





base_directory = r"/media/matthew/External_Harddrive_1/Opto_Test/KGCA7.1F/KGCA7.1F_2021_01_21_Opto_Test_Range"
base_directory = r"/media/matthew/External_Harddrive_1/Opto_Test/KVIP25.5H/2022_07_27_Opto_Test_Grid"
#base_directory = r"/media/matthew/External_Harddrive_1/Opto_Test/Projector_Calib/2022_07_26_Calibration"
#base_directory = r"/media/matthew/External_Harddrive_1/Opto_Test/KVIP25.5H/2022_07_26_Opto_Test_No_Filter"
plot_roi_traces(base_directory)
