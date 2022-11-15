import matplotlib.pyplot as plt
from matplotlib import cm
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


def get_roi_trace(mouse_index, stimuli_index, number_of_timepoints, data_list, mask_data_list, roi_indicies_list):

    roi_activity_trace = []
    for timepoint_index in range(number_of_timepoints):
        # Get Brain Activity
        brain_activity = data_list[mouse_index][stimuli_index][timepoint_index]

        # Create Images
        brain_image = Widefield_General_Functions.create_image_from_data(brain_activity, mask_data_list[mouse_index][0], mask_data_list[mouse_index][1], mask_data_list[mouse_index][2])

        # Get ROI indicies
        roi_indicies = roi_indicies_list[mouse_index][stimuli_index]

        # Get ROI Activity
        roi_activity = brain_image[roi_indicies]
        roi_activity = np.mean(roi_activity)

        # Append To List
        roi_activity_trace.append((roi_activity))

    return roi_activity_trace



# Setup File Structure
mouse_1_file_location = r"/media/matthew/29D46574463D2856/Opto_Test/KGCA7.1M/2021_01_24_Opto_Test_Range"
mouse_2_file_location = r"/media/matthew/29D46574463D2856/Opto_Test/KGCA7.1B/2021_01_27_Opto_Test_Range"
mouse_3_file_location = r"/media/matthew/29D46574463D2856/Opto_Test/KGCA7.1F/KGCA7.1F_2021_01_21_Opto_Test_Range"
mouse_file_list = [mouse_1_file_location, mouse_2_file_location, mouse_3_file_location]
number_of_mice = len(mouse_file_list)
number_of_stimuli = 9


# Load Data List:
data_list = []
for mouse_index in range(number_of_mice):
    mouse_data = []
    for stimuli_index in range(number_of_stimuli):
        data_file = os.path.join(mouse_file_list[mouse_index], "Stimuli_" + str(stimuli_index + 1), "mean_response.npy")
        data = np.load(data_file)
        mouse_data.append(data)
    data_list.append(mouse_data)


# Load Masks
mask_data_list = []
for mouse_index in range(number_of_mice):

    base_directory = mouse_file_list[mouse_index]

    # Get Mask Details
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    # Add Data To List
    mask_data_list.append([indicies, image_height, image_width])


# Get ROI Indicies
roi_indicies_list = []

for mouse_index in range(number_of_mice):

    # Load ROI Masks
    base_directory = mouse_file_list[mouse_index]
    stim_log_file = get_stim_log_file(base_directory)
    stim_log = loadmat(os.path.join(base_directory, stim_log_file))
    stim_log = stim_log['opto_session_data']
    roi_masks = stim_log[1][0][0]

    # Load Alignment Transformation
    transformation_dictionary = np.load(os.path.join(base_directory, "Transformation_Dictionary.npy"), allow_pickle=True)[()]

    mouse_indicies_list = []
    for roi_index in range(number_of_stimuli):

        roi_mask = roi_masks[roi_index]

        roi_mask = np.flip(roi_mask, axis=0)

        roi_mask = transform_image(transformation_dictionary, roi_mask)

        roi_mask = np.where(roi_mask > 0.5, 1, 0)

        indicies = np.nonzero(roi_mask)

        mouse_indicies_list.append(indicies)
    roi_indicies_list.append(mouse_indicies_list)



# Get Number Of Timepoints
number_of_timepoints = np.shape(data_list[0][0])[0]
print("Number of timepoints", number_of_timepoints)

# Create Save Directory
save_directory = "/home/matthew/Pictures/opto_test_batch_1"


low_colourmap    = cm.get_cmap('Blues')
medium_colourmap = cm.get_cmap('Oranges')
high_colourmap   = cm.get_cmap('Reds')

low_intensity_colour    = medium_colourmap(0.3)
medium_intensity_colour = medium_colourmap(0.7)
high_intensity_colour   = high_colourmap(0.9)


x_values = list(range(-100, 100))
x_values = np.multiply(x_values, 36)

# Create Axis Lis
for mouse_index in range(number_of_mice):

    roi_1_intensity_1_trace = get_roi_trace(mouse_index, 0, number_of_timepoints, data_list, mask_data_list, roi_indicies_list)
    roi_1_intensity_2_trace = get_roi_trace(mouse_index, 1, number_of_timepoints, data_list, mask_data_list, roi_indicies_list)
    roi_1_intensity_3_trace = get_roi_trace(mouse_index, 2, number_of_timepoints, data_list, mask_data_list, roi_indicies_list)

    roi_2_intensity_1_trace = get_roi_trace(mouse_index, 3, number_of_timepoints, data_list, mask_data_list, roi_indicies_list)
    roi_2_intensity_2_trace = get_roi_trace(mouse_index, 4, number_of_timepoints, data_list, mask_data_list, roi_indicies_list)
    roi_2_intensity_3_trace = get_roi_trace(mouse_index, 5, number_of_timepoints, data_list, mask_data_list, roi_indicies_list)

    roi_3_intensity_1_trace = get_roi_trace(mouse_index, 6, number_of_timepoints, data_list, mask_data_list, roi_indicies_list)
    roi_3_intensity_2_trace = get_roi_trace(mouse_index, 7, number_of_timepoints, data_list, mask_data_list, roi_indicies_list)
    roi_3_intensity_3_trace = get_roi_trace(mouse_index, 8, number_of_timepoints, data_list, mask_data_list, roi_indicies_list)


    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)

    axis_1.plot(x_values, roi_1_intensity_1_trace, c=low_intensity_colour)
    axis_1.plot(x_values, roi_2_intensity_1_trace, c=low_intensity_colour)
    axis_1.plot(x_values, roi_3_intensity_1_trace, c=low_intensity_colour)

    axis_1.plot(x_values, roi_1_intensity_2_trace, c=medium_intensity_colour)
    axis_1.plot(x_values, roi_2_intensity_2_trace, c=medium_intensity_colour)
    axis_1.plot(x_values, roi_3_intensity_2_trace, c=medium_intensity_colour)

    axis_1.plot(x_values, roi_1_intensity_3_trace, c=high_intensity_colour)
    axis_1.plot(x_values, roi_2_intensity_3_trace, c=high_intensity_colour)
    axis_1.plot(x_values, roi_3_intensity_3_trace, c=high_intensity_colour)

    axis_1.axvspan(0, 2000, alpha=0.5, color='blue')
    axis_1.set_ylim([0, 1])

    axis_1.set_xlabel("Time (ms)")
    axis_1.set_ylabel("Normalised Delta F")


    plt.show()


