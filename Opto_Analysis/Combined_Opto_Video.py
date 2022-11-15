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


def create_axis_list(figure_1, gridspec):

    # Create Axis
    mouse_1_roi_1_intensity_1_axis = figure_1.add_subplot(gridspec[0, 0])
    mouse_1_roi_1_intensity_2_axis = figure_1.add_subplot(gridspec[0, 1])
    mouse_1_roi_1_intensity_3_axis = figure_1.add_subplot(gridspec[0, 2])
    mouse_1_roi_2_intensity_1_axis = figure_1.add_subplot(gridspec[1, 0])
    mouse_1_roi_2_intensity_2_axis = figure_1.add_subplot(gridspec[1, 1])
    mouse_1_roi_2_intensity_3_axis = figure_1.add_subplot(gridspec[1, 2])
    mouse_1_roi_3_intensity_1_axis = figure_1.add_subplot(gridspec[2, 0])
    mouse_1_roi_3_intensity_2_axis = figure_1.add_subplot(gridspec[2, 1])
    mouse_1_roi_3_intensity_3_axis = figure_1.add_subplot(gridspec[2, 2])

    mouse_2_roi_1_intensity_1_axis = figure_1.add_subplot(gridspec[0, 4])
    mouse_2_roi_1_intensity_2_axis = figure_1.add_subplot(gridspec[0, 5])
    mouse_2_roi_1_intensity_3_axis = figure_1.add_subplot(gridspec[0, 6])
    mouse_2_roi_2_intensity_1_axis = figure_1.add_subplot(gridspec[1, 4])
    mouse_2_roi_2_intensity_2_axis = figure_1.add_subplot(gridspec[1, 5])
    mouse_2_roi_2_intensity_3_axis = figure_1.add_subplot(gridspec[1, 6])
    mouse_2_roi_3_intensity_1_axis = figure_1.add_subplot(gridspec[2, 4])
    mouse_2_roi_3_intensity_2_axis = figure_1.add_subplot(gridspec[2, 5])
    mouse_2_roi_3_intensity_3_axis = figure_1.add_subplot(gridspec[2, 6])

    mouse_3_roi_1_intensity_1_axis = figure_1.add_subplot(gridspec[0, 8])
    mouse_3_roi_1_intensity_2_axis = figure_1.add_subplot(gridspec[0, 9])
    mouse_3_roi_1_intensity_3_axis = figure_1.add_subplot(gridspec[0, 10])
    mouse_3_roi_2_intensity_1_axis = figure_1.add_subplot(gridspec[1, 8])
    mouse_3_roi_2_intensity_2_axis = figure_1.add_subplot(gridspec[1, 9])
    mouse_3_roi_2_intensity_3_axis = figure_1.add_subplot(gridspec[1, 10])
    mouse_3_roi_3_intensity_1_axis = figure_1.add_subplot(gridspec[2, 8])
    mouse_3_roi_3_intensity_2_axis = figure_1.add_subplot(gridspec[2, 9])
    mouse_3_roi_3_intensity_3_axis = figure_1.add_subplot(gridspec[2, 10])

    axis_list = [

        [mouse_1_roi_1_intensity_1_axis,
        mouse_1_roi_1_intensity_2_axis,
        mouse_1_roi_1_intensity_3_axis,
        mouse_1_roi_2_intensity_1_axis,
        mouse_1_roi_2_intensity_2_axis,
        mouse_1_roi_2_intensity_3_axis,
        mouse_1_roi_3_intensity_1_axis,
        mouse_1_roi_3_intensity_2_axis,
        mouse_1_roi_3_intensity_3_axis],

        [mouse_2_roi_1_intensity_1_axis,
        mouse_2_roi_1_intensity_2_axis,
        mouse_2_roi_1_intensity_3_axis,
        mouse_2_roi_2_intensity_1_axis,
        mouse_2_roi_2_intensity_2_axis,
        mouse_2_roi_2_intensity_3_axis,
        mouse_2_roi_3_intensity_1_axis,
        mouse_2_roi_3_intensity_2_axis,
        mouse_2_roi_3_intensity_3_axis],

        [mouse_3_roi_1_intensity_1_axis,
        mouse_3_roi_1_intensity_2_axis,
        mouse_3_roi_1_intensity_3_axis,
        mouse_3_roi_2_intensity_1_axis,
        mouse_3_roi_2_intensity_2_axis,
        mouse_3_roi_2_intensity_3_axis,
        mouse_3_roi_3_intensity_1_axis,
        mouse_3_roi_3_intensity_2_axis,
        mouse_3_roi_3_intensity_3_axis]
    ]

    return axis_list


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

# Setup File Structure
mouse_1_file_location = r"/media/matthew/External_Harddrive_1/Opto_Test/KGCA7.1M/2021_01_24_Opto_Test_Range"
mouse_2_file_location = r"/media/matthew/External_Harddrive_1/Opto_Test/KGCA7.1B/2021_01_27_Opto_Test_Range"
mouse_3_file_location = r"/media/matthew/External_Harddrive_1/Opto_Test/KGCA7.1F/KGCA7.1F_2021_01_21_Opto_Test_Range"
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


# Get ROI Edges
roi_mask_edge_list = []
dilated_roi_mask_edge_list = []

for mouse_index in range(number_of_mice):

    # Load ROI Masks
    base_directory = mouse_file_list[mouse_index]
    stim_log_file = get_stim_log_file(base_directory)
    stim_log = loadmat(os.path.join(base_directory, stim_log_file))
    stim_log = stim_log['opto_session_data']
    roi_masks = stim_log[1][0][0]

    # Load Alignment Transformation
    transformation_dictionary = np.load(os.path.join(base_directory, "Transformation_Dictionary.npy"), allow_pickle=True)[()]

    mouse_roi_mask_edge_list = []
    mouse_dilated_roi_mask_edge_list = []

    for roi_index in range(number_of_stimuli):

        roi_mask = roi_masks[roi_index]

        roi_mask = np.flip(roi_mask, axis=0)

        roi_mask = transform_image(transformation_dictionary, roi_mask)

        edges = cv2.Canny(roi_mask, 0.5, 1)

        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel=kernel, iterations=2)

        mouse_roi_mask_edge_list.append(edges)
        mouse_dilated_roi_mask_edge_list.append(dilated_edges)

    roi_mask_edge_list.append(mouse_dilated_roi_mask_edge_list)
    dilated_roi_mask_edge_list.append(mouse_dilated_roi_mask_edge_list)



# Create Figure
number_of_columns = 11
number_of_rows = 3
figure_1 = plt.figure(constrained_layout=False, figsize=(20,5))
gridspec = figure_1.add_gridspec(nrows=number_of_rows, ncols=number_of_columns)


# Get Number Of Timepoints
number_of_timepoints = np.shape(data_list[0][0])[0]
print("Number of timepoints", number_of_timepoints)

# Set Colours
colourmap = cm.get_cmap('jet')
black_rgba_colour = [0, 0, 0, 1]
white_rgba_colour = [1, 1, 1, 1]

# Create Save Directory
save_directory = "/home/matthew/Pictures/opto_test_batch_3"


plt.ion()
for timepoint_index in range(number_of_timepoints):

    # Create Axis List
    axis_list = create_axis_list(figure_1, gridspec)

    for mouse_index in range(number_of_mice):
        for stimuli_index in range(number_of_stimuli):

            # Get Brain Activity
            brain_activity = data_list[mouse_index][stimuli_index][timepoint_index]
            print("Activity shape", np.shape(brain_activity))

            # Create Images
            brain_image = Widefield_General_Functions.create_image_from_data(brain_activity, mask_data_list[mouse_index][0], mask_data_list[mouse_index][1], mask_data_list[mouse_index][2])

            # Colour Brain Image
            brain_image = colourmap(brain_image)

            # Get ROI Edges
            roi_edges = roi_mask_edge_list[mouse_index][stimuli_index]
            dilated_roi_edges = dilated_roi_mask_edge_list[mouse_index][stimuli_index]

            # Get Edges Indicies
            edge_indicies = np.nonzero(roi_edges)
            dilated_edge_indicies = np.nonzero(dilated_roi_edges)

            # Set Edge Colours
            brain_image[dilated_edge_indicies] = white_rgba_colour
            brain_image[edge_indicies] = black_rgba_colour

            # Draw Images
            axis_list[mouse_index][stimuli_index].imshow(brain_image)

            # Remove Axis
            axis_list[mouse_index][stimuli_index].set_yticks(())
            axis_list[mouse_index][stimuli_index].set_xticks(())
            axis_list[mouse_index][stimuli_index].set_xticklabels(())
            axis_list[mouse_index][stimuli_index].set_yticklabels(())


    # Set Titles
    axis_list[0][1].text(0, -200, "Control Mouse", fontsize=12)
    axis_list[1][1].text(0, -200, "Opsin Mouse 1", fontsize=12)
    axis_list[2][1].text(0, -200, "Opsin Mouse 2", fontsize=12)


    # Add ROI Intensities
    for mouse_index in range(number_of_mice):
        axis_list[mouse_index][0].set_title("Intensity: 20%")
        axis_list[mouse_index][1].set_title("Intensity: 30%")
        axis_list[mouse_index][2].set_title("Intensity: 40%")


    # Add ROI Labels
    axis_list[0][0].set_ylabel("Visual Cortex ROI", rotation='horizontal')
    axis_list[0][3].set_ylabel("Somatosensory Cortex ROI", rotation='horizontal')
    axis_list[0][6].set_ylabel("Motor Cortex ROI", rotation='horizontal')
    axis_list[0][0].yaxis.set_label_coords(-1, 0.5)
    axis_list[0][3].yaxis.set_label_coords(-1, 0.5)
    axis_list[0][6].yaxis.set_label_coords(-1, 0.5)


    # Add Time
    axis_list[0][0].text(-1200, -150, str(((timepoint_index - 100) * 36)) + " ms", fontsize = 12)

    if timepoint_index > 100 and timepoint_index < 173:
        axis_list[0][0].text(-600, -150, "Light ON", fontsize=12, color='r')
    else:
        axis_list[0][0].text(-600, -150, "Light off", fontsize=12, color='gray')

    plt.draw()
    plt.savefig(os.path.join(save_directory, str(timepoint_index).zfill(3) + ".svg"))
    plt.clf()


