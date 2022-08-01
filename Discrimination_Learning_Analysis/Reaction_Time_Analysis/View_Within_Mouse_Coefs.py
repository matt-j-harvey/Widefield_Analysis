import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import sys
from scipy import ndimage
from skimage.transform import resize
sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")
import Widefield_General_Functions

from matplotlib.pyplot import cm

def load_generous_mask(home_directory):

    # Loads the mask for a video, returns a list of which pixels are included, as well as the original image height and width
    mask = np.load(home_directory + "/Generous_Mask.npy")

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask>0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width

def transform_image(image, variable_dictionary, invert=False):

    # Unpack Dict
    angle = variable_dictionary['rotation']
    x_shift = variable_dictionary['x_shift']
    y_shift = variable_dictionary['y_shift']

    # Inverse
    if invert == True:
        angle = -1 * angle
        x_shift = -1 * x_shift
        y_shift = -1 * y_shift

    transformed_image = np.copy(image)
    transformed_image = ndimage.rotate(transformed_image, angle, reshape=False, prefilter=True)
    transformed_image = np.roll(a=transformed_image, axis=0, shift=y_shift)
    transformed_image = np.roll(a=transformed_image, axis=1, shift=x_shift)

    return transformed_image





def get_condition_average(session_list, model_name):

    # Create Empty Lists To Hold Variables
    group_condition_1_array = []
    group_condition_2_array = []

    # Iterate Through Session
    for base_directory in session_list:
        print("Base Directory", base_directory)
        # Create Empty Lists To Hold Variables
        session_condition_1_array = []
        session_condition_2_array = []

        # Open Regression Dictionary
        regression_dictionary_filepath = os.path.join(base_directory, "Simple_Regression", model_name + "_Regression_Model.npy")
        if not os.path.exists(regression_dictionary_filepath):
            print("No Model For Session:", base_directory)

        else:
            regression_dictionary = np.load(regression_dictionary_filepath, allow_pickle=True)[()]

            # Load Mask
            indicies, image_height, image_width = load_generous_mask(base_directory)

            # Load Alignment Dictionary
            alignment_dictionary = np.load(os.path.join(base_directory, "Cluster_Alignment_Dictionary.npy"), allow_pickle=True)[()]

            # Unpack Regression Dictionary
            coefficient_matrix = regression_dictionary["Regression_Coefficients"]
            print("Coef Matrix", np.shape(coefficient_matrix))
            #coefficient_matrix = regression_dictionary["Coefficients_of_Partial_Determination"]
            #coefficient_matrix = np.moveaxis(coefficient_matrix, (0, 1, 2), (0, 2, 1))
            print("Coef Matrix", np.shape(coefficient_matrix))
            start_window = regression_dictionary["Start_Window"]
            stop_window = regression_dictionary["Stop_Window"]
            trial_length = stop_window - start_window
            condition_1_coefs = coefficient_matrix[0]
            condition_2_coefs = coefficient_matrix[1]
            condition_1_coefs = np.transpose(condition_1_coefs)
            condition_2_coefs = np.transpose(condition_2_coefs)

            condition_1_coefs = np.nan_to_num(condition_1_coefs)
            condition_2_coefs = np.nan_to_num(condition_2_coefs)

            for timepoint_index in range(trial_length):

                # Get Timepoint Coefficients
                condition_1_timepoint_coefs = condition_1_coefs[timepoint_index]
                condition_2_timepoint_coefs = condition_2_coefs[timepoint_index]

                # Reconstruct Image
                condition_1_timepoint_coefs = Widefield_General_Functions.create_image_from_data(condition_1_timepoint_coefs, indicies, image_height, image_width)
                condition_2_timepoint_coefs = Widefield_General_Functions.create_image_from_data(condition_2_timepoint_coefs, indicies, image_height, image_width)

                # Align Image
                condition_1_timepoint_coefs = transform_image(condition_1_timepoint_coefs, alignment_dictionary)
                condition_2_timepoint_coefs = transform_image(condition_2_timepoint_coefs, alignment_dictionary)

                # Smooth Image
                #condition_1_timepoint_coefs = ndimage.gaussian_filter(condition_1_timepoint_coefs, sigma=2)
                #condition_2_timepoint_coefs = ndimage.gaussian_filter(condition_2_timepoint_coefs, sigma=2)

                # Append To List
                session_condition_1_array.append(condition_1_timepoint_coefs)
                session_condition_2_array.append(condition_2_timepoint_coefs)

            group_condition_1_array.append(session_condition_1_array)
            group_condition_2_array.append(session_condition_2_array)

    group_condition_1_array = np.array(group_condition_1_array)
    group_condition_2_array = np.array(group_condition_1_array)

    return group_condition_1_array, group_condition_2_array



def transform_mask_or_atlas(image, variable_dictionary):

    image_height = 600
    image_width = 608

    # Unpack Dictionary
    angle = variable_dictionary['rotation']
    x_shift = variable_dictionary['x_shift']
    y_shift = variable_dictionary['y_shift']
    x_scale = variable_dictionary['x_scale']
    y_scale = variable_dictionary['y_scale']

    # Copy
    transformed_image = np.copy(image)

    # Scale
    original_height, original_width = np.shape(transformed_image)
    new_height = int(original_height * y_scale)
    new_width = int(original_width * x_scale)
    transformed_image = resize(transformed_image, (new_height, new_width), preserve_range=True)
    print("new image height", np.shape(transformed_image))

    # Rotate
    transformed_image = ndimage.rotate(transformed_image, angle, reshape=False, prefilter=True)

    # Insert Into Background
    mask_height, mask_width = np.shape(transformed_image)
    centre_x = 200
    centre_y = 200
    background_array = np.zeros((1000, 1000))
    x_start = centre_x + x_shift
    x_stop = x_start + mask_width

    y_start = centre_y + y_shift
    y_stop = y_start + mask_height

    background_array[y_start:y_stop, x_start:x_stop] = transformed_image

    # Take Chunk
    transformed_image = background_array[centre_y:centre_y + image_height, centre_x:centre_x + image_width]

    # Rebinarize
    transformed_image = np.where(transformed_image > 0.5, 1, 0)

    return transformed_image

def get_time_window(session_list, model_name):

    # Open Regression Dictionary
    regression_dictionary = np.load(os.path.join(session_list[0], "Simple_Regression", model_name + "_Regression_Model.npy"), allow_pickle=True)[()]

    start_window = regression_dictionary["Start_Window"]
    stop_window = regression_dictionary["Stop_Window"]

    return start_window, stop_window


def create_regression_figure(session_list, model_name, save_directory):

    # Ensure Save Directory Exists
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Get Fine Mask and Atlas Outlines
    mask_location = "/home/matthew/Documents/Allen_Atlas_Templates/Mask_Array.npy"
    atlas_outline_location = "/home/matthew/Documents/Allen_Atlas_Templates/New_Outline.npy"

    fine_mask = np.load(mask_location)
    atlas_outline = np.load(atlas_outline_location)

    mask_alignment_dictionary = np.load("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Consensus_Cluster_Mask_Alignment_Dictionary.npy", allow_pickle=True)[()]
    atlas_alignment_dictionary = np.load("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Consensus_Cluster_Atlas_Alignment_Dictionary.npy", allow_pickle=True)[()]

    fine_mask = transform_mask_or_atlas(fine_mask, mask_alignment_dictionary)
    atlas_outline = transform_mask_or_atlas(atlas_outline, atlas_alignment_dictionary)

    inverse_mask = np.where(fine_mask == 1, 0, 1)
    inverse_mask_pixels = np.nonzero(inverse_mask)
    atlas_outline_pixels = np.nonzero(atlas_outline)

    # Get Learning Type Averages
    condition_1_coefs, condition_2_coefs = get_condition_average(session_list, model_name)
    number_of_sessions = np.shape(condition_1_coefs)[0]


    # Get Colourmaps
    vmin = 0
    vmax = np.percentile(np.array([condition_1_coefs, condition_2_coefs]), q=99.5)
    print("Vmax", vmax)
    #vmax=0.5
    diff_scale_factor = 0.5
    activity_colourmap = cm.ScalarMappable(cmap='jet')
    difference_colourmap = cm.ScalarMappable(cmap='bwr')
    activity_colourmap.set_clim(vmin=vmin, vmax=vmax)
    difference_colourmap.set_clim(vmin=-(vmax*diff_scale_factor), vmax=(vmax*diff_scale_factor))

    # Get Time Window
    timestep = 36
    start_window, stop_window = get_time_window(session_list, model_name)
    x_values = list(range(start_window , stop_window))
    x_values = np.multiply(x_values, timestep)

    # Plot Activity
    plt.ion()
    figure_1 = plt.figure(figsize=(75,100))
    number_of_timepoints = np.shape(condition_1_coefs[0])[0]
    print("Number Of Timepoints")

    for timepoint_index in range(number_of_timepoints):
        print("Timepoint index", timepoint_index)
        figure_1.suptitle(str(x_values[timepoint_index]) + "ms")
        gridspec_1 = GridSpec(nrows=3, ncols=number_of_sessions)

        for session_index in range(number_of_sessions):

            # Create Axes
            condition_1_axis    = figure_1.add_subplot(gridspec_1[0, session_index])
            condition_2_axis    = figure_1.add_subplot(gridspec_1[1, session_index])
            diff_axis           = figure_1.add_subplot(gridspec_1[2, session_index])

            # Get Difference Images
            condition_1_image  = condition_1_coefs[session_index, timepoint_index]
            condition_2_image  = condition_2_coefs[session_index, timepoint_index]
            diff_image = np.subtract(condition_1_image, condition_2_image)


            # Create Images
            condition_1_image = activity_colourmap.to_rgba(condition_1_image)
            condition_2_image = activity_colourmap.to_rgba(condition_2_image)
            diff_image = difference_colourmap.to_rgba(diff_image)

            # Add Masks
            condition_1_image[inverse_mask_pixels] = [1,1,1,1]
            condition_2_image[inverse_mask_pixels] = [1,1,1,1]
            diff_image[inverse_mask_pixels] = [1,1,1,1]

            # Add Atlas Outlines
            condition_1_image[atlas_outline_pixels] = [0,0,0,0]
            condition_2_image[atlas_outline_pixels] = [0,0,0,0]
            diff_image[atlas_outline_pixels] = [0,0,0,0]

            # Plot These Images
            condition_1_axis.imshow(condition_1_image)
            condition_2_axis.imshow(condition_2_image)
            diff_axis.imshow(diff_image)

            # Remove Axes

            # Create Axes
            condition_1_axis.axis('off')
            condition_2_axis.axis('off')
            diff_axis.axis('off')

        plt.draw()
        plt.pause(0.1)
        plt.savefig(os.path.join(save_directory, str(timepoint_index).zfill(3)))
        plt.clf()






# 4.1A - 15
"""
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_08_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_10_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_12_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_14_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_16_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_18_Discrimination_Imaging",
"""


session_list = [
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_23_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_25_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_27_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_03_01_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_03_03_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging",]


save_directory = "/media/matthew/Expansion/Widefield_Analysis/Discrimination_Analysis/Individual_Mice_Changes/NXAK4.1A"
model_name = "Correct_V_Incorrect_v2"
create_regression_figure(session_list, model_name, save_directory)
