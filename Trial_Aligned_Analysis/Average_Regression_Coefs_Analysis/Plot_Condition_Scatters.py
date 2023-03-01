import numpy as np
import h5py
import tables
import matplotlib.pyplot as plt
from scipy import stats
from skimage.transform import resize
import os

from Widefield_Utils import widefield_utils

def create_index_map( indicies, image_height, image_width):

    index_map = np.zeros(image_height * image_width)
    index_map[indicies] = list(range(np.shape(indicies)[1]))
    index_map = np.reshape(index_map, (image_height, image_width))

    """
    plt.title("Index map")
    plt.imshow(index_map)
    plt.show()
    """
    return index_map

def get_roi_pixels(roi_name):

    # Load Pixel Dict
    region_dict = np.load("/home/matthew/Documents/Github_Code_Clean/Widefield_Analysis/Files/Allen_Region_Dict.npy", allow_pickle=True)[()]
    pixel_labels = region_dict['pixel_labels']
    print("Region Dict", region_dict)

    # Load Atlas Dict
    atlas_alignment_dict = np.load("/home/matthew/Documents/Github_Code_Clean/Widefield_Analysis/Files/Atlas_Alignment_Dictionary.npy", allow_pickle=True)[()]

    # Transform Atlas
    pixel_labels = widefield_utils.transform_atlas_regions(pixel_labels, atlas_alignment_dict)

    # Get Selected ROI Mask
    selected_roi_label = region_dict[roi_name]
    roi_mask = np.where(pixel_labels == selected_roi_label, 1, 0)

    # Downsample To 100
    downsample_size = 100
    roi_mask = resize(roi_mask, (downsample_size, downsample_size), anti_aliasing=True, preserve_range=True)
    roi_mask = np.around(roi_mask, decimals=0)

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

    # Create Index Map
    index_map = create_index_map(indicies, image_height, image_width)

    # Get ROI Indicies
    roi_world_indicies = np.nonzero(roi_mask)
    roi_pixel_indicies = index_map[roi_world_indicies]
    roi_pixel_indicies = np.array(roi_pixel_indicies, dtype=np.int)

    return roi_pixel_indicies



def baseline_correct(condition_data, baseline_window):

    # Shape N Mice, N Timepoints, N Pixels
    number_of_mice, number_of_timepoints, number_of_pixels = np.shape(condition_data)

    baseline_corrected_data = []
    for mouse_index in range(number_of_mice):
        mouse_activity = condition_data[mouse_index]
        mouse_baseline = mouse_activity[baseline_window]
        mouse_baseline = np.mean(mouse_baseline, axis=0)
        print("Mouse baseline", np.shape(mouse_baseline))
        mouse_activity = np.subtract(mouse_activity, mouse_baseline)
        baseline_corrected_data.append(mouse_activity)

    baseline_corrected_data = np.array(baseline_corrected_data)
    print("Baseline Corrected Data", np.shape(baseline_corrected_data))

    return baseline_corrected_data


def plot_scatters(condition_1_data, condition_2_data, roi_name):

    number_of_mice = np.shape(condition_1_data)[0]
    print("Number of mice", number_of_mice)

    for mouse_index in range(number_of_mice):
        plt.plot([condition_1_data[mouse_index], condition_2_data[mouse_index]])

    plt.show()


def quantify_roi_activity_across_conditions(tensor_directory, condition_1_index, condition_2_index, baseline_window, response_window, roi_list, roi_name):

    """
    This Test Is Run Pixelwise - All Brains Must Be In Same Pixel Space
    """

    # Load Data
    condition_averages = np.load(os.path.join(tensor_directory, "Average_Coefs", "Mouse_Condition_Average_Matrix.npy"))
    print("Condition Averages", np.shape(condition_averages))

    condition_1_data = condition_averages[:, condition_1_index]
    condition_2_data = condition_averages[:, condition_2_index]

    # Baseline Correct Data
    condition_1_data = baseline_correct(condition_1_data, baseline_window)
    condition_2_data = baseline_correct(condition_2_data, baseline_window)

    # Get Response Window
    condition_1_response = condition_1_data[:, response_window]
    condition_2_response = condition_2_data[:, response_window]
    print("Condition 1 response", np.shape(condition_1_response))

    condition_1_response = np.mean(condition_1_response, axis=1)
    condition_2_response = np.mean(condition_2_response, axis=1)
    print("Condition 1 response", np.shape(condition_1_response))

    # Get ROI Average
    pooled_roi_pixels = []
    for roi in roi_list:
        roi_pixel_indicies = get_roi_pixels(roi)
        for index in roi_pixel_indicies:
            pooled_roi_pixels.append(index)

    print("pooled_roi_pixels", np.shape(pooled_roi_pixels))

    # Get Response Window
    condition_1_region_response = condition_1_response[:, pooled_roi_pixels]
    condition_2_region_response = condition_2_response[:, pooled_roi_pixels]
    print("condition_1_region_response", np.shape(condition_1_region_response))


    condition_1_response = np.mean(condition_1_response, axis=1)
    condition_2_response = np.mean(condition_2_region_response, axis=1)
    print("condition_1_response", np.shape(condition_1_response))

    # Plot ROI Trace
    plot_scatters(condition_2_response, condition_1_response, roi_name)



#tensor_directory = r"/media/matthew/External_Harddrive_2/Control_Switching_Analysis/Full_Model"

tensor_directory = r"/media/matthew/External_Harddrive_2/Neurexin_Switching_Analysis/Full_Model"
condition_1_index = 1
condition_2_index = 3
baseline_window = list(range(0, 14))
#baseline_window = list(range(55, 69))

response_window = list(range(69, 83))

analysis_name = "Full_Model"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)


roi_list = [ "m2_left",  "m2_right"]
roi_name = "M2"
quantify_roi_activity_across_conditions(tensor_directory, condition_1_index, condition_2_index, baseline_window, response_window, roi_list, roi_name)


roi_list = [ "primary_visual_left",  "primary_visual_right"]
roi_name = "V1"
quantify_roi_activity_across_conditions(tensor_directory, condition_1_index, condition_2_index, baseline_window,response_window, roi_list, roi_name)