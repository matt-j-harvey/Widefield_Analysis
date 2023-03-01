import numpy as np
import h5py
import tables
import matplotlib.pyplot as plt
from scipy import stats
from skimage.transform import resize
import os

from Widefield_Utils import widefield_utils
import Trial_Aligned_Utils



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


def get_std_bars(data):

    mean = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    upper_bound = np.add(mean, sd)
    lower_bound = np.subtract(mean, sd)

    return lower_bound, upper_bound


def get_sem_bars(data):

    mean = np.mean(data, axis=0)
    sd = stats.sem(data, axis=0)
    upper_bound = np.add(mean, sd)
    lower_bound = np.subtract(mean, sd)

    return lower_bound, upper_bound


def get_roi_pixels(roi_name):

    # Load Pixel Dict
    region_dict = np.load("/home/matthew/Documents/Github_Code_Clean/Widefield_Analysis/Files/Allen_Region_Dict.npy", allow_pickle=True)[()]
    pixel_labels = region_dict['pixel_labels']

    # Load Atlas Dict
    atlas_alignment_dict = np.load("/home/matthew/Documents/Github_Code_Clean/Widefield_Analysis/Files/Atlas_Alignment_Dictionary.npy", allow_pickle=True)[()]

    # Transform Atlas
    pixel_labels = widefield_utils.transform_atlas_regions(pixel_labels, atlas_alignment_dict)

    # Get Selected ROI Mask
    selected_roi_label = region_dict[roi_name]
    print("Selected ROI Label", selected_roi_label)
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
    print("roi pixel indicies", roi_pixel_indicies)

    """
    # Test THese
    indicies = np.array(indicies)
    print("Indicies ", np.shape(indicies))

    world_roi_indicies = []
    for roi_pixel_index in roi_pixel_indicies:
        print("ROI pixel index", roi_pixel_index)
        roi_world_index = indicies[0, roi_pixel_index]
        world_roi_indicies.append(roi_world_index)

    print("World roi indicies", world_roi_indicies)

    template_mask = np.zeros(image_height * image_width)
    template_mask[world_roi_indicies] = 2
    template_mask = np.reshape(template_mask, (image_height, image_width))

    plt.title("Template map")
    plt.imshow(template_mask)
    plt.show()
    """

    return roi_pixel_indicies

def plot_roi_trace_average(condition_1_data, condition_2_data, roi_name):

    roi_pixel_indicies = get_roi_pixels(roi_name)

    # Get ROI Data
    condition_1_data = condition_1_data[:, :, roi_pixel_indicies]
    condition_2_data = condition_2_data[:, :, roi_pixel_indicies]

    # Get Mean Within Region
    condition_1_data = np.mean(condition_1_data, axis=2)
    condition_2_data = np.mean(condition_2_data, axis=2)

    # Get Mean Across Trials
    condition_1_mean = np.mean(condition_1_data, axis=0)
    condition_2_mean = np.mean(condition_2_data, axis=0)

    print("Condition 1 mean", np.shape(condition_1_mean))
    print("Conditon 2 mean", np.shape(condition_2_mean))

    # Get SD
    c1_lower_bound, c1_upper_bound = get_sem_bars(condition_1_data)
    c2_lower_bound, c2_upper_bound = get_sem_bars(condition_2_data)

    x_values = list(range(len(condition_1_mean)))
    plt.plot(condition_1_mean)
    plt.fill_between(x=x_values, y1=c1_lower_bound, y2=c1_upper_bound, alpha=0.1)
    plt.fill_between(x=x_values, y1=c2_lower_bound, y2=c2_upper_bound, alpha=0.1)
    plt.plot(condition_2_mean)
    plt.title(roi_name)
    plt.show()




def plot_roi_trace_individual(condition_1_data, condition_2_data, roi_name):

    roi_pixel_indicies = get_roi_pixels(roi_name)

    # Get ROI Data
    condition_1_data = condition_1_data[:, :, roi_pixel_indicies]
    condition_2_data = condition_2_data[:, :, roi_pixel_indicies]

    # Get Mean Within Region
    condition_1_data = np.mean(condition_1_data, axis=2)
    condition_2_data = np.mean(condition_2_data, axis=2)

    for trace in condition_1_data:
        plt.plot(trace, c='b')


    for trace in condition_2_data:
        plt.plot(trace, c='g')

    plt.title(roi_name)
    plt.show()




def quantify_roi_activity_n_mouse(tensor_directory, analysis_name, roi_list):

    # Open Analysis Dataframe
    analysis_file = tables.open_file(os.path.join(tensor_directory, analysis_name + "_Trialwise_.h5"), mode="r")
    activity_dataset = analysis_file.root["Data"]
    metadata_dataset = analysis_file.root["Trial_Details"]
    activity_dataset = np.array(activity_dataset)
    metadata_dataset = np.array(metadata_dataset)

    # Get Average Session Response Per Mouse
    condition_1_mouse_average_list, condition_2_mouse_average_list = Trial_Aligned_Utils.get_mouse_session_averages(activity_dataset, metadata_dataset)

    n_mice = len(condition_1_mouse_average_list)

    for mouse_index in range(n_mice):
        mouse_condition_1_data = np.array(condition_1_mouse_average_list[mouse_index])
        mouse_condition_2_data = np.array(condition_2_mouse_average_list[mouse_index])
        print("Condition 1 mouse data", np.shape(mouse_condition_1_data))

        plot_roi_trace_individual(mouse_condition_1_data, mouse_condition_2_data, "primary_visual_left")
        plot_roi_trace_average(mouse_condition_1_data, mouse_condition_2_data, "primary_visual_left")





def quantify_roi_activity_n_session(tensor_directory, analysis_name, roi_list):

    """
    This Test Is Run Pixelwise - All Brains Must Be In Same Pixel Space
    """

    # Open Analysis Dataframe
    analysis_file = tables.open_file(os.path.join(tensor_directory, analysis_name + "_Trialwise_.h5"), mode="r")
    activity_dataset = analysis_file.root["Data"]
    metadata_dataset = analysis_file.root["Trial_Details"]
    activity_dataset = np.array(activity_dataset)
    metadata_dataset = np.array(metadata_dataset)

    # Split By Condition
    condition_details = metadata_dataset[:, 3]
    condition_1_indicies = np.where(condition_details == 0)[0]
    condition_2_indicies = np.where(condition_details == 1)[0]

    condition_1_data = activity_dataset[condition_1_indicies]
    condition_2_data = activity_dataset[condition_2_indicies]
    print("Condition 1 data", np.shape(condition_1_data))
    print("condition 2 data", np.shape(condition_2_data))

    for roi in roi_list:
        plot_roi_trace(condition_1_data, condition_2_data, roi)

# Select Sessions
selected_session_list = [

    [r"NRXN78.1A/2020_12_05_Switching_Imaging",
    r"NRXN78.1A/2020_12_09_Switching_Imaging"],

    [r"NRXN78.1D/2020_11_29_Switching_Imaging",
    r"NRXN78.1D/2020_12_05_Switching_Imaging"],

    [r"NXAK14.1A/2021_05_21_Switching_Imaging",
    r"NXAK14.1A/2021_05_23_Switching_Imaging",
    r"NXAK14.1A/2021_06_11_Switching_Imaging",
    r"NXAK14.1A/2021_06_13_Transition_Imaging",
    r"NXAK14.1A/2021_06_15_Transition_Imaging",
    r"NXAK14.1A/2021_06_17_Transition_Imaging"],

    [r"NXAK22.1A/2021_10_14_Switching_Imaging",
    r"NXAK22.1A/2021_10_20_Switching_Imaging",
    r"NXAK22.1A/2021_10_22_Switching_Imaging",
    r"NXAK22.1A/2021_10_29_Transition_Imaging",
    r"NXAK22.1A/2021_11_03_Transition_Imaging",
    r"NXAK22.1A/2021_11_05_Transition_Imaging"],

    [r"NXAK4.1B/2021_03_02_Switching_Imaging",
    r"NXAK4.1B/2021_03_04_Switching_Imaging",
    r"NXAK4.1B/2021_03_06_Switching_Imaging",
    #r"NXAK4.1B/2021_04_02_Transition_Imaging",
    r"NXAK4.1B/2021_04_08_Transition_Imaging",
    r"NXAK4.1B/2021_04_10_Transition_Imaging"],

    [r"NXAK7.1B/2021_02_26_Switching_Imaging",
    r"NXAK7.1B/2021_02_28_Switching_Imaging",
    r"NXAK7.1B/2021_03_02_Switching_Imaging",
    r"NXAK7.1B/2021_03_23_Transition_Imaging",
    r"NXAK7.1B/2021_03_31_Transition_Imaging",
    #r"NXAK7.1B/2021_04_02_Transition_Imaging"
    ],

]

# Set Tensor Directory
data_root_directory = r""
tensor_directory = r"//media/matthew/External_Harddrive_2/Control_Switching_Tensors_Mean_Only"


# Select Analysis Details
analysis_name = "Unrewarded_Contextual_Modulation"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)

roi_list = ["primary_visual_left", "primary_visual_right", "retrosplenial", "m2_left", "m2_right"]

#quantify_roi_activity(tensor_directory, analysis_name, roi_list)

quantify_roi_activity_n_mouse(tensor_directory, analysis_name, roi_list)
