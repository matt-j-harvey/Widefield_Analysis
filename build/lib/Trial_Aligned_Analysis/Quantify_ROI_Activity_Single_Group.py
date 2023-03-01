import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import sys
from scipy import ndimage, stats
from skimage.transform import resize
from skimage.segmentation import chan_vese
from tqdm import tqdm
import pickle
from matplotlib.pyplot import cm

from Widefield_Utils import widefield_utils
from Files import Session_List
import Transition_Utils

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



def split_sessions_By_d_prime(session_list, intermediate_threshold, post_threshold):

    pre_learning_sessions = []
    intermediate_learning_sessions = []
    post_learning_sessions = []

    # Iterate Throug Sessions
    for session in session_list:

        # Load D Prime
        behavioural_dictionary = np.load(os.path.join(session, "Behavioural_Measures", "Performance_Dictionary.npy"), allow_pickle=True)[()]
        d_prime = behavioural_dictionary["visual_d_prime"]

        if d_prime >= post_threshold:
            post_learning_sessions.append(session)

        if d_prime < post_threshold and d_prime >= intermediate_threshold:
            intermediate_learning_sessions.append(session)

        if d_prime < intermediate_threshold:
            pre_learning_sessions.append(session)

    return pre_learning_sessions, intermediate_learning_sessions, post_learning_sessions



def get_coefficient_traces(base_directory, model_name, region_name):

    # Load Region Pixels
    left_region_pixels = np.load(os.path.join(base_directory, "Custom_ROIs", region_name + "_Left_Coords.npy"))
    right_region_pixels = np.load(os.path.join(base_directory, "Custom_ROIs", region_name + "_Right_Coords.npy"))


    # Open Regression Dictionary
    regression_dictionary = np.load(os.path.join(base_directory, "Simple_Regression", model_name + "_Regression_Model.npy"), allow_pickle=True)[()]

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()

    # Unpack Regression Dictionary
    coefficient_matrix = regression_dictionary["Regression_Coefficients"]
    start_window = regression_dictionary["Start_Window"]
    stop_window = regression_dictionary["Stop_Window"]
    trial_length = stop_window - start_window
    coefficient_matrix = np.nan_to_num(coefficient_matrix)
    number_of_conditions = len(coefficient_matrix)

    session_condition_traces = []

    for condition_index in range(number_of_conditions):
        condition_region_trace = []
        condition_coefs = coefficient_matrix[condition_index]
        condition_coefs = np.transpose(condition_coefs)

        for timepoint_index in range(trial_length):

            # Get Timepoint Coefficients
            condition_timepoint_coefs = condition_coefs[timepoint_index]

            # Reconstruct Image
            condition_timepoint_coefs = Widefield_General_Functions.create_image_from_data(condition_timepoint_coefs, indicies, image_height, image_width)

            # Get Region Pixels
            left_region_coefs = condition_timepoint_coefs[left_region_pixels]
            right_region_coefs = condition_timepoint_coefs[right_region_pixels]

            # Get Region Mean
            left_region_mean = np.mean(left_region_coefs)
            right_region_mean = np.mean(right_region_coefs)
            region_total = np.add(left_region_mean, right_region_mean)

            # Add To List
            condition_region_trace.append(region_total)

        session_condition_traces.append(condition_region_trace)

    session_condition_traces = np.array(session_condition_traces)
    return session_condition_traces



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


def transform_region_mask(region_assignments, variable_dictionary):

    image_height = 600
    image_width = 608

    # Unpack Dictionary
    angle = variable_dictionary['rotation']
    x_shift = variable_dictionary['x_shift']
    y_shift = variable_dictionary['y_shift']
    x_scale = variable_dictionary['x_scale']
    y_scale = variable_dictionary['y_scale']

    transformed_image = np.zeros(np.shape(region_assignments))

    # Calculate New Height
    original_height, original_width = np.shape(transformed_image)
    new_height = int(original_height * y_scale)
    new_width = int(original_width * x_scale)

    # Get Unique Regions
    unique_regions = np.unique(region_assignments)
    for region in unique_regions:

        region_mask = np.where(region_assignments == region, 1, 0)

        # Scale
        region_mask = resize(region_mask, (new_height, new_width), preserve_range=True)

        # Rotate
        region_mask = ndimage.rotate(region_mask, angle, reshape=False, prefilter=True)

        # Insert Into Background
        mask_height, mask_width = np.shape(region_mask)
        centre_x = 200
        centre_y = 200
        background_array = np.zeros((1000, 1000))
        x_start = centre_x + x_shift
        x_stop = x_start + mask_width

        y_start = centre_y + y_shift
        y_stop = y_start + mask_height

        background_array[y_start:y_stop, x_start:x_stop] = region_mask

        # Take Chunk
        region_mask = background_array[centre_y:centre_y + image_height, centre_x:centre_x + image_width]

        # Rebinarize
        transformed_image = np.where(region_mask > 0.5, region, transformed_image)

    return transformed_image




def get_time_window(session_list, model_name):

    # Open Regression Dictionary
    regression_dictionary = np.load(os.path.join(session_list[0], "Simple_Regression", model_name + "_Regression_Model.npy"), allow_pickle=True)[()]

    start_window = regression_dictionary["Start_Window"]
    stop_window = regression_dictionary["Stop_Window"]

    return start_window, stop_window



def reconstruct_mean_trace(mean_activity):

    indicies, image_height, image_width = widefield_utils.load_tight_mask()

    reconstructed_mean = []

    for frame in mean_activity:

        # Reconstruct Image
        frame = widefield_utils.create_image_from_data(frame, indicies, image_height, image_width)

        reconstructed_mean.append(frame)

    reconstructed_mean = np.array(reconstructed_mean)
    return reconstructed_mean



def get_roi_activity(mean_activity_trace, roi_indicies):

    # Get Region Pixels
    region_trace = []
    number_of_timepoints = np.shape(mean_activity_trace)[0]
    for timepoint_index in range(number_of_timepoints):
        frame = mean_activity_trace[timepoint_index]
        roi_activity = frame[roi_indicies]
        roi_activity = np.mean(roi_activity)
        region_trace.append(roi_activity)
    region_trace = np.array(region_trace)

    return region_trace

def ensure_roi_tensor_structure(tensor):
    if np.ndim(tensor) != 2:
        tensor = pad_ragged_roi_tensor_with_nans(tensor)
    return tensor



def pad_ragged_roi_tensor_with_nans(ragged_tensor):
    # Get Longest Trial
    length_list = []
    for trial in ragged_tensor:
        trial_length = np.shape(trial)
        length_list.append(trial_length)

    max_length = np.max(length_list)

    # Create Padded Tensor
    number_of_trials = len(length_list)
    padded_tensor = np.empty((number_of_trials, max_length))
    padded_tensor[:] = np.nan

    # Fill Padded Tensor
    for trial_index in range(number_of_trials):
        trial_data = ragged_tensor[trial_index]
        trial_length = np.shape(trial_data)[0]
        padded_tensor[trial_index, 0:trial_length] = trial_data

    return padded_tensor



def ensure_tensor_structure(tensor):
    if np.ndim(tensor) != 3:
        tensor = pad_ragged_tensor_with_nans(tensor)
    return tensor


def pad_ragged_tensor_with_nans(ragged_tensor):
    # Get Longest Trial
    length_list = []
    for trial in ragged_tensor:
        trial_length, number_of_pixels = np.shape(trial)
        length_list.append(trial_length)

    max_length = np.max(length_list)

    # Create Padded Tensor
    number_of_trials = len(length_list)
    padded_tensor = np.empty((number_of_trials, max_length, number_of_pixels))
    padded_tensor[:] = np.nan

    # Fill Padded Tensor
    for trial_index in range(number_of_trials):
        trial_data = ragged_tensor[trial_index]
        trial_length = np.shape(trial_data)[0]
        padded_tensor[trial_index, 0:trial_length] = trial_data

    return padded_tensor



def plot_mean_trace(group_roi_traces, number_of_conditions, timestep):

    # Plot These
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    colour_list = ['g', 'r', 'b']

    for condition_index in range(number_of_conditions):

        condition_tensor = group_roi_traces[condition_index]
        #condition_tensor = ensure_roi_tensor_structure(condition_tensor)
        print("Condition tensor", np.shape(condition_tensor))

        condition_mean = np.nanmean(condition_tensor, axis=0)
        condition_sem = stats.sem(condition_tensor, axis=0, nan_policy='omit')

        condition_upper_bound = np.add(condition_mean, condition_sem)
        condition_lower_bounds = np.subtract(condition_mean, condition_sem)

        condition_colour = colour_list[condition_index]
        print("Condition mean shape", np.shape(condition_mean))

        x_values = list(range(0, len(condition_mean)))
        x_values = np.subtract(x_values, start_window)
        x_values = np.multiply(x_values, timestep)

        axis_1.plot(x_values, condition_mean, c=condition_colour)
        axis_1.fill_between(x=x_values, y1=condition_lower_bounds, y2=condition_upper_bound, alpha=0.2, color=condition_colour)

    #axis_1.axvspan(1800, 3240, color='tab:orange', alpha=0.2)

    # Add Legend
    axis_1.legend(loc="lower right")

    # Set Axis Labels
    axis_1.set_xlabel("Time (ms)")
    axis_1.set_ylabel("Activity (AU)")


    plt.show()



def correct_baseline_of_tensor(tensor, start_size):

    corrected_tensor = []

    for trial in tensor:
        trial_baseline = trial[0:-start_size]
        trial_baseline = np.mean(trial_baseline, axis=0)
        trial = np.subtract(trial, trial_baseline)
        corrected_tensor.append(trial)

    corrected_tensor = np.array(corrected_tensor)
    return corrected_tensor



def plot_individual_sessions(group_roi_traces, number_of_conditions, timestep, condition_names):

    # Plot These
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    colour_list = ['g', 'r', 'b']

    print("Group ROI Traces", np.shape(group_roi_traces))



    for condition_index in range(number_of_conditions):

        condition_tensor = group_roi_traces[condition_index]
        print("Condition tensor shape", np.shape(condition_tensor))

        #condition_tensor = ensure_roi_tensor_structure(condition_tensor)
        print("Condition Tensor", np.shape(condition_tensor))

        condition_colour = colour_list[condition_index]

        trace_count = 0
        for roi_trace in condition_tensor:
            x_values = list(range(0, len(roi_trace)))
            x_values = np.subtract(x_values, start_window)
            x_values = np.multiply(x_values, timestep)

            if trace_count == 0:
                axis_1.plot(x_values, roi_trace, c=condition_colour, label = condition_names[condition_index])
            else:
                axis_1.plot(x_values, roi_trace, c=condition_colour)

            trace_count += 1
        """
        condition_mean = np.nanmean(condition_tensor, axis=0)
        # condition_sem = np.nanstd(condition_tensor, axis=0)
        condition_sem = stats.sem(condition_tensor, axis=0)
        print("Condition Mean", np.shape(condition_mean))
        condition_upper_bound = np.add(condition_mean, condition_sem)
        condition_lower_bounds = np.subtract(condition_mean, condition_sem)
        """


        #axis_1.fill_between(x=x_values, y1=condition_lower_bounds, y2=condition_upper_bound, alpha=0.2, color=condition_colour)

    # Add Odour Window
    #axis_1.axvspan(1800, 3240, color='tab:orange', alpha=0.2)

    # Add Legend
    axis_1.legend(loc="lower right")

    # Set Axis Labels
    axis_1.set_xlabel("Time (ms)")
    axis_1.set_ylabel("Activity (AU)")

    plt.show()



def quantify_roi_activity(session_list, tensor_names, tensor_save_directory, roi_indicies, start_window, save_directory, condition_names, baseline_correct=False):

    indicies, image_height, image_width = widefield_utils.load_tight_mask()

    # Get Time Window
    timestep = 36
    #x_values = list(range(start_window , stop_window))
    #x_values = np.multiply(x_values, timestep)

    number_of_conditions = len(tensor_names)

    # Get Regions Of Selected pixes
    group_roi_traces = []

    for condition_index in range(number_of_conditions):
        trial_tensor_name = tensor_names[condition_index].replace("_onsets.npy", ".pickle")

        condition_tensor_list = []
        for base_directory in tqdm(session_list):

            # Load Trial Tensor Dict
            session_tensor_directory = Transition_Utils.check_save_directory(base_directory, tensor_save_directory)
            session_tensor_file = os.path.join(session_tensor_directory, trial_tensor_name)

            with open(session_tensor_file, 'rb') as handle:
                trial_tensor = pickle.load(handle)

            activity_tensor = trial_tensor["activity_tensor"]
            print("Activity Tensor Shape", np.shape(activity_tensor))


            # Baseline Correct If Needed
            if baseline_correct == True:
                activity_tensor = correct_baseline_of_tensor(activity_tensor, start_window)

            # Get Average
            mean_activity = np.nanmean(activity_tensor, axis=0)

            # Get ROI Activity
            roi_trace = get_roi_activity(mean_activity, roi_indicies)

            # Add To List
            condition_tensor_list.append(roi_trace)

        group_roi_traces.append(condition_tensor_list)

    #plot_individual_sessions(group_roi_traces, number_of_conditions, timestep, condition_names)
    plot_mean_trace(group_roi_traces, number_of_conditions, timestep)


def view_roi(left_coords, right_coords):
    print("left min", np.min(left_coords))
    print("left max", np.max(left_coords))
    print("right min", np.min(right_coords))
    print("right max", np.max(right_coords))

    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    template = np.zeros(image_height * image_width)
    template[indicies] = 1
    template = np.reshape(template, (image_height, image_width))

    template[left_coords[0], left_coords[1]] = 2

    template[right_coords[0], right_coords[1]] = 2
    plt.imshow(template)
    plt.show()

def get_roi_pixels():
    roi_indicies = np.load(os.path.join(base_directory, "Selected_ROI.npy"))
    print("Pixels Shape", np.shape(roi_indicies))
    indicies = np.array(indicies)
    print("Indicies shape", np.shape(indicies))
    selected_indicies = indicies[:, roi_indicies]


# Load Session List
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
tensor_directory = r"//media/matthew/External_Harddrive_2/Control_Switching_Tensors_Mean_Only"



#selected_session_list = Session_List.control_transition_sessions
#tensor_root_directory = r"/media/matthew/External_Harddrive_2/Control_Transition_Tensors/Raw_Activity"


# Select Region
custom_roi_directory = r"/media/matthew/Expansion/Custom_ROIs"
region_name = "Medial_Frontal_Secondary_Motor"
roi_indicies = np.load(os.path.join(custom_roi_directory, region_name + ".npy"))

# Load Analysis Details
analysis_name = "Absence Of Expected Odour"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)


# Set Save Directory
save_directory = r"/media/matthew/External_Harddrive_2/Neurexin_Transition_Tensors/Results/Absence Of Expected Odour/Figure/Graph"
condition_names = ["Odour Expected Present", "Odour Expected", "Odour Not Expected Absent"]

quantify_roi_activity(selected_session_list, onset_files, tensor_root_directory, roi_indicies, start_window, save_directory, condition_names, baseline_correct=False)
