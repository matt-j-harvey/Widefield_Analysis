import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import sys
from scipy import ndimage, stats
from skimage.transform import resize
from skimage.segmentation import chan_vese
from scipy import stats
from tqdm import tqdm

from matplotlib.pyplot import cm

import Trial_Aligned_Utils



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
    indicies, image_height, image_width = load_generous_mask(base_directory)

    # Load Alignment Dictionary
    alignment_dictionary = np.load(os.path.join(base_directory, "Cluster_Alignment_Dictionary.npy"), allow_pickle=True)[()]

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

            # Align Image
            condition_timepoint_coefs = transform_image(condition_timepoint_coefs, alignment_dictionary)

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

    indicies, image_height, image_width = Trial_Aligned_Utils.load_tight_mask()

    reconstructed_mean = []

    for frame in mean_activity:

        # Reconstruct Image
        frame = Trial_Aligned_Utils.create_image_from_data(frame, indicies, image_height, image_width)

        reconstructed_mean.append(frame)

    reconstructed_mean = np.array(reconstructed_mean)
    return reconstructed_mean

def get_roi_indicies(roi_pixels):

    roi_pixels = np.ndarray.astype(roi_pixels, int)
    indices, image_height, image_width = Trial_Aligned_Utils.load_tight_mask()
    number_of_indicies = np.shape(indices)[1]

    template = np.zeros(image_height * image_width)
    brain_values = np.zeros(number_of_indicies)
    brain_values[roi_pixels] = 1
    template[indices] = brain_values
    template = np.reshape(template, (image_height, image_width))
    two_d_indicies= np.nonzero(template)

    """
    new_template = np.zeros((image_height, image_width))
    new_template[two_d_indicies] = 1
    plt.imshow(new_template)
    plt.show()
    """
    return two_d_indicies


def get_roi_activity(base_directory, mean_activity_trace, roi_name):

    # Reconstruct Mean Activity Trace
    mean_activity_trace = reconstruct_mean_trace(mean_activity_trace)
    print("Mean Trace Shape", np.shape(mean_activity_trace))
    # Get ROI Pixels
    """
    roi_pixels_left = np.load(os.path.join(base_directory, "Custom_ROIs", roi_name + "_Left_Coords.npy"))
    roi_pixels_right = np.load(os.path.join(base_directory, "Custom_ROIs", roi_name + "_Right_Coords.npy"))
    roi_pixels = [np.concatenate([roi_pixels_left[0], roi_pixels_right[0]]),
                  np.concatenate([roi_pixels_left[1], roi_pixels_right[1]])]
    """
    roi_pixels = np.load(os.path.join(base_directory, roi_name))
    roi_pixels = get_roi_indicies(roi_pixels)

    # Get Region Pixels
    region_trace = []
    number_of_timepoints = np.shape(mean_activity_trace)[0]
    for timepoint_index in range(number_of_timepoints):
        frame = mean_activity_trace[timepoint_index]
        roi_activity = frame[roi_pixels]
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


def compare_signifiance(group_roi_traces):

    for condition in group_roi_traces:
        print("Condition Shape", np.shape(condition))

    cond_1_v_2_t, cond_1_v_2_p = stats.ttest_rel(group_roi_traces[0], group_roi_traces[1], axis=0)
    cond_2_v_3_t, cond_2_v_3_p = stats.ttest_rel(group_roi_traces[1], group_roi_traces[2], axis=0)

    # Save These
    np.save(r"/media/matthew/29D46574463D2856/Nature_Transition_Analysis_Results/ROI_Trace_Raw_Values/Present_v_Absent_Expected_t.npy", cond_1_v_2_t)
    np.save(r"/media/matthew/29D46574463D2856/Nature_Transition_Analysis_Results/ROI_Trace_Raw_Values/Present_v_Absent_Expected_p.npy", cond_1_v_2_p)
    np.save(r"/media/matthew/29D46574463D2856/Nature_Transition_Analysis_Results/ROI_Trace_Raw_Values/Absent_Expected_v_Absent_Not_Expected_t.npy", cond_2_v_3_t)
    np.save(r"/media/matthew/29D46574463D2856/Nature_Transition_Analysis_Results/ROI_Trace_Raw_Values/Absent_Expected_v_Absent_Not_Expected_p.npy", cond_2_v_3_p)

    return cond_1_v_2_p

def plot_mean_trace(group_roi_traces, number_of_conditions, timestep):

    # Get P Values
    p_values = compare_signifiance(group_roi_traces)
    sign_binary = np.where(p_values < 0.01, 0.020, 0)

    # Plot These
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    colour_list = ['g', 'r', 'b']
    condition_names = ["odour_expected_arrives", "odour_expected_absent", "odour_not_expected_absent"]

    save_base_directory = r"/media/matthew/29D46574463D2856/Nature_Transition_Analysis_Results/ROI_Trace_Raw_Values"

    for condition_index in range(number_of_conditions):

        condition_tensor = group_roi_traces[condition_index]
        condition_tensor = ensure_roi_tensor_structure(condition_tensor)

        condition_mean = np.nanmean(condition_tensor, axis=0)
        condition_sem = stats.sem(condition_tensor, axis=0, nan_policy='omit')

        condition_upper_bound = np.add(condition_mean, condition_sem)
        condition_lower_bounds = np.subtract(condition_mean, condition_sem)

        condition_colour = colour_list[condition_index]

        x_values = list(range(0, len(condition_mean)))
        x_values = np.subtract(x_values, start_window)
        x_values = np.multiply(x_values, timestep)

        axis_1.plot(x_values, condition_mean, c=condition_colour)
        axis_1.fill_between(x=x_values, y1=condition_lower_bounds, y2=condition_upper_bound, alpha=0.2, color=condition_colour)

        # Save Values
        np.save(os.path.join(save_base_directory, condition_names[condition_index] + "_Mean.npy"), condition_mean)
        np.save(os.path.join(save_base_directory, condition_names[condition_index] + "_SEM.npy"), condition_sem)
        np.save(os.path.join(save_base_directory, condition_names[condition_index] + "_x_values.npy"), x_values)

    axis_1.axvspan(1800, 3240, color='tab:orange', alpha=0.2)

    # Add Legend
    axis_1.legend(loc="lower right")

    # Set Axis Labels
    axis_1.set_xlabel("Time (ms)")
    axis_1.set_ylabel("dF/F")

    # Add Signfiance
    axis_1.scatter(x_values, sign_binary)



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
    axis_1.axvspan(1800, 3240, color='tab:orange', alpha=0.2)

    # Add Legend
    axis_1.legend(loc="lower right")

    # Set Axis Labels
    axis_1.set_xlabel("Time (ms)")
    axis_1.set_ylabel("dF/F (%)")

    plt.show()



def quantify_roi_activity(session_list, tensor_names, extended_tensor_root_directory, roi_name, start_window, baseline_correct=False):

    # Get Time Window
    timestep = 36
    #x_values = list(range(start_window , stop_window))
    #x_values = np.multiply(x_values, timestep)

    number_of_conditions = len(tensor_names)

    # Get Regions Of Selected pixes
    group_roi_traces = []

    for condition_index in range(number_of_conditions):

        condition_name = tensor_names[condition_index].replace("_onsets.npy", "")
        condition_tensor_list = []

        for base_directory in tqdm(session_list):

            # Get Tensor Filename
            session_tensor_directory = Trial_Aligned_Utils.check_save_directory(base_directory, extended_tensor_root_directory)
            activity_tensor_file = os.path.join(session_tensor_directory, condition_name + "_Extended_Activity_Tensor.npy")
            print("Activity tensor file")
            print(activity_tensor_file)
            if os.path.exists(activity_tensor_file):

                # Load Activity Tensor
                activity_tensor = np.load(activity_tensor_file, allow_pickle=True)

                # Baseline Correct If Needed
                if baseline_correct == True:
                    activity_tensor = correct_baseline_of_tensor(activity_tensor, start_window)

                # Ensure Structure
                activity_tensor = ensure_tensor_structure(activity_tensor)

                # Get Average
                mean_activity = np.nanmean(activity_tensor, axis=0)

                # Get ROI Activity
                roi_trace = get_roi_activity(base_directory, mean_activity, roi_name)

                # Add To List
                condition_tensor_list.append(roi_trace)

        group_roi_traces.append(condition_tensor_list)

    #plot_individual_sessions(group_roi_traces, number_of_conditions, timestep, condition_names)
    plot_mean_trace(group_roi_traces, number_of_conditions, timestep)



session_list = [

        r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_23_Transition_Imaging",
        r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_31_Transition_Imaging",
        r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_04_02_Transition_Imaging",

        r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_02_Transition_Imaging",
        # r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_08_Transition_Imaging",
        r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_10_Transition_Imaging",

        r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_13_Transition_Imaging",
        r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_15_Transition_Imaging",
        r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_17_Transition_Imaging",

        r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_10_29_Transition_Imaging",
        r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_11_03_Transition_Imaging",
        r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_11_05_Transition_Imaging"

    ]


# Get Analysis Details
analysis_name = "Absence Of Expected Odour"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Trial_Aligned_Utils.load_analysis_container(analysis_name)
stop_stimuli_list = [["Odour 1", "Visual 1", "Visual 2"], ["Odour 1", "Odour 2", "Visual 1", "Visual 2"], ["Odour 1", "Odour 2", "Visual 1", "Visual 2"]]
tensor_save_directory = r"/media/matthew/29D46574463D2856/Nature_Transition_Analysis_Results/Extended_Tensors"
roi_name = "Selected_ROI.npy"
quantify_roi_activity(session_list, onset_files, tensor_save_directory, roi_name, start_window, baseline_correct=False)
