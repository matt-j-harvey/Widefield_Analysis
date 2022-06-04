import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import sys
import scipy.signal
from sklearn.linear_model import LinearRegression

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)



def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def smooth_activity_tensor(activity_tensor):

    number_of_regions = np.shape(activity_tensor)[1]

    smoothed_tensor = []
    for region_index in range(number_of_regions):
        region_trace = activity_tensor[:, region_index]
        smoothed_trace = moving_average(region_trace, n=5)
        smoothed_tensor.append(smoothed_trace)
    smoothed_tensor = np.array(smoothed_tensor)
    smoothed_tensor = np.transpose(smoothed_tensor)
    return smoothed_tensor


def get_activity_tensor(base_directory, onset_list, start_window, stop_window):

    # Get Data Structure
    number_of_trials = len(onset_list)

    # Load Delta F Matrix
    delta_f_matrix_filepath = os.path.join(base_directory, "Delta_F_Registered.hdf5")
    delta_f_matrix_container = h5py.File(delta_f_matrix_filepath, 'r')
    delta_f_matrix = delta_f_matrix_container['Data']

    # Create Empty Tensor To Hold Data
    activity_tensor = []

    # Get Correlation Matrix For Each Trial
    for trial_index in range(number_of_trials):

        trial_onset = onset_list[trial_index]
        trial_start = trial_onset + start_window
        trial_stop = trial_onset + stop_window

        # Get Trial Actvity
        trial_activity = delta_f_matrix[trial_start:trial_stop]

        # Subtract Baseline
        baseline = delta_f_matrix[trial_start-20:trial_start]
        baseline = np.mean(baseline, axis=0)
        trial_activity = np.subtract(trial_activity, baseline)

        activity_tensor.append(trial_activity)
    
    activity_tensor = np.array(activity_tensor)
    activity_tensor = np.nan_to_num(activity_tensor)
    return activity_tensor



def get_region_trace(activity_tensor, pixel_assignments, selected_region_labels):

    # Get Region Indicies
    region_indicies_list = []
    for label in selected_region_labels:
        label_mask = np.where(pixel_assignments == label, 1, 0)
        label_indicies = list(np.nonzero(label_mask)[0])
        for index in label_indicies:
            region_indicies_list.append(index)

    region_indicies_list.sort()

    # Get Region Tensor
    region_tensor = activity_tensor[:, :, region_indicies_list]

    # Get Region Mean
    region_tensor = np.mean(region_tensor, axis=2)

    return region_tensor



def get_region_indexes(unique_labels, region_labels):

    region_index_list = []
    for label in region_labels:
        label_index = unique_labels.index(label)
        region_index_list.append(label_index)

    return region_index_list


def flatten_region_tensor(region_tensor):
    number_of_trials = np.shape(region_tensor)[0]
    trial_length = np.shape(region_tensor)[1]
    region_tensor = np.reshape(region_tensor, (number_of_trials * trial_length))
    return region_tensor


def flatten_activity_tensor(activity_tensor):
    number_of_trials = np.shape(activity_tensor)[0]
    trial_length = np.shape(activity_tensor)[1]
    number_of_pixels = np.shape(activity_tensor)[2]
    activity_tensor = np.reshape(activity_tensor, (number_of_trials * trial_length, number_of_pixels))
    return activity_tensor



def create_stimuli_regressor(activity_tensor_list):

    # Get Data Structure
    trial_number_list = []
    for tensor in activity_tensor_list:
        number_of_trials = np.shape(tensor)[0]
        trial_number_list.append(number_of_trials)
    total_number_of_trials = np.sum(trial_number_list)
    trial_length = np.shape(activity_tensor_list[0])[1]
    number_of_stimuli = len(activity_tensor_list)
    total_number_of_timepoints = total_number_of_trials * trial_length


    # Create Stimuli Regressors
    stimuli_regressors = np.zeros((total_number_of_timepoints, number_of_stimuli * trial_length))

    trial_count = 0
    for stimuli_index in range(number_of_stimuli):
        stimuli_trials = trial_number_list[stimuli_index]

        for trial_index in range(stimuli_trials):
            row_start = trial_count * trial_length
            row_stop = row_start + trial_length

            column_start = stimuli_index * trial_length
            column_stop = column_start + trial_length

            stimuli_regressors[row_start:row_stop, column_start:column_stop] = np.identity(trial_length)
            trial_count += 1

    """
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.imshow(stimuli_regressors)
    forceAspect(axis_1)
    plt.show()
    """

    return stimuli_regressors



def create_regression_model(region_tensor_list, activity_tensor_list, running_tensor_list):

    # Create Stimuli Regressors
    stimuli_regressors = create_stimuli_regressor(activity_tensor_list)


    # Get Stimuli_Trials
    stimuli_trial_numbers = []
    for tensor in activity_tensor_list:
        stimuli_trial_numbers.append(np.shape(tensor)[0])
    total_number_of_trials = np.sum(stimuli_trial_numbers)
    trial_length = np.shape(activity_tensor_list[0])[1]
    context_1_trials = stimuli_trial_numbers[0] + stimuli_trial_numbers[1]


    # Flatten Tensors
    flat_activity_tensor = []
    for tensor in activity_tensor_list:
        tensor = flatten_activity_tensor(tensor)
        flat_activity_tensor.append(tensor)
    flat_activity_tensor = np.vstack(flat_activity_tensor)

    # Flatten Region Tensor
    flat_region_tensor = []
    for tensor in region_tensor_list:
        tensor = flatten_region_tensor(tensor)
        print("region tensor shape", np.shape(tensor))
        flat_region_tensor.append(tensor)

    # Concatenate Running Tensor
    running_tensor_list = np.concatenate(running_tensor_list)
    print("Running tensor list", running_tensor_list)


    flat_region_tensor = np.concatenate(flat_region_tensor)

    context_1_boxcar = np.zeros(total_number_of_trials * trial_length)
    context_2_boxcar = np.zeros(total_number_of_trials * trial_length)
    context_1_boxcar[0: context_1_trials * trial_length] = 1
    context_2_boxcar[context_1_trials * trial_length:] = 1

    context_1_region_trace = np.multiply(flat_region_tensor, context_1_boxcar)
    context_2_region_trace = np.multiply(flat_region_tensor, context_2_boxcar)

    flat_region_tensor = np.expand_dims(flat_region_tensor, 1)
    context_1_region_trace = np.expand_dims(context_1_region_trace, 1)
    context_2_region_trace = np.expand_dims(context_2_region_trace, 1)

    print("Stimuli Regressors", np.shape(stimuli_regressors))
    print("Flat Region Trace", np.shape(flat_region_tensor))
    print("Context 1 trace", np.shape(context_1_region_trace))
    print("Context 2 region trace", np.shape(context_2_region_trace))
    print("Activity tensor shape", np.shape(flat_activity_tensor))

    design_matrix = np.hstack([stimuli_regressors, running_tensor_list, flat_region_tensor, context_1_region_trace, context_2_region_trace])

    # Create Model
    model = LinearRegression()
    model.fit(X=design_matrix, y=flat_activity_tensor)

    # Get Coefs
    coefs = model.coef_
    print("Coefs shape", np.shape(coefs))
    baseline_coef = coefs[:, -3]
    context_1_coef = coefs[:, -2]
    context_2_coef = coefs[:, -1]

    return context_1_coef, context_2_coef, baseline_coef



def get_running_tensor(downsampled_running_trace, onset_list, start_window, stop_window):

    # Load Downsampled_Running_Trace
    running_tensor = []
    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window
        running_data = downsampled_running_trace[trial_start:trial_stop]
        running_tensor.append(running_data)
    running_tensor = np.array(running_tensor)

    # Flatten Tensor
    number_of_trials = len(onset_list)
    trial_length = stop_window - start_window

    running_tensor = np.reshape(running_tensor, (number_of_trials * trial_length, 1))
    return running_tensor


def view_traces(base_directory, start_window, stop_window, view=True):

    v1 = [45, 46]
    m2 = [8,9]
    rsc = [28, 32]
    m1 = [11, 12]
    mv2 = [47, 48]
    selected_region = v1


    # Load Pixel assignments
    pixel_assignments = np.load("/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/Pixel_Assignmnets.npy")

    # Load Onsets
    visual_context_vis_1_onsets = list(np.load(os.path.join(base_directory, "Stimuli_Onsets", "visual_context_stable_vis_1_onsets.npy")))
    visual_context_vis_2_onsets = list(np.load(os.path.join(base_directory, "Stimuli_Onsets", "visual_context_stable_vis_2_onsets.npy")))
    odour_context_vis_1_onsets = list(np.load(os.path.join(base_directory,  "Stimuli_Onsets", "odour_context_stable_vis_1_onsets.npy")))
    odour_context_vis_2_onsets = list(np.load(os.path.join(base_directory,  "Stimuli_Onsets", "odour_context_stable_vis_2_onsets.npy")))

    # Load Activity Tensors
    visual_context_vis_1_tensor = get_activity_tensor(base_directory, visual_context_vis_1_onsets, start_window, stop_window)
    visual_context_vis_2_tensor = get_activity_tensor(base_directory, visual_context_vis_2_onsets, start_window, stop_window)
    odour_context_vis_1_tensor  = get_activity_tensor(base_directory, odour_context_vis_1_onsets, start_window, stop_window)
    odour_context_vis_2_tensor  = get_activity_tensor(base_directory, odour_context_vis_2_onsets, start_window, stop_window)

    # Get Region Tensors
    visual_context_vis_1_region_tensor = get_region_trace(visual_context_vis_1_tensor,  pixel_assignments, selected_region)
    visual_context_vis_2_region_tensor = get_region_trace(visual_context_vis_2_tensor,  pixel_assignments, selected_region)
    odour_context_vis_1_region_tensor  = get_region_trace(odour_context_vis_1_tensor,  pixel_assignments, selected_region)
    odour_context_vis_2_region_tensor  = get_region_trace(odour_context_vis_2_tensor, pixel_assignments, selected_region)

    # Get Running Tensors
    downsampled_running_trace = np.load(os.path.join(base_directory, "Movement_Controls", "Downsampled_Running_Trace.npy"))
    visual_context_vis_1_running_tensor = get_running_tensor(downsampled_running_trace, visual_context_vis_1_onsets, start_window, stop_window)
    visual_context_vis_2_running_tensor = get_running_tensor(downsampled_running_trace, visual_context_vis_2_onsets, start_window, stop_window)
    odour_context_vis_1_running_tensor = get_running_tensor(downsampled_running_trace, odour_context_vis_1_onsets, start_window, stop_window)
    odour_context_vis_2_running_tensor = get_running_tensor(downsampled_running_trace, odour_context_vis_2_onsets, start_window, stop_window)

    # Create Regression Model
    activity_tensor_list = [visual_context_vis_1_tensor, visual_context_vis_2_tensor, odour_context_vis_1_tensor, odour_context_vis_2_tensor]
    region_tensor_list   = [visual_context_vis_1_region_tensor, visual_context_vis_2_region_tensor, odour_context_vis_1_region_tensor, odour_context_vis_2_region_tensor]
    running_tensor_list = [visual_context_vis_1_running_tensor, visual_context_vis_2_running_tensor, odour_context_vis_1_running_tensor, odour_context_vis_2_running_tensor]
    context_1_interaction_coef, context_2_interaction_coef, baseline_coef = create_regression_model(region_tensor_list, activity_tensor_list, running_tensor_list)

    np.save(os.path.join(base_directory, "Linear_Model_Baseline_Coef_V1.npy"), baseline_coef)
    np.save(os.path.join(base_directory, "Linear_Model_Context_1_Coef_V1.npy"), context_1_interaction_coef)
    np.save(os.path.join(base_directory, "Linear_Model_Context_2_Coef_V1.npy"), context_2_interaction_coef)


    # iew Maps

    """
    # Load Mask
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    figure_1 = plt.figure()
    visual_context_axis = figure_1.add_subplot(1, 3, 1)
    odour_context_axis = figure_1.add_subplot(1, 3, 2)
    difference_axis = figure_1.add_subplot(1, 3, 3)

    v1_visual_correlation_map_image = Widefield_General_Functions.create_image_from_data(context_1_interaction_coef, indicies, image_height, image_width)
    v1_odour_correlation_map_image = Widefield_General_Functions.create_image_from_data(context_2_interaction_coef, indicies, image_height, image_width)
    v1_difference_map_image = Widefield_General_Functions.create_image_from_data(difference_map, indicies, image_height, image_width)

    visual_context_axis.imshow(v1_visual_correlation_map_image, cmap='bwr')
    odour_context_axis.imshow(v1_odour_correlation_map_image, cmap='bwr')
    difference_axis.imshow(v1_difference_map_image, cmap='bwr')
    plt.show()
    """




    return context_1_interaction_coef, context_2_interaction_coef


controls = [
    #"/media/matthew/Seagate Expansion Drive1/Switching_Analysis/Wildtype/NRXN78.1A/2020_12_05_Switching_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Switching_Analysis/Wildtype/NRXN78.1A/2020_12_09_Switching_Imaging",

    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_02_Transition_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_08_Transition_Imaging",

    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_23_Transition_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_31_Transition_Imaging",

    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_15_Transition_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",

    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_10_29_Transition_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_11_05_Transition_Imaging",

    "/media/matthew/Seagate Expansion Drive1/Switching_Analysis/Wildtype/NXAK4.1B/2021_03_04_Switching_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Switching_Analysis/Wildtype/NXAK7.1B/2021_03_02_Switching_Imaging",




# "/media/matthew/Seagate Expansion Drive1/Switching_Analysis/Wildtype/NRXN78.1D/2020_11_29_Switching_Imaging"



"""
mutants = [
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_10_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK24.1C/2021_11_10_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive1/Transition_2/NXAK20.1B/2021_11_22_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Transition_2/NXAK20.1B/2021_11_24_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Transition_2/NXAK20.1B/2021_11_26_Transition_Imaging",
]
"""


start_window = 0
stop_window = 15

for base_directory in controls:
    print(base_directory)
    context_1_correlation, context_2_correlation = view_traces(base_directory, start_window, stop_window, view=False)


for base_directory in mutants:
    context_1_correlation, context_2_correlation = view_traces(base_directory, start_window, stop_window, view=False)
