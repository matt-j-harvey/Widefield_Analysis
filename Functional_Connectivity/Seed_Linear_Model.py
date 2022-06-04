import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import sys
import scipy.signal
from sklearn.linear_model import LinearRegression

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions



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
        activity_tensor.append(trial_activity)
    
    activity_tensor = np.array(activity_tensor)

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

def get_noise_tensor(activity_tensor):

    mean_trace = np.mean(activity_tensor, axis=0)
    activity_tensor = np.subtract(activity_tensor, mean_trace)
    return activity_tensor


def create_regression_model(region_tensor, activity_tensor):

    number_of_trials = np.shape(region_tensor)[0]
    trial_length = np.shape(region_tensor)[1]
    total_number_of_timepoints = number_of_trials * trial_length

    # Create Stimuli Regressors
    stimuli_regressors = np.zeros((total_number_of_timepoints, trial_length))
    for trial_index in range(number_of_trials):
        trial_start = trial_index * trial_length
        trial_stop = trial_start + trial_length
        stimuli_regressors[trial_start:trial_stop] = np.identity(trial_length)
    print("stimuli Regressors", np.shape(stimuli_regressors))

    # Flatten Region Tensor
    region_tensor = flatten_region_tensor(region_tensor)
    region_tensor = np.expand_dims(region_tensor, 1)
    print("region 2 tensor shape", np.shape(region_tensor))

    design_matrix = np.hstack([stimuli_regressors, region_tensor])
    print("Design Matrix Shape", np.shape(design_matrix))

    # Flatten Activity Tensor
    number_of_pixels = np.shape(activity_tensor)[2]
    activity_tensor = np.reshape(activity_tensor, (number_of_trials * trial_length, number_of_pixels))

    # Create Model
    model = LinearRegression()
    model.fit(X=design_matrix, y=activity_tensor)

    # Get Coefs
    coefs = model.coef_
    interaction_coef = coefs[:, -1]

    return interaction_coef








def view_traces(base_directory, start_window, stop_window, context_1_onset_files, context_2_onset_files, view=True):

    v1 = [45, 46]
    m2 = [8,9]
    rsc = [28, 32]
    m1 = [11, 12]
    mv2 = [47, 48]
    selected_region = v1


    # Load Pixel assignments
    pixel_assignments = np.load("/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/Pixel_Assignmnets.npy")

    # Load Onsets
    context_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", context_1_onset_files))
    context_2_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", context_2_onset_files))

    # Load Activity Tensors
    context_1_activity_tensor = get_activity_tensor(base_directory, context_1_onsets, start_window, stop_window)
    context_2_activity_tensor = get_activity_tensor(base_directory, context_2_onsets, start_window, stop_window)

    # Get Region Tensors
    context_1_region_tensor = get_region_trace(context_1_activity_tensor, pixel_assignments, selected_region)
    context_2_region_tensor = get_region_trace(context_2_activity_tensor, pixel_assignments, selected_region)

    # Create Regression Model
    context_1_interaction_coef = create_regression_model(context_1_region_tensor, context_1_activity_tensor)
    context_2_interaction_coef = create_regression_model(context_2_region_tensor, context_2_activity_tensor)
    difference_map = np.subtract(context_1_interaction_coef, context_2_interaction_coef)

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

    visual_context_axis.imshow(v1_visual_correlation_map_image, cmap='jet')
    odour_context_axis.imshow(v1_odour_correlation_map_image, cmap='jet')
    difference_axis.imshow(v1_difference_map_image, cmap='bwr')
    plt.show()
    """




    return context_1_interaction_coef, context_2_interaction_coef


controls = [
            #"/media/matthew/Seagate Expansion Drive1/Switching_Analysis/Wildtype/NRXN78.1A/2020_12_05_Switching_Imaging",

            "/media/matthew/Seagate Expansion Drive1/Switching_Analysis/Wildtype/NRXN78.1D/2020_11_29_Switching",

            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_02_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_08_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_23_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_31_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_15_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_10_29_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_11_05_Transition_Imaging"
            ]
#pixel_assigments_image = list(np.load(r"C:\Users\matth\Documents\Allen_Atlas_Delta_F\Pixel_Assignmnets_Image.npy"))
#plt.imshow(pixel_assigments_image)
#plt.show()


start_window = -10
stop_window = 20
context_1_onset_files = "visual_context_stable_vis_2_onsets.npy"
context_2_onset_files = "odour_context_stable_vis_2_onsets.npy"

for base_directory in controls:
    context_1_correlation, conttext_2_correlation = view_traces(base_directory, start_window, stop_window, context_1_onset_files, context_2_onset_files, view=False)


for base_directory in mutants:
    context_1_correlation, conttext_2_correlation = view_traces(base_directory, start_window, stop_window, context_1_onset_files, context_2_onset_files, view=False)
