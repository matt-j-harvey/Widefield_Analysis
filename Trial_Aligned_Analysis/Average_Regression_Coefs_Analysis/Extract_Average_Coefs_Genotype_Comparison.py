import matplotlib.pyplot as plt
import tables
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.gridspec as gridspec
import os

from Widefield_Utils import widefield_utils


def load_analysis_data(tensor_directory, analysis_name):

    # Open Analysis Dataframe
    analysis_file = tables.open_file(os.path.join(tensor_directory, analysis_name + "_Trialwise_.h5"), mode="r")
    activity_dataset = analysis_file.root["Data"]
    metadata_dataset = analysis_file.root["Trial_Details"]
    metadata_dataset = np.array(metadata_dataset)
    activity_dataset = np.array(activity_dataset)
    print("metadata_dataset", np.shape(metadata_dataset))
    print("activity_dataset", np.shape(activity_dataset))

    return activity_dataset, metadata_dataset




def get_mouse_data(activity_list, metadata_dataset, selected_mouse, baseline_correct, baseline_window):

    # Unpack Metadata
    mouse_list = metadata_dataset[:, 1]
    condition_list = metadata_dataset[:, 3]

    # Get Number Of Unique Conditions
    unique_conditions = np.unique(condition_list)

    # Get Mouse Data For Each Condition
    condition_data_list = []
    for condition in unique_conditions:
        condition_indicies = np.where((mouse_list == selected_mouse) & (condition_list == condition))
        condition_data = activity_list[condition_indicies] # Shape ( N Sessions, N Timepoints, N Pixels)

        # Baseline Correct If Selected
        if baseline_correct == True:
            condition_data = perform_baseline_correction(condition_data, baseline_window)

        mouse_condition_mean = np.mean(condition_data, axis=0)
        condition_data_list.append(mouse_condition_mean)

    return condition_data_list



def perform_baseline_correction(condition_data, baseline_window):

    # Shape N Mice, N Timepoints, N Pixels
    number_of_mice, number_of_timepoints, number_of_pixels = np.shape(condition_data)

    baseline_corrected_data = []
    for mouse_index in range(number_of_mice):
        mouse_activity = condition_data[mouse_index]
        mouse_baseline = mouse_activity[baseline_window]
        mouse_baseline = np.mean(mouse_baseline, axis=0)
        mouse_activity = np.subtract(mouse_activity, mouse_baseline)
        baseline_corrected_data.append(mouse_activity)

    baseline_corrected_data = np.array(baseline_corrected_data)

    return baseline_corrected_data



def extract_condition_averages(tensor_directory, analysis_name, baseline_correct=False, baseline_window=None):

    # group_index, mouse_index, session_index, condition_index

    # Load Analysis Data
    activity_list, metadata_dataset = load_analysis_data(tensor_directory, analysis_name)

    # Get List Of Unique Mice
    group_list = metadata_dataset[:, 0]
    mouse_list = metadata_dataset[:, 1]
    unique_groups = np.unique(group_list)
    print("Mouse List", mouse_list)

    # Get Average For Each Group - Will Be Shape: (Groups x Mice x Conditions x Timepoint x Pixels) at the selected condition
    mouse_average_list = []
    for group_index in unique_groups:
        group_indicies = np.where(group_list == group_index)[0]
        group_mice = np.unique(mouse_list[group_indicies])
        group_data = []
        for selected_mouse in group_mice:
            mouse_condition_mean_list = get_mouse_data(activity_list, metadata_dataset, selected_mouse, baseline_correct, baseline_window)
            group_data.append(mouse_condition_mean_list)
        mouse_average_list.append(group_data)

    # Create Mouse Average List
    mouse_average_list = np.array(mouse_average_list)
    print("Average List Shape", np.shape(mouse_average_list))

    # Get Average Across Mice For Each Genotype - Will Be Shape: (Genotype x Conditions x Timepoint x Pixels)
    genotype_average_list = np.mean(mouse_average_list, axis=1)
    print("Genotype Average List", np.shape(genotype_average_list))

    # Save These
    save_directory = os.path.join(tensor_directory, "Average_Coefs")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    np.save(os.path.join(save_directory, "Mouse_Condition_Average_Matrix.npy"), mouse_average_list)
    np.save(os.path.join(save_directory, "Genotype_Average_Matrix.npy"), genotype_average_list)
