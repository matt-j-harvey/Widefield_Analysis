import matplotlib.pyplot as plt
import tables
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.gridspec as gridspec
import os

from Widefield_Utils import widefield_utils

#([group_index, mouse_index, session_index, condition_index])])

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

    #(group_index, mouse_index, session_index, condition_index)

    # Unpack Metadata
    mouse_list = metadata_dataset[:, 1]
    condition_list = metadata_dataset[:, 3]
    print("Condition List", condition_list)

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

    # Shape N Trials, N Timepoints, N Pixels
    number_of_trials, number_of_timepoints, number_of_pixels = np.shape(condition_data)
    print("condition_data", np.shape(condition_data))

    baseline_corrected_data = []
    for trial_index in range(number_of_trials):
        trial_activity = condition_data[trial_index]
        trial_baseline = trial_activity[baseline_window]
        trial_baseline = np.mean(trial_baseline, axis=0)
        trial_activity = np.subtract(trial_activity, trial_baseline)
        baseline_corrected_data.append(trial_activity)

    baseline_corrected_data = np.array(baseline_corrected_data)
    print("Baseline Corrected Data", np.shape(baseline_corrected_data))

    return baseline_corrected_data



def extract_condition_averages(tensor_directory, analysis_name, baseline_correct=False, baseline_window=None):

    #(group_index, mouse_index, session_index, condition_index)

    # Load Analysis Data
    activity_list, metadata_dataset = load_analysis_data(tensor_directory, analysis_name)

    # Get List Of Unique Mice
    mouse_list = metadata_dataset[:, 1]
    unique_mice = np.unique(mouse_list)

    # Get Average For Each Mouse - Will Be Shape: (Mice x Conditions x Timepoint x Pixels)
    mouse_average_list = []
    for selected_mouse in unique_mice:
        mouse_condition_mean_list = get_mouse_data(activity_list, metadata_dataset, selected_mouse, baseline_correct, baseline_window)
        print("Mouse Condition Mean List", np.shape(mouse_condition_mean_list))
        mouse_average_list.append(mouse_condition_mean_list)
    mouse_average_list = np.array(mouse_average_list)
    print("Mouse Average List", np.shape(mouse_average_list))

    # Get Average Across Mice - Will Be Shape: (Conditions x Timepoint x Pixels)
    condition_average_list = np.mean(mouse_average_list, axis=0)

    # Save These
    save_directory = os.path.join(tensor_directory, "Average_Activity")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    np.save(os.path.join(save_directory, "Mouse_Condition_Average_Matrix.npy"), mouse_average_list)
    np.save(os.path.join(save_directory, "Condition_Average_Matrix.npy"), condition_average_list)

