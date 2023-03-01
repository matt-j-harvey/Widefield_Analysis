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
    mouse_indicies = np.where(mouse_list == selected_mouse)
    mouse_data = activity_list[mouse_indicies] # Shape ( N Trials, N Timepoints, N Pixels)

    # Baseline Correct If Selected
    if baseline_correct == True:
        mouse_data = perform_baseline_correction(mouse_data, baseline_window)

    mouse_mean = np.mean(mouse_data, axis=0)

    return mouse_mean



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
    genotype_list = metadata_dataset[:, 0]
    unique_genotypes = np.unique(genotype_list)
    mouse_list = metadata_dataset[:, 1]

    # Get Average For Each Mouse - Will Be Shape: (Genotype x Mice x Timepoint x Pixels)
    mouse_genotype_average_list = []
    for genotype in unique_genotypes:

        genotype_indicies = np.where(genotype_list == genotype)[0]
        genotype_mice = mouse_list[genotype_indicies]
        genotype_mice = np.unique(genotype_mice)
        print("Genotype: ", genotype, "Genotype Mice", genotype_mice)

        mouse_average_list = []
        for selected_mouse in genotype_mice:
            mouse_mean = get_mouse_data(activity_list, metadata_dataset, selected_mouse, baseline_correct, baseline_window)
            mouse_average_list.append(mouse_mean)

        mouse_genotype_average_list.append(mouse_average_list)

    mouse_genotype_average_list = np.array(mouse_genotype_average_list)
    print("mouse_genotype_average_list", np.shape(mouse_genotype_average_list))


    # Get Average Across Mice - Will Be Shape: (Conditions x Timepoint x Pixels)
    genotype_1_average = np.mean(mouse_genotype_average_list[0], axis=0)
    genotype_2_average = np.mean(mouse_genotype_average_list[1], axis=0)
    genotype_average_list = np.array([genotype_1_average, genotype_2_average])

    # Save These
    save_directory = os.path.join(tensor_directory, "Average_Activity")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    np.save(os.path.join(save_directory, "Mouse_Genotype_Average_Matrix.npy"), mouse_genotype_average_list)
    np.save(os.path.join(save_directory, "Condition_Average_Matrix.npy"), genotype_average_list)

