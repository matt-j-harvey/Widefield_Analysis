import os
import h5py
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tables
from datetime import datetime
from pymer4.models import Lmer

from Widefield_Utils import widefield_utils


def get_mouse_averages_baseline_correct(activity_list, metadata_dataset, condition_1, condition_2, baseline_window, response_window):

    condition_1_nested_mouse_list = []
    condition_2_nested_mouse_list = []

    mouse_list = metadata_dataset[:, 1]
    unique_mice = np.unique(mouse_list)

    for selected_mouse in unique_mice:
        mouse_condition_1_data, mouse_condition_2_data = get_mouse_data_baseline_correct(activity_list, metadata_dataset, selected_mouse, condition_1, condition_2, baseline_window, response_window)
        condition_1_nested_mouse_list.append(mouse_condition_1_data)
        condition_2_nested_mouse_list.append(mouse_condition_2_data)

    return condition_1_nested_mouse_list, condition_2_nested_mouse_list


def baseline_correct_data(mouse_condition_data, baseline_window, response_window):

    corrected_data = []

    # Baseline Correct
    for session in mouse_condition_data:
        print("Session Shape", np.shape(session))
        session_baseline = session[baseline_window]
        session_baseline = np.mean(session_baseline, axis=0)
        session_response = session[response_window]
        session_response = np.mean(session_response, axis=0)
        session_response = np.subtract(session_response, session_baseline)
        corrected_data.append(session_response)

    return corrected_data


def get_mouse_data_baseline_correct(activity_list, metadata_dataset, selected_mouse, condition_1, condition_2, baseline_window, response_window):

    # Unpack Metadata
    mouse_list = metadata_dataset[:, 1]
    condition_list = metadata_dataset[:, 2]

    # Get Mouse Sessions
    condition_1_mouse_indicies = np.where((mouse_list == selected_mouse) & (condition_list == condition_1))
    condition_2_mouse_indicies = np.where((mouse_list == selected_mouse) & (condition_list == condition_2))

    # Get Mouse Data
    mouse_condition_1_data = activity_list[condition_1_mouse_indicies]
    mouse_condition_2_data = activity_list[condition_2_mouse_indicies]

    # Baseline Correct
    mouse_condition_1_data = baseline_correct_data(mouse_condition_1_data, baseline_window, response_window)
    mouse_condition_2_data = baseline_correct_data(mouse_condition_2_data, baseline_window, response_window)

    return mouse_condition_1_data, mouse_condition_2_data






def repackage_data_into_dataframe(activity_list, mouse_list, condition_list):

    # Combine_Into Dataframe
    dataframe = pd.DataFrame(dtype=np.float64)
    dataframe["Data_Value"] = activity_list
    dataframe["Condition"] = condition_list
    dataframe["Mouse"] = mouse_list

    return dataframe


def prepare_data_for_dataframe(condition_1_mouse_average_list, condition_2_mouse_average_list):

    # Create Dataframe
    activity_list = []
    mouse_list = []
    condition_list = []

    number_of_mice = len(condition_1_mouse_average_list)
    for mouse_index in range(number_of_mice):
        mouse_condition_1_sessions = condition_1_mouse_average_list[mouse_index]
        mouse_condition_2_sessions = condition_2_mouse_average_list[mouse_index]
        number_of_sessions = len(mouse_condition_1_sessions)

        for session_index in range(number_of_sessions):
            session_data_condition_1 = mouse_condition_1_sessions[session_index]
            session_data_condition_2 = mouse_condition_2_sessions[session_index]
            print("Session Data Condition 1", np.shape(session_data_condition_1))

            mouse_list.append("M_" + str(mouse_index).zfill(3))
            condition_list.append("C_000")
            activity_list.append(session_data_condition_1)

            mouse_list.append("M_" + str(mouse_index).zfill(3))
            condition_list.append("C_001")
            activity_list.append(session_data_condition_2)

    activity_list = np.array(activity_list)
    return activity_list, mouse_list, condition_list




def mixed_effects_two_levels_random_slope_intercept(dataframe):

    model = Lmer('Data_Value ~ Condition + (1 + Condition|Mouse)', data=dataframe)

    results = model.fit(verbose=False)
    results = np.array(results)
    slope = results[1, 0]
    t_statistic = results[1, 5]
    p_value = results[1, 6]

    print("t stat", t_statistic)

    return slope, t_statistic, p_value




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


def test_significance_window(tensor_directory, analysis_name, condition_1, condition_2, baseline_window, response_window):

    # Open Analysis Dataframe
    activity_dataset, metadata_dataset = load_analysis_data(tensor_directory, analysis_name)
    number_of_trials, number_of_timepoints, number_of_pixels = np.shape(activity_dataset)

    # Get Mouse Condition Averages
    condition_1_mouse_average_list, condition_2_mouse_average_list = get_mouse_averages_baseline_correct(activity_dataset, metadata_dataset, condition_1, condition_2, baseline_window, response_window)

    # Prepare For DF
    activity_list, mouse_list, condition_list = prepare_data_for_dataframe(condition_1_mouse_average_list, condition_2_mouse_average_list)
    print("Activity list shape", np.shape(activity_list))

    # Create P and Slope Tensors
    p_value_tensor = np.ones(number_of_pixels)
    slope_tensor = np.zeros(number_of_pixels)
    t_stat_tensor = np.zeros(number_of_pixels)

    for pixel_index in tqdm(range(number_of_pixels), position=1, desc="Pixel", leave=False):

        # Package Into Dataframe
        pixel_activity = activity_list[:, pixel_index]
        pixel_dataframe = repackage_data_into_dataframe(pixel_activity, mouse_list, condition_list)

        # Fit Mixed Effects Model
        slope, t_statistic, p_value = mixed_effects_two_levels_random_slope_intercept(pixel_dataframe)

        # Add To Tensors
        p_value_tensor[pixel_index] = p_value
        slope_tensor[pixel_index] = slope
        t_stat_tensor[pixel_index]= t_statistic

    return p_value_tensor, slope_tensor, t_stat_tensor
