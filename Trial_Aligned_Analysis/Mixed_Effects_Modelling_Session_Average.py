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



def split_trials_by_condition(activity_dataset, metata_dataset):

    condition_list = metata_dataset[:, 3]
    unique_conditions = np.unique(condition_list)
    print("Unique Conditions", unique_conditions)

    combined_activity_list = []

    for condition in unique_conditions:
        condition_indicies = np.where(condition_list == condition)[0]
        combined_activity_list.append(activity_dataset[condition_indicies])

    return combined_activity_list

def get_session_averages(mouse_activity_data, mouse_metadata):
    # group_index, mouse_index, session_index, condition_index

    session_list = mouse_metadata[:, 2]
    unique_sessions = np.unique(session_list)

    condition_1_session_averages = []
    condition_2_session_averages = []

    for session in unique_sessions:
        session_indicies = np.where(session_list == session)[0]
        session_activity = mouse_activity_data[session_indicies]
        session_metadata = mouse_metadata[session_indicies]

        [condition_1_activity, condition_2_activity] = split_trials_by_condition(session_activity, session_metadata)
        condition_1_session_averages.append(condition_1_activity)
        condition_2_session_averages.append(condition_2_activity)

    return condition_1_session_averages, condition_2_session_averages


def get_session_averages_for_all_mice(activity_dataset, metadata_dataset):

    # Load Session List
    mouse_list = metadata_dataset[:, 1]
    unique_mice = np.unique(mouse_list)

    condition_1_mouse_average_list = []
    condition_2_mouse_average_list = []

    for mouse in unique_mice:
        print("Mouse", mouse)

        mouse_indicies = np.where(mouse_list == mouse)[0]

        mouse_activity_data = activity_dataset[mouse_indicies]
        mouse_metadata = metadata_dataset[mouse_indicies]

        # Get Session Averages
        condition_1_session_averages, condition_2_session_averages = get_session_averages(mouse_activity_data, mouse_metadata)

        # Add To List
        condition_1_mouse_average_list.append(condition_1_session_averages)
        condition_2_mouse_average_list.append(condition_2_session_averages)

    return condition_1_mouse_average_list, condition_2_mouse_average_list






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
            session_data_condition_1 = np.mean(mouse_condition_1_sessions[session_index], axis=0)
            session_data_condition_2 = np.mean(mouse_condition_2_sessions[session_index], axis=0)
            print("Session Data Condition 1", np.shape(session_data_condition_1))

            mouse_list.append("M_" + str(mouse_index).zfill(3))
            condition_list.append("C_000")
            activity_list.append(session_data_condition_1)

            mouse_list.append(mouse_index)
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


def test_significance_window(tensor_directory, analysis_name, window, random_effects="mouse"):

    # Open Analysis Dataframe
    analysis_file = tables.open_file(os.path.join(tensor_directory, analysis_name + "_Trialwise_.h5"), mode="r")
    activity_dataset = analysis_file.root["Data"]
    metadata_dataset = analysis_file.root["Trial_Details"]
    activity_dataset = np.array(activity_dataset)
    metadata_dataset = np.array(metadata_dataset)
    number_of_trials, number_of_timepoints, number_of_pixels = np.shape(activity_dataset)

    # Get Timepoint Data
    timepoint_activity = activity_dataset[:, window]
    timepoint_activity = np.nanmean(timepoint_activity, axis=1)

    # Get Mouse Condition Averages
    condition_1_mouse_average_list, condition_2_mouse_average_list = get_session_averages_for_all_mice(timepoint_activity, metadata_dataset)

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
