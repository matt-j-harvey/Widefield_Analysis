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



def convert_list_of_ints_to_list_of_str(list_of_ints):

    list_of_strings = []
    for item in list_of_ints:
        list_of_strings.append(str(item))
    return list_of_strings




def repackage_data_into_dataframe(pixel_activity, pixel_metadata):

    # Combine_Into Dataframe
    dataframe = pd.DataFrame(dtype=np.float64)
    dataframe["Data_Value"] = pixel_activity
    dataframe["Condition"] = pixel_metadata[:, 3]
    dataframe["Group"] = convert_list_of_ints_to_list_of_str(pixel_metadata[:, 0])
    dataframe["Mouse"] = convert_list_of_ints_to_list_of_str(pixel_metadata[:, 1])
    dataframe["Session"] = convert_list_of_ints_to_list_of_str(pixel_metadata[:, 2])

    return dataframe


def mixed_effects_three_levels_random_slope_intercept(dataframe):

    model = Lmer('Data_Value ~ Condition + (1 + Condition|Mouse) + (1 + Condition|Session)', data=dataframe)

    results = model.fit(verbose=False)
    results = np.array(results)
    slope = results[1, 0]
    t_statistic = results[1, 5]
    p_value = results[1, 6]

    print("t stat", t_statistic)

    return slope, t_statistic, p_value




def mixed_effects_two_levels_random_slope_intercept(dataframe):

    model = Lmer('Data_Value ~ Condition + (1 + Condition|Mouse)', data=dataframe)

    results = model.fit(verbose=False)
    results = np.array(results)
    slope = results[1, 0]
    t_statistic = results[1, 5]
    p_value = results[1, 6]

    print("t stat", t_statistic)

    return slope, t_statistic, p_value





def test_significance_individual_timepoints(tensor_directory, analysis_name):

    """
    This Test Is Run Pixelwise - All Brains Must Be In Same Pixel Space

    :return:
    Tensor of P Values
    """

    """
    # Open Analysis Dataframe
    analysis_file = h5py.File(os.path.join(tensor_directory, analysis_name + ".hdf5"), "r")
    activity_dataset = analysis_file["Data"]
    metadata_dataset = analysis_file["metadata"]
    number_of_timepoints, number_of_trials, number_of_pixels = np.shape(activity_dataset)
    print("metadata_dataset", np.shape(metadata_dataset))
    """
    # Open Analysis Dataframe
    analysis_file = tables.open_file(os.path.join(tensor_directory, analysis_name + "_Trialwise_.h5"), mode="r")
    activity_dataset = analysis_file.root["Data"]
    metadata_dataset = analysis_file.root["Trial_Details"]

    # Create P and Slope Tensors
    p_value_tensor = np.ones((number_of_timepoints, number_of_pixels))
    slope_tensor = np.zeros((number_of_timepoints, number_of_pixels))

    for timepoint_index in tqdm(range(number_of_timepoints), position=0, desc="Timepoint"):

        # Get Timepoint Data
        timepoint_activity = activity_dataset[timepoint_index]

        for pixel_index in tqdm(range(number_of_pixels), position=1, desc="Pixel", leave=True):

            # Package Into Dataframe
            pixel_activity = timepoint_activity[:, pixel_index]
            pixel_dataframe = repackage_data_into_dataframe(pixel_activity, metadata_dataset)

            # Fit Mixed Effects Model
            p_value, slope = mixed_effects_random_slope_and_intercept(pixel_dataframe)
            p_value_tensor[timepoint_index, pixel_index] = p_value
            slope_tensor[timepoint_index, pixel_index] = slope


    # Save These Tensors
    np.save(os.path.join(tensor_directory, analysis_name + "_p_value_tensor.npy"), p_value_tensor)
    np.save(os.path.join(tensor_directory, analysis_name + "_slope_tensor.npy"), slope_tensor)



def test_significance_window(tensor_directory, analysis_name, window, random_effects="mouse"):

    """
    This Test Is Run Pixelwise - All Brains Must Be In Same Pixel Space

    :return:
    Tensor of P Values
    """

    # Open Analysis Dataframe
    analysis_file = tables.open_file(os.path.join(tensor_directory, analysis_name + "_Trialwise_.h5"), mode="r")
    activity_dataset = analysis_file.root["Data"]
    metadata_dataset = analysis_file.root["Trial_Details"]
    activity_dataset = np.array(activity_dataset)
    metadata_dataset = np.array(metadata_dataset)

    number_of_trials, number_of_timepoints, number_of_pixels = np.shape(activity_dataset)

    print("Number of timepoints", number_of_timepoints)
    print("number of pixels", number_of_pixels)
    print("number of trials", number_of_trials)

    # Create P and Slope Tensors
    p_value_tensor = np.ones(number_of_pixels)
    slope_tensor = np.zeros(number_of_pixels)
    t_stat_tensor = np.zeros(number_of_pixels)

    # Get Timepoint Data
    timepoint_activity = activity_dataset[:, window]
    print("Timepoint activity shape", np.shape(timepoint_activity))
    timepoint_activity = np.nanmean(timepoint_activity, axis=1)

    for pixel_index in tqdm(range(number_of_pixels), position=1, desc="Pixel", leave=False):

        # Package Into Dataframe
        pixel_activity = timepoint_activity[:, pixel_index]
        pixel_dataframe = repackage_data_into_dataframe(pixel_activity, metadata_dataset)

        # Fit Mixed Effects Model
        if random_effects == "mouse":
            slope, t_statistic, p_value = mixed_effects_two_levels_random_slope_intercept(pixel_dataframe)

        elif random_effects == "mouse_and_session":
            slope, t_statistic, p_value = mixed_effects_three_levels_random_slope_intercept(pixel_dataframe)


        p_value_tensor[pixel_index] = p_value
        slope_tensor[pixel_index] = slope
        t_stat_tensor[pixel_index]= t_statistic

    return p_value_tensor, slope_tensor, t_stat_tensor



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

    #group_index, mouse_index, session_index, condition_index

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


def get_mouse_averages(activity_dataset, metadata_dataset):

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

        # Get Mouse Averages
        condition_1_mouse_average = np.mean(condition_1_session_averages, axis=0)
        condition_2_mouse_average = np.mean(condition_2_session_averages, axis=0)

        # Add To List
        condition_1_mouse_average_list.append(condition_1_mouse_average)
        condition_2_mouse_average_list.append(condition_2_mouse_average)

    return condition_1_mouse_average_list, condition_2_mouse_average_list
