import os
import h5py
from tqdm import tqdm
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import tables
from datetime import datetime

from Widefield_Utils import widefield_utils




def get_session_averages(activity_dataset, metadata_dataset):

    # Load Session List
    session_list = metadata_dataset[:, 2]
    unique_sessions = np.unique(session_list)

    condition_1_session_average_list = []
    condition_2_session_average_list = []

    for session in unique_sessions:
        session_indicies = np.where(session_list == session)[0]

        session_trials = activity_dataset[session_indicies]
        session_metadata = metadata_dataset[session_indicies]

        [condition_1_trials, condition_2_trials] = split_trials_by_condition(session_trials, session_metadata)


        condition_1_mean = np.mean(condition_1_trials, axis=0)
        condition_2_mean = np.mean(condition_2_trials, axis=0)



        condition_1_session_average_list.append(condition_1_mean)
        condition_2_session_average_list.append(condition_2_mean)

    return condition_1_session_average_list, condition_2_session_average_list


def get_mouse_averages(activity_dataset, metadata_dataset):

    # Load Session List
    mouse_list = metadata_dataset[:, 1]
    unique_mice = np.unique(mouse_list)

    condition_1_mouse_average_list = []
    condition_2_mouse_average_list = []

    for mouse in unique_mice:

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



def split_trials_by_condition(activity_dataset, metata_dataset):

    condition_list = metata_dataset[:, 3]
    unique_conditions = np.unique(condition_list)

    combined_activity_list = []

    for condition in unique_conditions:
        condition_indicies = np.where(condition_list == condition)[0]
        combined_activity_list.append(activity_dataset[condition_indicies])

    return combined_activity_list




def get_mouse_session_averages(activity_dataset, metadata_dataset):

    # Load Session List
    mouse_list = metadata_dataset[:, 1]
    unique_mice = np.unique(mouse_list)

    condition_1_mouse_average_list = []
    condition_2_mouse_average_list = []

    for mouse in unique_mice:

        mouse_indicies = np.where(mouse_list == mouse)[0]

        mouse_activity_data = activity_dataset[mouse_indicies]
        mouse_metadata = metadata_dataset[mouse_indicies]

        # Get Session Averages
        condition_1_session_averages, condition_2_session_averages = get_session_averages(mouse_activity_data, mouse_metadata)

        # Add To List
        condition_1_mouse_average_list.append(condition_1_session_averages)
        condition_2_mouse_average_list.append(condition_2_session_averages)

    return condition_1_mouse_average_list, condition_2_mouse_average_list