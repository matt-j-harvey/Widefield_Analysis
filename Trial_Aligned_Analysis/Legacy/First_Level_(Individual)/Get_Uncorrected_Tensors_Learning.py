import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from Widefield_Utils import Create_Activity_Tensor, widefield_utils
from Files import Session_List


def uncorrected_workflow_single_mouse(base_directory, onsets_file_list, tensor_names, start_window, stop_window, selected_behaviour_traces, experiment_name, plot_titles, tensor_save_directory):

    # Check Workflow Directories
    video_directory = os.path.join(base_directory, "Response_Videos")
    video_save_directory = os.path.join(video_directory, experiment_name)
    plot_directory = os.path.join(base_directory, "Plots")
    current_plot_directory = os.path.join(plot_directory, experiment_name)

    widefield_utils.check_directory(video_directory)
    widefield_utils.check_directory(video_save_directory)
    widefield_utils.check_directory(plot_directory)
    widefield_utils.check_directory(current_plot_directory)

    # Create Activity Tensors
    print("Creating Activity Tensor")
    number_of_conditions = len(onsets_file_list)
    for condition_index in range(number_of_conditions):
        Create_Activity_Tensor.create_activity_tensor(base_directory, onsets_file_list[condition_index], start_window, stop_window, tensor_save_directory, start_cutoff=3000, align_within_mice=True, align_across_mice=True)


def get_flat_session_list(nested_session_list):

    flat_session_list = []

    for mouse in nested_session_list:
        for condition in mouse:
            for session in condition:
                flat_session_list.append(session)

    return flat_session_list


session_list = Session_List.expanded_controls_learning_nested
session_list = get_flat_session_list(session_list)

session_list = Session_List.expanded_mutants_learning_nested
session_list = get_flat_session_list(session_list)
print("session list", session_list)
print(len(session_list))


# Load Analysis Settings
analysis_name = "Hits_Pre_Post_Learning_response"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)
tensor_save_directory = r"//media/matthew/Expansion/Widefield_Analysis/Learning_Analysis"

onset_files = ["Mixed_Effects_Distribution_Matched_Onsets.npy"]
tensor_names = ["Mixed_Effects_Hit_RT_Matched"]

for base_directory in tqdm(session_list):
    print(base_directory)
    uncorrected_workflow_single_mouse(base_directory, onset_files, tensor_names, start_window, stop_window, behaviour_traces, analysis_name, tensor_names, tensor_save_directory)

