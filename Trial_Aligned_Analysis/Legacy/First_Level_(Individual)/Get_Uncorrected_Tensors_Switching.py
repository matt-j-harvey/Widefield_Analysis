import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import Create_Activity_Tensor
import Create_Behaviour_Tensor
import Create_Activity_Video_Individual_Session
import Single_Session_Analysis_Utils



def uncorrected_workflow_single_mouse(base_directory, onsets_file_list, tensor_names, start_window, stop_window, selected_behaviour_traces, experiment_name, plot_titles, tensor_save_directory):

    # Check Workflow Directories
    video_directory = os.path.join(base_directory, "Response_Videos")
    video_save_directory = os.path.join(video_directory, experiment_name)
    plot_directory = os.path.join(base_directory, "Plots")
    current_plot_directory = os.path.join(plot_directory, experiment_name)

    Single_Session_Analysis_Utils.check_directory(video_directory)
    Single_Session_Analysis_Utils.check_directory(video_save_directory)
    Single_Session_Analysis_Utils.check_directory(plot_directory)
    Single_Session_Analysis_Utils.check_directory(current_plot_directory)

    # Create Activity Tensors
    print("Creating Activity Tensor")
    number_of_conditions = len(onsets_file_list)
    for condition_index in range(number_of_conditions):
        Create_Activity_Tensor.create_activity_tensor(base_directory, onsets_file_list[condition_index], start_window, stop_window, tensor_save_directory, ridge_correction=False)



session_list = [

    r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_05_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_09_Switching_Imaging",

    r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_29_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_05_Switching_Imaging",

    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_13_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_15_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_17_Switching_Imaging",

    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_31_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_02_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_04_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_08_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_10_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_12_Transition_Imaging",

    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_20_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_22_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_24_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_14_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_16_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_18_Transition_Imaging",

    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_17_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_19_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_23_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_30_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_06_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_08_Transition_Imaging",

    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_17_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_19_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_22_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_24_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_26_Transition_Imaging",

    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_14_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_20_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_26_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_05_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_08_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_10_Transition_Imaging",

    r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_21_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_23_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_11_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_13_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_15_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_17_Transition_Imaging",

    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_14_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_20_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_22_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_29_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_03_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_05_Transition_Imaging",

    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_02_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_04_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_06_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_02_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_08_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_10_Transition_Imaging",

    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_26_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_28_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_02_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_23_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_31_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_04_02_Transition_Imaging",

]


# Load Analysis Settings
analysis_name = "Unrewarded_Contextual_Modulation"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Single_Session_Analysis_Utils.load_analysis_container(analysis_name)
tensor_save_directory = r"//media/matthew/Expansion/Widefield_Analysis/Learning_Analysis"

for base_directory in tqdm(session_list):
    print(base_directory)
    uncorrected_workflow_single_mouse(base_directory, onset_files, tensor_names, start_window, stop_window, behaviour_traces, analysis_name, tensor_names, tensor_save_directory)

