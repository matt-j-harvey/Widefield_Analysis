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

    """
    # Create Behaviour Tensor
    print("Creating Behaviour Tensor")
    behaviour_tensor_dict_list = []
    for onsets_file in onsets_file_list:
        behaviour_tensor_dict = Create_Behaviour_Tensor.create_behaviour_tensor(base_directory, onsets_file, start_window, stop_window, selected_behaviour_traces)
        behaviour_tensor_dict_list.append(behaviour_tensor_dict)

    # Load Activity Tensors
    print("Loading Activity Tensors")
    indicies, image_height, image_width = Single_Session_Analysis_Utils.load_tight_mask()

    mean_activity_tensor_list = []
    for tensor_name in tensor_names:
        split_base_directory = Path(base_directory).parts
        mouse_name = split_base_directory[-2]
        session_name = split_base_directory[-1]
        tensor_filepath = os.path.join(tensor_save_directory, mouse_name, session_name, tensor_name + "_Activity_Tensor.npy")
        tensor = np.load(tensor_filepath)


        mean_tensor = np.mean(tensor, axis=0)

        mean_activity_tensor_list.append(mean_tensor)

    # View Individual Movie
    print("Creating Video")
    indicies, image_height, image_width = Single_Session_Analysis_Utils.load_tight_mask()
    Create_Activity_Video_Individual_Session.create_activity_video(indicies, image_height, image_width, mean_activity_tensor_list, start_window, stop_window, plot_titles, video_save_directory, behaviour_tensor_dict_list, selected_behaviour_traces, difference_conditions=difference_conditions)
    """




session_list = [
    r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_15_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_03_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_01_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_29_Discrimination_Imaging",

     r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_24_Discrimination_Imaging",
     r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_25_Discrimination_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_22_Discrimination_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_24_Discrimination_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_09_Discrimination_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging",

    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_14_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_04_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_02_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_02_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_30_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_22_Discrimination_Imaging",

     r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_09_Discrimination_Imaging",
     r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging",
     r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_14_Discrimination_Imaging",
     r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_15_Discrimination_Imaging",
     r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging",
     r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_08_Discrimination_Imaging",
]


post_learning_session_list = [

    r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_24_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_25_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_22_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_24_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_09_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging",

    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_09_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_14_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_15_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_08_Discrimination_Imaging",

]

# Load Analysis Settings
analysis_name = "Correct_Rejections_Response"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Single_Session_Analysis_Utils.load_analysis_container(analysis_name)
tensor_save_directory = r"//media/matthew/Expansion/Widefield_Analysis/Learning_Analysis"

for base_directory in tqdm(post_learning_session_list):
    print(base_directory)
    uncorrected_workflow_single_mouse(base_directory, onset_files, tensor_names, start_window, stop_window, behaviour_traces, analysis_name, tensor_names, tensor_save_directory)

