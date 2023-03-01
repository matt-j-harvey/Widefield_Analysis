import os

number_of_threads = 2
os.environ["OMP_NUM_THREADS"] = str(number_of_threads) # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = str(number_of_threads) # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = str(number_of_threads) # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(number_of_threads) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(number_of_threads) # export NUMEXPR_NUM_THREADS=1

import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from Widefield_Utils import widefield_utils


import Create_Trial_Tensors
import Create_Analysis_Dataset
import Extract_Averages
import Visualise_Average_Activity

from Files import Session_List



def run_residual_only_model_pipeline(selected_session_list, analysis_name, data_root_diretory, tensor_save_directory, start_cutoff=3000):

    # Select Analysis Details
    # For 2 Seconds Pre
    # To 1.5 Seconds Post
    # -56 to 40

    [start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)
    print("Start Window", start_window)
    print("Stop Window", stop_window)

    print("Onset Files", onset_files)

    """
    # Create Trial Tensors
    for mouse in tqdm(selected_session_list, leave=True, position=0, desc="Trial Tensors Mouse"):
        for base_directory in tqdm(mouse, leave=True, position=1, desc="Trial Tensors Session"):
            for onsets_file in onset_files:
                Create_Trial_Tensors.create_trial_tensor(os.path.join(data_root_diretory, base_directory), onsets_file, start_window, stop_window, tensor_save_directory,
                                    start_cutoff=start_cutoff,
                                    ridge_regression_correct=True,
                                    gaussian_filter=False,
                                    baseline_correct=False,
                                    align_within_mice=False,
                                    align_across_mice=False,
                                    extended_tensor=False,
                                    mean_only=False,
                                    stop_stimuli=None,
                                    use_100_df=True)

    # Create Activity Dataset
    nested_session_list = [selected_session_list]
    Create_Analysis_Dataset.create_analysis_dataset(tensor_save_directory, nested_session_list, onset_files, analysis_name, start_window, stop_window)
    """

    # Extract Averages
    baseline_window = list(range(0, 14))
    #baseline_window= list(range(69-14, 69))
    #Extract_Averages.extract_condition_averages(tensor_directory, analysis_name, baseline_correct=True, baseline_window=baseline_window)


    # Visualise Averages
    condition_1_index = 1
    condition_2_index = 3
    comparison_name = "Unrewarded_Contextual_Modulation"
    Visualise_Average_Activity.view_average_difference(tensor_directory, condition_1_index, condition_2_index, comparison_name, start_window, stop_window, vmin=0, vmax=0.012, diff_magnitude=0.01)
    #Visualise_Average_Activity.view_average_difference_per_mouse(tensor_directory, condition_1_index, condition_2_index, comparison_name, vmin=-0.05, vmax=0.05)

    # Plot Regions


    # Test Region Significance




Control_Switching_Sessions = [

        [r"NRXN78.1A/2020_11_28_Switching_Imaging",
        r"NRXN78.1A/2020_12_05_Switching_Imaging",
        r"NRXN78.1A/2020_12_09_Switching_Imaging"],

        [r"NRXN78.1D/2020_12_07_Switching_Imaging",
        r"NRXN78.1D/2020_11_29_Switching_Imaging",
        r"NRXN78.1D/2020_12_05_Switching_Imaging"],

        [r"NXAK14.1A/2021_05_21_Switching_Imaging",
        r"NXAK14.1A/2021_05_23_Switching_Imaging",
        r"NXAK14.1A/2021_06_11_Switching_Imaging"],

        [r"NXAK22.1A/2021_10_14_Switching_Imaging",
        r"NXAK22.1A/2021_10_20_Switching_Imaging",
        r"NXAK22.1A/2021_10_22_Switching_Imaging"],

        [r"NXAK4.1B/2021_03_02_Switching_Imaging",
        r"NXAK4.1B/2021_03_04_Switching_Imaging",
        r"NXAK4.1B/2021_03_06_Switching_Imaging"],

        [r"NXAK7.1B/2021_02_26_Switching_Imaging",
        r"NXAK7.1B/2021_02_28_Switching_Imaging"]
]





mutant_switching_only_sessions_nested = [

    [r"NRXN71.2A/2020_12_13_Switching_Imaging",
    r"NRXN71.2A/2020_12_15_Switching_Imaging",
    r"NRXN71.2A/2020_12_17_Switching_Imaging"],

    [r"NXAK4.1A/2021_03_31_Switching_Imaging",
    r"NXAK4.1A/2021_04_02_Switching_Imaging",
    r"NXAK4.1A/2021_04_04_Switching_Imaging"],

    [r"NXAK10.1A/2021_05_20_Switching_Imaging",
    r"NXAK10.1A/2021_05_22_Switching_Imaging",
    r"NXAK10.1A/2021_05_24_Switching_Imaging"],

    [r"NXAK16.1B/2021_06_17_Switching_Imaging",
    r"NXAK16.1B/2021_06_19_Switching_Imaging",
    r"NXAK16.1B/2021_06_23_Switching_Imaging"],

    [r"NXAK20.1B/2021_11_15_Switching_Imaging",
    r"NXAK20.1B/2021_11_17_Switching_Imaging",
    r"NXAK20.1B/2021_11_19_Switching_Imaging"],

    [r"NXAK24.1C/2021_10_14_Switching_Imaging",
    r"NXAK24.1C/2021_10_20_Switching_Imaging",
    r"NXAK24.1C/2021_10_26_Switching_Imaging"],

]


# Set Directories
selected_session_list = Control_Switching_Sessions
data_root_diretory = r"/media/matthew/Expansion/Control_Data"
tensor_directory = r"/media/matthew/External_Harddrive_2/Regression_Modelling_Results/Control_Switching_Residual_Only"
analysis_name = "Full_Model"
"""

selected_session_list = mutant_switching_only_sessions_nested
data_root_diretory = r"/media/matthew/External_Harddrive_1/Neurexin_Data"
tensor_directory = r"/media/matthew/External_Harddrive_2/Regression_Modelling_Results/Mutant_Switching_Residual_Only"
analysis_name = "Full_Model"
"""

run_residual_only_model_pipeline(selected_session_list, analysis_name, data_root_diretory, tensor_directory)



