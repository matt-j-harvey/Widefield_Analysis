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
import Create_Analysis_Dataset_Learning
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

    print("Data Root Directory", data_root_diretory)

    print("Onset Files", onset_files)
    """
    # Create Trial Tensors
    for mouse in tqdm(selected_session_list, leave=True, position=0, desc="Trial Tensors Mouse"):
        for learning_stage in mouse:
            for base_directory in tqdm(learning_stage, leave=True, position=1, desc="Trial Tensors Session"):
                for onsets_file in onset_files:

                    full_base_directory_path = os.path.join(data_root_diretory, base_directory)
                    print("Full base directory path", full_base_directory_path)
                    Create_Trial_Tensors.create_trial_tensor(full_base_directory_path, onsets_file, start_window, stop_window, tensor_save_directory,
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
    Create_Analysis_Dataset_Learning.create_analysis_dataset(tensor_save_directory, nested_session_list, onset_files, analysis_name, start_window, stop_window)


    # Extract Averages
    #baseline_window = list(range(0, 3))
    baseline_window = list(range(0, 14))
    #baseline_window= list(range(69-14, 69))
    Extract_Averages.extract_condition_averages(tensor_save_directory, analysis_name, baseline_correct=True, baseline_window=baseline_window)
    """

    # Visualise Averages
    condition_1_index = 1
    condition_2_index = 0
    comparison_name = "Learning_Activity_Changes"
    Visualise_Average_Activity.view_average_difference(tensor_save_directory, condition_1_index, condition_2_index, comparison_name, start_window, stop_window, vmin=-0, vmax=0.012, diff_magnitude=0.006)
    #Visualise_Average_Activity.view_average_difference_per_mouse(tensor_save_directory, condition_1_index, condition_2_index, comparison_name, vmin=-0.05, vmax=0.05)

    # Plot Regions


    # Test Region Significance





Control_Learning_Tuples = [

        # NRXN78.1A
        [
            # Pre
            ["NRXN78.1A/2020_11_15_Discrimination_Imaging"],

            # Post
            ["NRXN78.1A/2020_11_17_Discrimination_Imaging",
             "NRXN78.1A/2020_11_19_Discrimination_Imaging",
             "NRXN78.1A/2020_11_21_Discrimination_Imaging"],
        ],

        # NRXN78.1D
        [
            # Pre
            ["NRXN78.1D/2020_11_15_Discrimination_Imaging"],

            # Post
            ["NRXN78.1D/2020_11_21_Discrimination_Imaging",
             "NRXN78.1D/2020_11_23_Discrimination_Imaging",
             "NRXN78.1D/2020_11_25_Discrimination_Imaging"],
        ],

        # NXAK4.1B
        [
            # Pre
            [
             #"NXAK4.1B/2021_02_04_Discrimination_Imaging", not full running
             #"NXAK4.1B/2021_02_06_Discrimination_Imaging", not full running
             "NXAK4.1B/2021_02_08_Discrimination_Imaging",
             "NXAK4.1B/2021_02_10_Discrimination_Imaging"],

            # Post
            ["NXAK4.1B/2021_02_14_Discrimination_Imaging",
             "NXAK4.1B/2021_02_22_Discrimination_Imaging"],
        ],

        # NXAK7.1B
        [
            # Pre
            [
            #"NXAK7.1B/2021_02_01_Discrimination_Imaging", not full running
             "NXAK7.1B/2021_02_03_Discrimination_Imaging",
             "NXAK7.1B/2021_02_05_Discrimination_Imaging",
             "NXAK7.1B/2021_02_07_Discrimination_Imaging",
             "NXAK7.1B/2021_02_09_Discrimination_Imaging"],

            # Post
            ["NXAK7.1B/2021_02_24_Discrimination_Imaging"],
        ],

        # NXAK14.1A
        [
            # Pre
            ["NXAK14.1A/2021_04_29_Discrimination_Imaging",
             "NXAK14.1A/2021_05_01_Discrimination_Imaging",
             "NXAK14.1A/2021_05_03_Discrimination_Imaging"],

            # Post
            ["NXAK14.1A/2021_05_05_Discrimination_Imaging",
             "NXAK14.1A/2021_05_07_Discrimination_Imaging",
             "NXAK14.1A/2021_05_09_Discrimination_Imaging"],
        ],


        # NXAK22.1A
        [
            # Pre
            ["NXAK22.1A/2021_09_25_Discrimination_Imaging"],

            # Post
            ["NXAK22.1A/2021_10_07_Discrimination_Imaging",
             "NXAK22.1A/2021_10_08_Discrimination_Imaging"],
        ],
]

"""
selected_session_list = Control_Learning_Tuples
data_root_directory = r"/media/matthew/Expansion/Control_Data"
tensor_save_directory = r"/media/matthew/External_Harddrive_2/Regression_Modelling_Results/Control_Learning_D_Prime_Split_Vis_Aligned"
"""


## Mutants




Neurexin_Learning_Tuples = [

    # NRXN71.2A
    [
        # Pre
        [
        #"NRXN71.2A/2020_11_14_Discrimination_Imaging", Not Full Running
        "NRXN71.2A/2020_11_16_Discrimination_Imaging",
        "NRXN71.2A/2020_11_17_Discrimination_Imaging",
        "NRXN71.2A/2020_11_19_Discrimination_Imaging",
        "NRXN71.2A/2020_11_21_Discrimination_Imaging",
        "NRXN71.2A/2020_11_23_Discrimination_Imaging",
        "NRXN71.2A/2020_11_25_Discrimination_Imaging",
        "NRXN71.2A/2020_11_27_Discrimination_Imaging",
        "NRXN71.2A/2020_11_29_Discrimination_Imaging",
        "NRXN71.2A/2020_12_01_Discrimination_Imaging",
        "NRXN71.2A/2020_12_03_Discrimination_Imaging"],

        # Post
        ["NRXN71.2A/2020_12_05_Discrimination_Imaging"],
    ],


    # NXAK4.1A
    [
        # Pre
        [
        #"NXAK4.1A/2021_02_02_Discrimination_Imaging", Not Full Running
        "NXAK4.1A/2021_02_04_Discrimination_Imaging",
        "NXAK4.1A/2021_02_06_Discrimination_Imaging"],

        # Post
        ["NXAK4.1A/2021_03_03_Discrimination_Imaging",
         "NXAK4.1A/2021_03_05_Discrimination_Imaging"],
    ],


    # NXAK10.1A
    [
        # Pre
        ["NXAK10.1A/2021_04_30_Discrimination_Imaging",
        "NXAK10.1A/2021_05_02_Discrimination_Imaging"],

        # Post
        ["NXAK10.1A/2021_05_12_Discrimination_Imaging",
         "NXAK10.1A/2021_05_14_Discrimination_Imaging"],
    ],


    # NXAK16.1B
    [
        # Pre
        ["NXAK16.1B/2021_05_02_Discrimination_Imaging",
         "NXAK16.1B/2021_05_06_Discrimination_Imaging",
         "NXAK16.1B/2021_05_08_Discrimination_Imaging",
         "NXAK16.1B/2021_05_10_Discrimination_Imaging",
         "NXAK16.1B/2021_05_12_Discrimination_Imaging",
         "NXAK16.1B/2021_05_14_Discrimination_Imaging",
         "NXAK16.1B/2021_05_16_Discrimination_Imaging",
         "NXAK16.1B/2021_05_18_Discrimination_Imaging"],

        # Post
        ["NXAK16.1B/2021_06_15_Discrimination_Imaging"],
    ],


    # NXAK20.1B
    [
        # Pre
        ["NXAK20.1B/2021_09_28_Discrimination_Imaging",
        "NXAK20.1B/2021_09_30_Discrimination_Imaging",
        "NXAK20.1B/2021_10_02_Discrimination_Imaging"],

        # Post
        ["NXAK20.1B/2021_10_11_Discrimination_Imaging",
         "NXAK20.1B/2021_10_13_Discrimination_Imaging",
         "NXAK20.1B/2021_10_15_Discrimination_Imaging",
         "NXAK20.1B/2021_10_17_Discrimination_Imaging",
         "NXAK20.1B/2021_10_19_Discrimination_Imaging"],
    ],


    # NXAK24.1V
    [
        # Pre
        [
        #"NXAK24.1C/2021_09_20_Discrimination_Imaging", Not Full Running
        "NXAK24.1C/2021_09_22_Discrimination_Imaging",
        "NXAK24.1C/2021_09_24_Discrimination_Imaging",
        "NXAK24.1C/2021_09_26_Discrimination_Imaging"],

        # Post
        ["NXAK24.1C/2021_10_02_Discrimination_Imaging",
         "NXAK24.1C/2021_10_04_Discrimination_Imaging",
         "NXAK24.1C/2021_10_06_Discrimination_Imaging"],
    ],

]


selected_session_list = Neurexin_Learning_Tuples
data_root_directory = "/media/matthew/External_Harddrive_1/Neurexin_Data"
tensor_save_directory = r"/media/matthew/External_Harddrive_2/Regression_Modelling_Results/Neurexin_Learning_Full_Prestim"


"""
selected_session_list = Control_Learning_Tuples
data_root_directory = "/media/matthew/Expansion/Control_Data"
tensor_save_directory = r"/media/matthew/External_Harddrive_2/Regression_Modelling_Results/Control_Learning_Full_Prestim"
"""


analysis_name = "Hits_Vis_1_Aligned_Post"
run_residual_only_model_pipeline(selected_session_list, analysis_name, data_root_directory, tensor_save_directory)