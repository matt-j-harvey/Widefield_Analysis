import os

number_of_threads = 1
os.environ["OMP_NUM_THREADS"] = str(number_of_threads) # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = str(number_of_threads) # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = str(number_of_threads) # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(number_of_threads) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(number_of_threads) # export NUMEXPR_NUM_THREADS=1


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import fdrcorrection

import Create_Trial_Tensors
import Create_Analysis_Dataset_Learning
import Test_Significance_Mixed_Effects_Model

from Files import Session_List
from Widefield_Utils import widefield_utils

# Load Session Lists
pre_learning_session_list = Session_List.control_pre_learning_session_list
intermediate_learning_session_list = Session_List.control_intermediate_learning_session_list
post_learning_session_list = Session_List.control_post_learning_session_list


nrxn_78_1a = [

     # Pre
    ["NRXN78.1A/2020_11_15_Discrimination_Imaging"],

    # Int
    ["NRXN78.1A/2020_11_16_Discrimination_Imaging"],

    # Post
    ["NRXN78.1A/2020_11_17_Discrimination_Imaging",
    "NRXN78.1A/2020_11_19_Discrimination_Imaging",
    "NRXN78.1A/2020_11_21_Discrimination_Imaging"]

]


nrxn_78_1d = [

    # Pre
    ["NRXN78.1D/2020_11_15_Discrimination_Imaging"],

    # Int
    ["NRXN78.1D/2020_11_16_Discrimination_Imaging",
    "NRXN78.1D/2020_11_17_Discrimination_Imaging",
    "NRXN78.1D/2020_11_19_Discrimination_Imaging"],

    # Post
    ["NRXN78.1D/2020_11_21_Discrimination_Imaging",
    "NRXN78.1D/2020_11_23_Discrimination_Imaging",
    "NRXN78.1D/2020_11_25_Discrimination_Imaging"]

]

nxak_4_1b = [

    # Pre
    ["NXAK4.1B/2021_02_06_Discrimination_Imaging",
    "NXAK4.1B/2021_02_08_Discrimination_Imaging",
    "NXAK4.1B/2021_02_10_Discrimination_Imaging"],

    # Int
    ["NXAK4.1B/2021_02_12_Discrimination_Imaging"],

    # Post
    ["NXAK4.1B/2021_02_14_Discrimination_Imaging",
    "NXAK4.1B/2021_02_22_Discrimination_Imaging"]

]


nxak_14_1_a = [

    # Pre
    ["NXAK14.1A/2021_05_01_Discrimination_Imaging",
    "NXAK14.1A/2021_05_03_Discrimination_Imaging"],

    # Int
    [],

    # Post
    ["NXAK14.1A/2021_05_05_Discrimination_Imaging",
    "NXAK14.1A/2021_05_07_Discrimination_Imaging",
    "NXAK14.1A/2021_05_09_Discrimination_Imaging"]

]

nxak_22_1_a = [

    # Pre
    [],

    # Int
    ["NXAK22.1A/2021_09_29_Discrimination_Imaging",
    "NXAK22.1A/2021_10_01_Discrimination_Imaging",
    "NXAK22.1A/2021_10_03_Discrimination_Imaging",
    "NXAK22.1A/2021_10_05_Discrimination_Imaging"],

    # Post
    ["NXAK22.1A/2021_10_07_Discrimination_Imaging",
    "NXAK22.1A/2021_10_08_Discrimination_Imaging"],

]

nxak_7_1_b = [

    # Pre
    ["NXAK7.1B/2021_02_03_Discrimination_Imaging",
    "NXAK7.1B/2021_02_05_Discrimination_Imaging",
    "NXAK7.1B/2021_02_07_Discrimination_Imaging",
    "NXAK7.1B/2021_02_09_Discrimination_Imaging"],

    # Int
    ["NXAK7.1B/2021_02_15_Discrimination_Imaging",
    "NXAK7.1B/2021_02_17_Discrimination_Imaging",
    "NXAK7.1B/2021_02_19_Discrimination_Imaging",
    "NXAK7.1B/2021_02_22_Discrimination_Imaging"],

    # Post
    ["NXAK7.1B/2021_02_24_Discrimination_Imaging"]

]

control_nested_learning_session_list = [nrxn_78_1a, nrxn_78_1d, nxak_4_1b, nxak_14_1_a, nxak_22_1_a, nxak_7_1_b]


# Set Tensor Directory
data_root_directory = r"/media/matthew/Expansion/Control_Data"

# Select Analysis Details
#analysis_name = "Hits_Vis_1_Aligned"
#tensor_directory = r"//media/matthew/External_Harddrive_2/Control_Learning_Vis_Aligned"

analysis_name = "Hits_Lick_Aligned"
tensor_directory = r"//media/matthew/External_Harddrive_2/Control_Learning_lick_Aligned"


[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)

# Create Trial Tensors
print("creating trial tensors")
"""
for mouse in tqdm(control_nested_learning_session_list, leave=True, position=0, desc="Mouse"):
    for learning_stage in tqdm(mouse, leave=True, position=1, desc="Learning Stage"):
        for base_directory in tqdm(learning_stage, leave=True, position=1, desc="Session"):
            for onsets_file in onset_files:
                Create_Trial_Tensors.create_trial_tensor(os.path.join(data_root_directory, base_directory), onsets_file, start_window, stop_window, tensor_directory,
                            start_cutoff=3000,
                            ridge_regression_correct=False,
                            gaussian_filter=False,
                            baseline_correct=True,
                            align_within_mice=False,
                            align_across_mice=False,
                            extended_tensor=False,
                            mean_only=False,
                            stop_stimuli=None,
                            use_100_df=True,
                            z_score=False)

"""
#Create_Analysis_Dataset_Learning.create_analysis_dataset(tensor_directory, control_nested_learning_session_list, onset_files, analysis_name, start_window, stop_window)


Test_Significance_Mixed_Effects_Model.view_learning_raw_difference(tensor_directory, analysis_name, vmin=-0.05, vmax=0.05)