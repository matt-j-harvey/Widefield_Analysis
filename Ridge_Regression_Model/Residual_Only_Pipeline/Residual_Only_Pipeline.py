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

import Create_Behaviour_Design_Matrix
import Create_Behaviour_Regression_Tensors
from Ridge_Regression_Model import Ridge_Regression_Model_General

#import Visualise_Regression_Results
#from Files import Session_List








def run_full_model_pipeline(selected_session_list, analysis_name, data_root_directory, start_cutoff=3000):

    # Select Analysis Details
    # For 2 Seconds Pre
    # To 1.5 Seconds Post
    # -56 to 40
    # Vis 1 Visual - Stable
    # Vis 1 Odour - Stable
    # Vis 2 Visual - Stable
    # Vis 2 Odour - Stable

    [start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)
    print("Start Window", start_window)
    print("Stop Window", stop_window)


    # Create Full Behaviour Design Matricies
    for mouse in tqdm(selected_session_list, leave=True, position=0, desc="Design Matrix Mouse"):
        for base_directory in tqdm(mouse, leave=True, position=1, desc="Design Matrix Session"):
            Create_Behaviour_Design_Matrix.create_design_matrix(os.path.join(data_root_directory, base_directory))

    # Fit Models
    for mouse in tqdm(selected_session_list, leave=True, position=0, desc="Fitting Model Mouse"):
        for base_directory in tqdm(mouse, leave=True, position=1, desc="Fitting Model Session"):
            print("Base directory", base_directory)

            # Get Design Matricies For Regression
            behaviour_design_matrix, delta_f_prediction_matrix = Create_Behaviour_Regression_Tensors.create_behaviour_regression_tensors(analysis_name, os.path.join(data_root_directory, base_directory), start_cutoff)

            # Run Regression
            regression_save_directory = os.path.join(data_root_directory, base_directory, "Behaviour_Regression_Trials")
            Ridge_Regression_Model_General.fit_ridge_model(delta_f_prediction_matrix, behaviour_design_matrix, regression_save_directory, chunk_size=5000)

            # Visualise Regression
            #regression_dictionary = np.load(os.path.join(regression_save_directory, "Regression_Dictionary_Simple.npy"), allow_pickle=True)[()]
            #Visualise_Regression_Results.visualise_regression_results(base_directory, regression_dictionary, behaviour_design_matrix, design_matrix_key_dict, delta_f_matrix)



"""
Control_Switching_Sessions = [

        [r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_28_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_05_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_09_Switching_Imaging"],

        [r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_29_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_05_Switching_Imaging"],

        [r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_21_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_23_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_11_Switching_Imaging"],

        [r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_14_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_20_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_22_Switching_Imaging"],

        [r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_02_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_04_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_06_Switching_Imaging"],

        [r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_26_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_28_Switching_Imaging"]
]



# Set Directories
selected_session_list = Control_Switching_Sessions
data_root_diretory = r"/media/matthew/Expansion/Control_Data"
analysis_name = "Switching_Behaviour_Trials"

run_full_model_pipeline(selected_session_list, analysis_name, data_root_diretory)
"""



"""
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

selected_session_list = mutant_switching_only_sessions_nested
data_root_diretory = r"/media/matthew/External_Harddrive_1/Neurexin_Data"
#tensor_directory = r"/media/matthew/External_Harddrive_2/Regression_Modelling_Results/Mutant_Switching_Residual_Only"
analysis_name = "Switching_Behaviour_Trials"
"""

"""
mutant_session_tuples = [
    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_14_Discrimination_Imaging",
     r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_09_Discrimination_Imaging"],

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_04_Discrimination_Imaging",
     r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging"],

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_02_Discrimination_Imaging",
     r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_14_Discrimination_Imaging"],

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_02_Discrimination_Imaging",
     r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_15_Discrimination_Imaging"],

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_30_Discrimination_Imaging",
     r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging"],

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_22_Discrimination_Imaging",
     r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_08_Discrimination_Imaging"],

]


selected_session_list = mutant_session_tuples
data_root_diretory = r"/media/matthew/External_Harddrive_1/Neurexin_Data"
analysis_name = "Discrimination_Behaviour_Trials"
"""


"""
control_session_tuples = [

    [r"NRXN78.1A/2020_11_15_Discrimination_Imaging",
     r"NRXN78.1A/2020_11_24_Discrimination_Imaging"],

    [r"NRXN78.1D/2020_11_15_Discrimination_Imaging",
     r"NRXN78.1D/2020_11_25_Discrimination_Imaging"],

    [r"NXAK4.1B/2021_02_06_Discrimination_Imaging",
     r"NXAK4.1B/2021_02_22_Discrimination_Imaging"],

    [r"NXAK7.1B/2021_02_03_Discrimination_Imaging",
     r"NXAK7.1B/2021_02_24_Discrimination_Imaging"],

    [r"NXAK14.1A/2021_05_01_Discrimination_Imaging",
     r"NXAK14.1A/2021_05_09_Discrimination_Imaging"],

    [r"NXAK22.1A/2021_09_29_Discrimination_Imaging",
     r"NXAK22.1A/2021_10_08_Discrimination_Imaging"]

]


control_session_post = [

    [r"NRXN78.1A/2020_11_17_Discrimination_Imaging",
    r"NRXN78.1A/2020_11_19_Discrimination_Imaging",
    r"NRXN78.1A/2020_11_21_Discrimination_Imaging"],

    [r"NRXN78.1D/2020_11_21_Discrimination_Imaging",
    r"NRXN78.1D/2020_11_23_Discrimination_Imaging",
    r"NRXN78.1D/2020_11_25_Discrimination_Imaging"],

    [r"NXAK4.1B/2021_02_14_Discrimination_Imaging",
    r"NXAK4.1B/2021_02_22_Discrimination_Imaging"],

    [r"NXAK7.1B/2021_02_24_Discrimination_Imaging"],

    [r"NXAK14.1A/2021_05_05_Discrimination_Imaging",
    r"NXAK14.1A/2021_05_07_Discrimination_Imaging",
    r"NXAK14.1A/2021_05_09_Discrimination_Imaging"]

    [r"2021_10_07_Discrimination_Imaging",
    r"2021_10_08_Discrimination_Imaging"],

]


mutant_session_post =

selected_session_list = control_session_tuples
data_root_diretory = r"/media/matthew/Expansion/Control_Data"
analysis_name = "Discrimination_Behaviour_Trials"
"""

Control_Post_learning = [

    ["NRXN78.1A/2020_11_17_Discrimination_Imaging",
     "NRXN78.1A/2020_11_19_Discrimination_Imaging",
     "NRXN78.1A/2020_11_21_Discrimination_Imaging"],

    ["NRXN78.1D/2020_11_21_Discrimination_Imaging",
     "NRXN78.1D/2020_11_23_Discrimination_Imaging",
     "NRXN78.1D/2020_11_25_Discrimination_Imaging"],

    ["NXAK4.1B/2021_02_14_Discrimination_Imaging",
     "NXAK4.1B/2021_02_22_Discrimination_Imaging"],

    ["NXAK7.1B/2021_02_24_Discrimination_Imaging"],

    ["NXAK14.1A/2021_05_05_Discrimination_Imaging",
     "NXAK14.1A/2021_05_07_Discrimination_Imaging",
     "NXAK14.1A/2021_05_09_Discrimination_Imaging"],

    ["NXAK22.1A/2021_10_07_Discrimination_Imaging",
     "NXAK22.1A/2021_10_08_Discrimination_Imaging"]

]


Mutant_Post_Learning = [

    ["NRXN71.2A/2020_12_05_Discrimination_Imaging"],

    ["NXAK4.1A/2021_03_03_Discrimination_Imaging",
     "NXAK4.1A/2021_03_05_Discrimination_Imaging"],

    ["NXAK10.1A/2021_05_12_Discrimination_Imaging",
     "NXAK10.1A/2021_05_14_Discrimination_Imaging"],

    ["NXAK16.1B/2021_06_15_Discrimination_Imaging"],

    ["NXAK20.1B/2021_10_11_Discrimination_Imaging",
     "NXAK20.1B/2021_10_13_Discrimination_Imaging",
     "NXAK20.1B/2021_10_15_Discrimination_Imaging",
     "NXAK20.1B/2021_10_17_Discrimination_Imaging",
     "NXAK20.1B/2021_10_19_Discrimination_Imaging"],

    ["NXAK24.1C/2021_10_02_Discrimination_Imaging",
     "NXAK24.1C/2021_10_04_Discrimination_Imaging",
     "NXAK24.1C/2021_10_06_Discrimination_Imaging"]

]




Control_Pre_Learning = [

        ["/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_15_Discrimination_Imaging"],

        ["/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_04_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_06_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_08_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_10_Discrimination_Imaging"],

        ["/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_01_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_03_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_05_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_07_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_09_Discrimination_Imaging"],

        ["/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_04_29_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_01_Discrimination_Imaging",
        "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_03_Discrimination_Imaging"],

        ["/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_25_Discrimination_Imaging"],

]


Neurexin_Pre_Learning = [

        [#"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_13_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_14_Discrimination_Imaging",
        #"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_15_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_16_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_17_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_19_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_21_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_23_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_25_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_27_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_29_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_01_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_03_Discrimination_Imaging"],

        ["/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_02_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_04_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_06_Discrimination_Imaging"],

        ["/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_04_30_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_02_Discrimination_Imaging"],

        [#"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_04_30_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_02_Discrimination_Imaging",
        #"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_04_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_06_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_08_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_10_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_12_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_14_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_16_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_18_Discrimination_Imaging"],

        ["/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_28_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_30_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_02_Discrimination_Imaging"],

        ["/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_20_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_22_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_24_Discrimination_Imaging",
        "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_26_Discrimination_Imaging"],

]


"""
selected_session_list = Control_Pre_Learning
data_root_diretory = r"/media/matthew/Expansion/Control_Data"
analysis_name = "Discrimination_Behaviour_Trials"
run_full_model_pipeline(selected_session_list, analysis_name, data_root_diretory)
"""

selected_session_list = Neurexin_Pre_Learning
data_root_diretory = r"/media/matthew/External_Harddrive_1/Neurexin_Data"
analysis_name = "Discrimination_Behaviour_Trials"
run_full_model_pipeline(selected_session_list, analysis_name, data_root_diretory)