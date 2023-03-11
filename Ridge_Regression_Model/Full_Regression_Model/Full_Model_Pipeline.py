import os

number_of_threads = 3
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

from Trial_Aligned_Analysis import Create_Trial_Tensors
from Ridge_Regression_Model import Ridge_Regression_Model_General
import Create_Full_Model_Design_Matrix
import Visualise_Regression_Results
from Files import Session_List


def flatten_tensor(tensor):
    n_trial, trial_length, n_var = np.shape(tensor)
    tensor = np.reshape(tensor, (n_trial * trial_length, n_var))
    return tensor


def create_delta_f_matrix(tensor_directory, session, onset_file_list):

    delta_f_matrix = []
    for condition in onset_file_list:

        # Get Tensor Name
        tensor_name = condition.replace("_onsets.npy", "")
        tensor_name = tensor_name.replace("_onset_frames.npy", "")

        # Open Trial Tensor
        session_trial_tensor_dict_path = os.path.join(tensor_directory, session, tensor_name)
        with open(session_trial_tensor_dict_path + ".pickle", 'rb') as handle:
            session_trial_tensor_dict = pickle.load(handle)
            activity_tensor = session_trial_tensor_dict["activity_tensor"]
            activity_tensor = flatten_tensor(activity_tensor)

        # Add To List
        delta_f_matrix.append(activity_tensor)

    delta_f_matrix = np.vstack(delta_f_matrix)

    return delta_f_matrix


def run_full_model_pipeline(selected_session_list, analysis_name, data_root_diretory, tensor_save_directory, start_cutoff=3000):

    # Select Analysis Details
    # For 2 Seconds Pre
    # To 1.5 Seconds Post
    # -56 to 40
    # Vis 1 Visual - Stable
    # Vis 1 Odour - Stable
    # Vis 2 Visual - Stable
    # Vis 2 Odour - Stable

    [start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)
    start_window = int(start_window)
    stop_window = int(stop_window)
    print("Start Window", start_window)
    print("Stop Window", stop_window)


    # Create Full Design Matricies
    for mouse in tqdm(selected_session_list, leave=True, position=0, desc="Design Matrix Mouse"):
        for base_directory in tqdm(mouse, leave=True, position=1, desc="Design Matrix Session"):
            Create_Full_Model_Design_Matrix.create_full_model_design_matrix(base_directory, data_root_diretory, tensor_save_directory, start_window, stop_window, onset_files, start_cutoff=3000,
                                                                                                                                                                                lick_lags=14,
                                                                                                                                                                                running_lags=14,
                                                                                                                                                                                n_face_components=360)

    # Create Trial Tensors
    for mouse in tqdm(selected_session_list, leave=True, position=0, desc="Trial Tensors Mouse"):
        for base_directory in tqdm(mouse, leave=True, position=1, desc="Trial Tensors Session"):
            for onsets_file in onset_files:
                Create_Trial_Tensors.create_trial_tensor(os.path.join(data_root_diretory, base_directory), onsets_file, start_window, stop_window, tensor_save_directory,
                                    start_cutoff=start_cutoff,
                                    ridge_regression_correct=False,
                                    gaussian_filter=False,
                                    baseline_correct=False,
                                    align_within_mice=False,
                                    align_across_mice=False,
                                    extended_tensor=False,
                                    mean_only=False,
                                    stop_stimuli=None,
                                    use_100_df=True)


    # Fit Models
    for mouse in tqdm(selected_session_list, leave=True, position=0, desc="Fitting Model Mouse"):
        for base_directory in tqdm(mouse, leave=True, position=1, desc="Fitting Model Session"):

            print("Base directory", base_directory)

            # Load Design Matrix
            design_matrix = np.load(os.path.join(tensor_save_directory, base_directory, "Full_Model_Design_Matrix.npy"))
            print("Design Matrix Shape", np.shape(design_matrix))

            # Load Delta F Matrix
            delta_f_matrix = create_delta_f_matrix(tensor_save_directory, base_directory, onset_files)
            print("Delta F Matrix Shape", np.shape(delta_f_matrix))

            # Save Delta F Matrix
            np.save(os.path.join(tensor_save_directory, base_directory, "Full_Model_Delta_F_Matrix.npy"), delta_f_matrix)

            # Run Regression
            Ridge_Regression_Model_General.fit_ridge_model(delta_f_matrix, design_matrix, os.path.join(tensor_save_directory, base_directory), chunk_size=5000)


    # Visualise Results
    for mouse in tqdm(selected_session_list, leave=True, position=0, desc="Mouse"):
      for base_directory in tqdm(mouse, leave=True, position=1, desc="Session"):

          base_directory = os.path.join(tensor_save_directory, base_directory)
          regression_dictionary = np.load(os.path.join(base_directory, "Regression_Dictionary_Simple.npy"), allow_pickle=True)[()]
          design_matrix = np.load(os.path.join(base_directory, "Full_Model_Design_Matrix.npy"))
          design_matrix_key_dict = np.load(os.path.join(base_directory, "design_matrix_key_dict.npy"), allow_pickle=True)[()]
          delta_f_matrix = np.load(os.path.join(base_directory, "Full_Model_Delta_F_Matrix.npy"))
          Visualise_Regression_Results.visualise_regression_results(base_directory, regression_dictionary, design_matrix, design_matrix_key_dict, delta_f_matrix)




# Set Directories

"""
# Control Switching
selected_session_list = Session_List.control_switching_and_transition_nested
data_root_diretory = r"/media/matthew/Expansion/Control_Data"
tensor_directory = r"/media/matthew/External_Harddrive_1/Regression_Analysis_No_Run_Lag/Control_Switching_Analysis"
analysis_name = "Full_Model"
run_full_model_pipeline(selected_session_list, analysis_name, data_root_diretory, tensor_directory)
"""

# Neurexin Switching
selected_session_list = Session_List.mutant_switching_and_transition_nested
data_root_diretory = r"/media/matthew/External_Harddrive_1/Neurexin_Data"
tensor_directory = r"/media/matthew/External_Harddrive_1/Regression_Analysis_No_Run_Lag/Neurexin_Switching_Analysis"
analysis_name = "Full_Model"
run_full_model_pipeline(selected_session_list, analysis_name, data_root_diretory, tensor_directory)

# Control Post Learning
selected_session_list = Session_List.Control_Post_learning_Nested
data_root_diretory = r"/media/matthew/Expansion/Control_Data"
tensor_directory = r"/media/matthew/External_Harddrive_1/Regression_Analysis_No_Run_Lag/Control_Post_Learning"
analysis_name = "Correct_Rejections"
run_full_model_pipeline(selected_session_list, analysis_name, data_root_diretory, tensor_directory)

# Neurexin Post Learning
selected_session_list = Session_List.Mutant_Post_Learning_Nested
data_root_diretory = r"/media/matthew/External_Harddrive_1/Neurexin_Data"
tensor_directory = r"/media/matthew/External_Harddrive_1/Regression_Analysis_No_Run_Lag/Neurexin_Post_Learning"
analysis_name = "Correct_Rejections"
run_full_model_pipeline(selected_session_list, analysis_name, data_root_diretory, tensor_directory)

