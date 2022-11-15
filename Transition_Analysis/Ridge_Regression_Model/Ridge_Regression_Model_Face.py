import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import tables
from tqdm import tqdm

import Regression_Utils


def set_stationary_to_zero(running_trace):

    plt.hist(running_trace)
    plt.show()

def get_matched_bodycam_data(base_directory, ):

    matched_bodycam_data = []

def fit_ridge_model(base_directory, early_cutoff=3000):

    # Load Delta F Matrix
    delta_f_file = os.path.join(base_directory, "Downsampled_Delta_F.h5")
    delta_f_file_container = tables.open_file(delta_f_file, "r")
    delta_f_matrix = delta_f_file_container.root.Data
    print("Delta F Matrix Shape", np.shape(delta_f_matrix))

    # Load Downsampled AI
    downsampled_ai_file = os.path.join(base_directory, "Downsampled_AI_Matrix_Framewise.npy")
    downsampled_ai_matrix = np.load(downsampled_ai_file)

    # Create Stimuli Dictionary
    stimuli_dictionary = Regression_Utils.create_stimuli_dictionary()

    # Extract Lick and Running Traces
    lick_trace = downsampled_ai_matrix[stimuli_dictionary["Lick"]]
    running_trace = downsampled_ai_matrix[stimuli_dictionary["Running"]]

    # Subtract Traces So When Mouse Not Running Or licking They Equal 0
    running_baseline = np.load(os.path.join(base_directory, "Running_Baseline.npy"))
    running_trace = np.subtract(running_trace, running_baseline)
    running_trace = np.clip(running_trace, a_min=0, a_max=None)
    running_trace = np.expand_dims(running_trace, 1)
    print("Running Trace Shape", np.shape(running_trace))

    lick_baseline = np.load(os.path.join(base_directory, "Lick_Baseline.npy"))
    lick_trace = np.subtract(lick_trace, lick_baseline)
    lick_trace = np.clip(lick_trace, a_min=0, a_max=None)
    lick_trace = np.expand_dims(lick_trace, 1)

    # Load Bodycam Components
    bodycam_components = np.load(os.path.join(base_directory, "Mousecam_Analysis", "Matched_Transformed_Mousecam_Face_Data.npy"))
    print("Bodycam Components", np.shape(bodycam_components))


    # Create Design Matrix
    design_matrix = np.hstack([
        lick_trace,
        running_trace,
        bodycam_components,
    ])

    design_matrix = design_matrix[early_cutoff:]
    print("Design Matrix Shape", np.shape(design_matrix))

    # Iterate Through Pixels
    model = Ridge()


    # Get Chunk Structure
    chunk_size = 10000
    number_of_frames, number_of_pixels = np.shape(delta_f_matrix)
    number_of_chunks, chunk_sizes, chunk_start_list, chunk_stop_list = Regression_Utils.get_chunk_structure(chunk_size, number_of_pixels)

    # Fit Model For Each Chunk
    regression_coefs_list = []
    regression_intercepts_list = []
    for chunk_index in tqdm(range(number_of_chunks)):

        # Get Chunk Data
        chunk_start = chunk_start_list[chunk_index]
        chunk_stop = chunk_stop_list[chunk_index]
        chunk_data = delta_f_matrix[early_cutoff:, chunk_start:chunk_stop]
        chunk_data = np.nan_to_num(chunk_data)

        # Fit Model
        model.fit(y=chunk_data, X=design_matrix)

        # Get Coefs
        model_coefs = model.coef_
        model_intercept = model.intercept_

        # Add To Lists
        for coef in model_coefs:
            regression_coefs_list.append(coef)

        for intercept in model_intercept:
            regression_intercepts_list.append(intercept)

    # Save These
    save_directory = os.path.join(base_directory, "Regression_Coefs")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Create Regressor Names List
    regressor_names_list = ["Lick", "Running"]
    number_of_bodycam_components = np.shape(bodycam_components)[1]
    for component_index in range(number_of_bodycam_components):
        regressor_names_list.append("Bodycam Component: " + str(component_index))


    # Create Regression Dictionary
    regression_dict = {
        "Coefs":regression_coefs_list,
        "Intercepts":regression_intercepts_list,
        "Regressor_Names":regressor_names_list,
    }

    np.save(os.path.join(save_directory, "Regression_Dicionary_Bodycam.npy"), regression_dict)

    # Close Delta F File
    delta_f_file_container.close()


# Fit Model
session_list = [
    r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_23_Transition_Imaging",
    r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_31_Transition_Imaging",
    r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_04_02_Transition_Imaging",

    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_02_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_08_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_10_Transition_Imaging",

    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_13_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_15_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_17_Transition_Imaging",

    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_10_29_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_11_03_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_11_05_Transition_Imaging"

]

for session in session_list:
    fit_ridge_model(session)