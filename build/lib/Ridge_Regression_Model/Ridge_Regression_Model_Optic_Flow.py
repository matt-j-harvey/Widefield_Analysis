from sklearnex import patch_sklearn
patch_sklearn()

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import tables
from tqdm import tqdm
import joblib
from datetime import datetime

import Regression_Utils


def calculate_coefficient_of_partial_determination(sum_sqaure_error_full, sum_sqaure_error_reduced):
    coefficient_of_partial_determination = np.subtract(sum_sqaure_error_reduced, sum_sqaure_error_full)
    sum_sqaure_error_reduced = np.add(sum_sqaure_error_reduced, 0.000001) # Ensure We Do Not Divide By Zero
    coefficient_of_partial_determination = np.divide(coefficient_of_partial_determination, sum_sqaure_error_reduced)
    coefficient_of_partial_determination = np.nan_to_num(coefficient_of_partial_determination)

    #print("CPD Shape", np.shape(coefficient_of_partial_determination))
    #print("CPD", coefficient_of_partial_determination)
    #plt.hist(coefficient_of_partial_determination)
    #plt.show()
    return coefficient_of_partial_determination


def get_sum_square_errors(real, prediction):
    error = np.subtract(real, prediction)
    error = np.square(error)
    error = np.sum(error, axis=0)
    return error



def create_event_kernel_from_event_list(event_list, number_of_widefield_frames, preceeding_window=-14, following_window=28):

    kernel_size = following_window - preceeding_window
    design_matrix = np.zeros((number_of_widefield_frames, kernel_size))

    for timepoint_index in range(number_of_widefield_frames):

        if event_list[timepoint_index] == 1:

            # Get Start and Stop Times Of Kernel
            start_time = timepoint_index + preceeding_window
            stop_time = timepoint_index + following_window

            # Ensure Start and Stop Times Dont Fall Below Zero Or Above Number Of Frames
            start_time = np.max([0, start_time])
            stop_time = np.min([number_of_widefield_frames-1, stop_time])

            # Fill In Design Matrix
            number_of_regressor_timepoints = stop_time - start_time
            for regressor_index in range(number_of_regressor_timepoints):
                design_matrix[start_time + regressor_index, regressor_index] = 1

    return design_matrix


def fit_ridge_model(base_directory, early_cutoff=3000):

    # Iterate Through Pixels
    model = Ridge(solver='auto')

    # Load Delta F Matrix
    delta_f_file = os.path.join(base_directory, "Downsampled_Delta_F.h5")
    delta_f_file_container = tables.open_file(delta_f_file, "r")
    delta_f_matrix = delta_f_file_container.root.Data
    delta_f_matrix = np.array(delta_f_matrix)
    number_of_widefield_frames, number_of_pixels = np.shape(delta_f_matrix)
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
    #bodycam_components = np.load(os.path.join(base_directory, "Mousecam_Analysis", "Matched_Transformed_Mousecam_Face_Data.npy"))
    #print("Bodycam Components", np.shape(bodycam_components))

    # Load Limb Movements
    #limb_movements = np.load(os.path.join(base_directory, "Mousecam_Analysis", "Matched_Limb_Movements_Individual.npy"))
    limb_movements = np.load(os.path.join(base_directory, "Mousecam_Analysis", "Matched_Limb_Movements_Optic_Flow.npy"))
    print("Limb Movement Shape", np.shape(limb_movements))

    # Load Blink and Eye Movement Event Lists
    eye_movement_event_list = np.load(os.path.join(base_directory, "Eyecam_Analysis", "Matched_Eye_Movement_Events.npy"))
    blink_event_list = np.load(os.path.join(base_directory, "Eyecam_Analysis", "Matched_Blink_Events.npy"))

    # Create Regressor Kernels
    eye_movement_event_kernel = create_event_kernel_from_event_list(eye_movement_event_list, number_of_widefield_frames)
    blink_event_kernel = create_event_kernel_from_event_list(blink_event_list, number_of_widefield_frames)

    # Load Whisker Pad Motion
    whisker_pad_motion_components = np.load(os.path.join(base_directory, "Mousecam_Analysis", "matched_whisker_data.npy"))
    print("Whisker Pad Motion Components", np.shape(whisker_pad_motion_components))

    # Create Design Matrix
    design_matrix = np.hstack([
        lick_trace,
        running_trace,
        #bodycam_components,
        eye_movement_event_kernel,
        blink_event_kernel,
        whisker_pad_motion_components,
        limb_movements
    ])

    #regressor_group_chunks = [[0], 1, []]


    design_matrix = design_matrix[early_cutoff:]
    design_matrix = np.nan_to_num(design_matrix)
    print("Design Matrix Shape", np.shape(design_matrix))
    number_of_regressors = np.shape(design_matrix)[1]

    # Get Chunk Structure
    chunk_size = 1000
    number_of_frames, number_of_pixels = np.shape(delta_f_matrix)
    number_of_chunks, chunk_sizes, chunk_start_list, chunk_stop_list = Regression_Utils.get_chunk_structure(chunk_size, number_of_pixels)

    # Fit Model For Each Chunk
    regression_intercepts_list = np.zeros(number_of_pixels)
    regression_coefs_list = np.zeros((number_of_pixels, number_of_regressors))
    regression_cpd_matrix = np.zeros((number_of_pixels, number_of_regressors))

    for chunk_index in tqdm(range(number_of_chunks)):

        # Get Chunk Data
        chunk_start = chunk_start_list[chunk_index]
        chunk_stop = chunk_stop_list[chunk_index]
        chunk_data = delta_f_matrix[early_cutoff:, chunk_start:chunk_stop]
        chunk_data = np.nan_to_num(chunk_data)

        # Fit Model
        print("Fitting", datetime.now())
        model.fit(y=chunk_data, X=design_matrix)
        print("Finished Fitting", datetime.now())

        # Get Coefs
        model_coefs = model.coef_
        model_intercepts = model.intercept_

        # Save These
        regression_coefs_list[chunk_start:chunk_stop] = model_coefs
        regression_intercepts_list[chunk_start:chunk_stop] = model_intercepts

        # Get CPDs
        full_prediction = model.predict(X=design_matrix)
        full_sse = get_sum_square_errors(real=chunk_data, prediction=full_prediction)

        for regressor_index in range(number_of_regressors):

            # Create Partial Design Matrix With Regressor In Question Shuffled
            partial_design_matrix = np.copy(design_matrix)
            np.random.shuffle(partial_design_matrix[:, regressor_index])

            # Get Sum Of Squared Error Of This Prediction
            partial_prediction = model.predict(partial_design_matrix)
            partial_sse = get_sum_square_errors(real=chunk_data, prediction=partial_prediction)

            # Calculate Coefficient Of Partial Determination
            coefficient_of_partial_determination = calculate_coefficient_of_partial_determination(full_sse, partial_sse)
            regression_cpd_matrix[chunk_start:chunk_stop, regressor_index] = coefficient_of_partial_determination

    # Save These
    save_directory = os.path.join(base_directory, "Regression_Coefs")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Create Regressor Names List
    """
    regressor_names_list = ["Lick", "Running"]
    number_of_bodycam_components = np.shape(bodycam_components)[1]
    for component_index in range(number_of_bodycam_components):
        regressor_names_list.append("Bodycam Component: " + str(component_index))
    """
    # Create Regression Dictionary
    regression_dict = {
        "Coefs":regression_coefs_list,
        "Intercepts":regression_intercepts_list,
        "Coefficients_of_Partial_Determination":regression_cpd_matrix,
        #"Regressor_Names":regressor_names_list,
    }

    print("Saving")
    np.save(os.path.join(save_directory, "Regression_Dictionary_Optic_Flow.npy"), regression_dict)

    # Close Delta F File
    delta_f_file_container.close()


# Fit Model
session_list = [
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_17_Switching_Imaging",
]

for session in session_list:
    fit_ridge_model(session)


