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
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import explained_variance_score

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


def get_design_matrix_structure(design_matrix):

    # Get Matrix Chunk Structure
    coef_group_sizes = []
    coef_group_starts = []
    coef_group_stops = []
    coef_index_count = 0

    for coef_group in design_matrix:
        group_size = np.shape(coef_group)[1]
        group_start = coef_index_count
        group_stop = group_start + group_size

        coef_group_sizes.append(group_size)
        coef_group_starts.append(group_start)
        coef_group_stops.append(group_stop)

        coef_index_count += group_size

    number_of_regressor_groups = len(coef_group_sizes)


    return number_of_regressor_groups, coef_group_sizes, coef_group_starts, coef_group_stops


def create_lagged_matrix(matrix, n_lags=14):
    """
    :param matrix: Matrix of shape (n_samples, n_dimensionns)
    :param n_lags: Number Of steps to include lagged versions of the matrix
    :return: Matrix with duplicated shifted version of origional matrix with shape (n_samples, n_dimensions * n_lags)
    """
    print("Getting Lagged Matrix Shape", np.shape(matrix))

    lagged_combined_matrix = []
    for lag_index in range(n_lags):
        lagged_matrix = np.copy(matrix)
        lagged_matrix = np.roll(a=lagged_matrix, axis=1, shift=lag_index)
        lagged_matrix[0:lag_index] = 0
        lagged_combined_matrix.append(lagged_matrix)

    lagged_combined_matrix = np.hstack(lagged_combined_matrix)

    return lagged_combined_matrix

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def get_common_length(design_matrix, delta_f_matirx):

    # Add Number Of Delta F Timepoints
    timepoints_list = []
    timepoints_list.append(np.shape(delta_f_matirx)[0])

    # Add Number Of Timepoints For Each Regressor
    for regresor in design_matrix:
        number_of_timepoints = np.shape(regresor)[0]
        timepoints_list.append(number_of_timepoints)

    # Get Smallest Number Of Timepoints
    common_length = np.min(timepoints_list)
    print("Timepoints List: ", timepoints_list)

    # Trim Design Matrix To This Length
    trimmed_design_matrix = []
    for regressor in design_matrix:
        trimmed_design_matrix.append(regressor[:common_length])

    return common_length, trimmed_design_matrix



def denoise_delta_f(delta_f_matrix):

    delta_f_matrix = np.nan_to_num(delta_f_matrix)
    denoise_model = TruncatedSVD(n_components=200)
    inverse_data = denoise_model.fit_transform(delta_f_matrix)
    delta_f_matrix = denoise_model.inverse_transform(inverse_data)

    return delta_f_matrix


def fit_ridge_model(base_directory, early_cutoff=3000):

    # Iterate Through Pixels
    model = Ridge(solver='auto')

    # Load Delta F Matrix
    delta_f_file = os.path.join(base_directory, "Downsampled_Delta_F.h5")
    delta_f_file_container = tables.open_file(delta_f_file, "r")
    delta_f_matrix = delta_f_file_container.root.Data
    delta_f_matrix = np.array(delta_f_matrix)

    # Denoise Delta F
    delta_f_matrix = denoise_delta_f(delta_f_matrix)

    number_of_widefield_frames, number_of_pixels = np.shape(delta_f_matrix)

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

    lick_baseline = np.load(os.path.join(base_directory, "Lick_Baseline.npy"))
    lick_trace = np.subtract(lick_trace, lick_baseline)
    lick_trace = np.clip(lick_trace, a_min=0, a_max=None)
    lick_trace = np.expand_dims(lick_trace, 1)

    # Get Lagged Lick and Running Traces
    lick_trace = create_lagged_matrix(lick_trace)
    running_trace = create_lagged_matrix(running_trace)


    # Create Lick Kernel
    lick_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Lick_Events.npy"))
    lick_event_kernel = create_event_kernel_from_event_list(lick_onsets, number_of_widefield_frames, preceeding_window=-5, following_window=14)
    lick_regressors = np.hstack([lick_trace, lick_event_kernel])

    # Create Running Kernel
    running_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Running_Events.npy"))
    running_event_kernel = create_event_kernel_from_event_list(running_onsets, number_of_widefield_frames, preceeding_window=-14, following_window=28)
    running_regressors = np.hstack([running_trace, running_event_kernel])

    # Load Limb Movements
    limb_movements = np.load(os.path.join(base_directory, "Mousecam_Analysis", "Matched_Limb_Movements_Simple.npy"))

    # Load Blink and Eye Movement Event Lists
    #eye_movement_event_list = np.load(os.path.join(base_directory, "Eyecam_Analysis", "Matched_Eye_Movement_Events.npy"))
    #blink_event_list = np.load(os.path.join(base_directory, "Eyecam_Analysis", "Matched_Blink_Events.npy"))

    # Create Regressor Kernels
    #eye_movement_event_kernel = create_event_kernel_from_event_list(eye_movement_event_list, number_of_widefield_frames)
    #blink_event_kernel = create_event_kernel_from_event_list(blink_event_list, number_of_widefield_frames)

    # Load Whisker Pad Motion
    whisker_pad_motion_components = np.load(os.path.join(base_directory, "Mousecam_Analysis", "matched_whisker_data.npy"))

    # Load Face Motion Data
    face_motion_components = np.load(os.path.join(base_directory, "Mousecam_Analysis", "matched_face_data.npy"))

    # Get Lagged Versions
    whisker_pad_motion_components = create_lagged_matrix(whisker_pad_motion_components)
    limb_movements = create_lagged_matrix(limb_movements)
    face_motion_components = create_lagged_matrix(face_motion_components)

    # Create Design Matrix
    coef_names = ["Lick", "Running", "Face_Motion", "Whisking", "Limbs"] #"Eye_Movements", "Blinks",
    design_matrix = [
        lick_regressors,
        running_regressors,
        face_motion_components,
        #eye_movement_event_kernel,
        #blink_event_kernel,
        whisker_pad_motion_components,
        limb_movements
    ]

    # Get Design Matrix Structure
    number_of_regressor_groups, coef_group_sizes, coef_group_starts, coef_group_stops = get_design_matrix_structure(design_matrix)

    # Get Common Length
    common_length, design_matrix = get_common_length(design_matrix, delta_f_matrix)
    delta_f_matrix = delta_f_matrix[:common_length]

    # Convert To Array
    design_matrix = np.hstack(design_matrix)
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
    variance_explained_list = np.zeros(number_of_pixels)
    regression_coefs_list = np.zeros((number_of_pixels, number_of_regressors))
    regression_cpd_matrix = np.zeros((number_of_pixels, number_of_regressor_groups))

    for chunk_index in range(number_of_chunks):

        # Get Chunk Data
        chunk_start = chunk_start_list[chunk_index]
        chunk_stop = chunk_stop_list[chunk_index]
        chunk_data = delta_f_matrix[early_cutoff:, chunk_start:chunk_stop]
        chunk_data = np.nan_to_num(chunk_data)

        # Fit Model
        model.fit(y=chunk_data, X=design_matrix)

        # Get Coefs
        model_coefs = model.coef_
        model_intercepts = model.intercept_

        # Get CPDs
        full_prediction = model.predict(X=design_matrix)
        full_sse = get_sum_square_errors(real=chunk_data, prediction=full_prediction)

        # Get Percentage Variance Explained
        variance_explained = explained_variance_score(y_true=chunk_data, y_pred=full_prediction, multioutput='raw_values')

        # Save These
        regression_coefs_list[chunk_start:chunk_stop] = model_coefs
        regression_intercepts_list[chunk_start:chunk_stop] = model_intercepts
        variance_explained_list[chunk_start:chunk_stop] = variance_explained

        for regressor_group_index in range(number_of_regressor_groups):
            regressor_group_start = coef_group_starts[regressor_group_index]
            regressor_group_stop = coef_group_stops[regressor_group_index]

            #print("Regressor Group: ", regressor_group_index, "Group Start", regressor_group_start, "Group Stop", regressor_group_stop)

            # Create Partial Design Matrix With Regressor In Question Shuffled
            partial_design_matrix = np.copy(design_matrix)
            np.random.shuffle(partial_design_matrix[:, regressor_group_start:regressor_group_stop])

            # Get Sum Of Squared Error Of This Prediction
            partial_prediction = model.predict(partial_design_matrix)
            partial_sse = get_sum_square_errors(real=chunk_data, prediction=partial_prediction)

            # Calculate Coefficient Of Partial Determination
            coefficient_of_partial_determination = calculate_coefficient_of_partial_determination(full_sse, partial_sse)
            regression_cpd_matrix[chunk_start:chunk_stop, regressor_group_index] = coefficient_of_partial_determination

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
        "Coef_Names":coef_names,
        "Coef_Group_Sizes:":coef_group_sizes,
        "Coef_Group_Starts":coef_group_starts,
        "Coef_Group_Stops":coef_group_stops,
        "Coefs":regression_coefs_list,
        "Intercepts":regression_intercepts_list,
        "Coefficients_of_Partial_Determination":regression_cpd_matrix,
        "Variance_Explained":variance_explained_list
    }


    print("Saving")
    np.save(os.path.join(save_directory, "Regression_Dictionary_Simple.npy"), regression_dict)

    # Close Delta F File
    delta_f_file_container.close()


# Fit Model
session_list = [

        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_13_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_15_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_17_Switching_Imaging",

        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_31_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_02_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_04_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_08_Transition_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_10_Transition_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_12_Transition_Imaging",

        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_20_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_22_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_24_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_14_Transition_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_16_Transition_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_18_Transition_Imaging",

        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_17_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_19_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_23_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_30_Transition_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_06_Transition_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_08_Transition_Imaging",

        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_15_Switching_Imaging", - error
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_17_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_19_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_22_Transition_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_24_Transition_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_26_Transition_Imaging",

        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_14_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_20_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_26_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_05_Transition_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_08_Transition_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_10_Transition_Imaging",

        #r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_28_Switching_Imaging", - error
        #r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_05_Switching_Imaging",
        #r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_09_Switching_Imaging",

        #r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_29_Switching_Imaging",
        #r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_05_Switching_Imaging",

        #r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_02_Switching_Imaging",
        #r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_04_Switching_Imaging",
        #r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_06_Switching_Imaging",
        #r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_02_Transition_Imaging",
        #r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_08_Transition_Imaging",
        #r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_10_Transition_Imaging",

        #r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_26_Switching_Imaging",
        #r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_28_Switching_Imaging",
        #r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_02_Switching_Imaging",
        #r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_23_Transition_Imaging",
        #r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_31_Transition_Imaging",
        #r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_04_02_Transition_Imaging",

        #r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_21_Switching_Imaging",
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
    ]


for session in tqdm(session_list):
    print(session)
    fit_ridge_model(session)


