import os
number_of_threads = 1
os.environ["OMP_NUM_THREADS"] = str(number_of_threads) # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = str(number_of_threads) # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = str(number_of_threads) # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(number_of_threads) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(number_of_threads) # export NUMEXPR_NUM_THREADS=1

#from sklearnex import patch_sklearn
#patch_sklearn()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import tables
from tqdm import tqdm
import joblib
from datetime import datetime
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold

from Widefield_Utils import widefield_utils
from Ridge_Regression_Model import Get_Cross_Validated_Ridge_Penalties
from Files import Session_List




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

    # Trim Design Matrix To This Length
    trimmed_design_matrix = []
    for regressor in design_matrix:
        trimmed_design_matrix.append(regressor[:common_length])

    trimmed_design_matrix = np.array(trimmed_design_matrix)
    return common_length, trimmed_design_matrix



def denoise_delta_f(delta_f_matrix):

    delta_f_matrix = np.nan_to_num(delta_f_matrix)
    denoise_model = TruncatedSVD(n_components=200)
    inverse_data = denoise_model.fit_transform(delta_f_matrix)
    delta_f_matrix = denoise_model.inverse_transform(inverse_data)

    return delta_f_matrix


def extract_only_running_and_licking(base_directory, design_matrix):

    design_matrix_dict = np.load(os.path.join(base_directory, "Ride_Regression", "design_matrix_key_dict.npy"), allow_pickle=True)[()]
    group_names = design_matrix_dict["Group Names"]
    group_sizes = design_matrix_dict["Group Sizes"]
    print("Group Names", group_names)
    print("Group sizes", group_sizes)

    only_lick_and_running = design_matrix[:, [0, group_sizes[0]]]
    print("Only lickig and running", np.shape(only_lick_and_running))

    return only_lick_and_running


def evaluate_model():
    """
         for regressor_group_index in range(number_of_regressor_groups):
             regressor_group_start = coef_group_starts[regressor_group_index]
             regressor_group_stop = coef_group_stops[regressor_group_index]

             # Create Partial Design Matrix With Regressor In Question Shuffled
             partial_design_matrix = np.copy(design_matrix)
             np.random.shuffle(partial_design_matrix[:, regressor_group_start:regressor_group_stop])

             # Get Sum Of Squared Error Of This Prediction
             partial_prediction = model.predict(partial_design_matrix)
             partial_sse = get_sum_square_errors(real=chunk_data, prediction=partial_prediction)

             # Calculate Coefficient Of Partial Determination
             coefficient_of_partial_determination = calculate_coefficient_of_partial_determination(full_sse, partial_sse)
             regression_cpd_matrix[chunk_start:chunk_stop, regressor_group_index] = coefficient_of_partial_determination
         """

    pass


def fit_ridge_model(base_directory, early_cutoff=3000, use_100=False):

    # Check Regression Directory
    save_directory = os.path.join(base_directory, "Ridge_Regression")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Load Design Matrix
    design_matrix = np.load(os.path.join(save_directory, "Design_Matrix.npy"))
    
    number_of_regressors = np.shape(design_matrix)[1]
    print("Design Matrix Shape", np.shape(design_matrix))

    # Load Delta F Matrix
    if use_100 == False:
        delta_f_file = os.path.join(base_directory, "Downsampled_Delta_F.h5")
        delta_f_file_container = tables.open_file(delta_f_file, "r")
        delta_f_matrix = delta_f_file_container.root.Data
        delta_f_matrix = np.array(delta_f_matrix)
    else:
        delta_f_matrix = np.load(os.path.join(base_directory, "Delta_F_Matrix_100_by_100_SVD.npy"))

    print("Delta f matrix shape", np.shape(delta_f_matrix))

    # Get Common Length
    #common_length, design_matrix = get_common_length(design_matrix, delta_f_matrix)
    #print("Common length", common_length)
    #delta_f_matrix = delta_f_matrix[:common_length]

    # Get Chunk Structure
    chunk_size = 1000
    number_of_pixels = np.shape(delta_f_matrix)[1]
    number_of_chunks, chunk_sizes, chunk_start_list, chunk_stop_list = widefield_utils.get_chunk_structure(chunk_size, number_of_pixels)

    # Fit Model For Each Chunk
    regression_intercepts_list = np.zeros(number_of_pixels)
    regression_coefs_list = np.zeros((number_of_pixels, number_of_regressors))
    ridge_penalty_list = np.zeros(number_of_pixels)

    # Remove Early Cutoff For Design Matrix
    design_matrix = design_matrix[early_cutoff:]

    for chunk_index in tqdm(range(number_of_chunks), position=1, desc="Chunk: ", leave=False):

        # Get Chunk Data
        chunk_start = chunk_start_list[chunk_index]
        chunk_stop = chunk_stop_list[chunk_index]
        chunk_data = delta_f_matrix[early_cutoff:, chunk_start:chunk_stop]
        chunk_data = np.nan_to_num(chunk_data)
        print("Chunk Data Shape", np.shape(chunk_data))

        # Get Cross Validated Ridge Penalties
        print("Getting CV Ridge Penalty")
        ridge_penalties = Get_Cross_Validated_Ridge_Penalties.get_cross_validated_ridge_penalties(design_matrix, chunk_data)

        # Create Model
        model = Ridge(solver='auto', alpha=ridge_penalties)

        # Fit Model
        model.fit(y=chunk_data, X=design_matrix)

        # Get Coefs
        model_coefs = model.coef_
        model_intercepts = model.intercept_

        # Save These
        regression_coefs_list[chunk_start:chunk_stop] = model_coefs
        regression_intercepts_list[chunk_start:chunk_stop] = model_intercepts
        ridge_penalty_list[chunk_start:chunk_stop] = ridge_penalties

    # Save These

    # Create Regression Dictionary
    regression_dict = {
        "Coefs": regression_coefs_list,
        "Intercepts": regression_intercepts_list,
        "Ridge_Penalties": ridge_penalty_list,
    }

    np.save(os.path.join(save_directory, "Regression_Dictionary_Simple.npy"), regression_dict)

    # Close Delta F File
    if use_100 == False:
        delta_f_file_container.close()


