import os
import numpy as np
from scipy import stats
import tables
from tqdm import tqdm
import matplotlib.pyplot as plt

from Files import Session_List
from Widefield_Utils import widefield_utils


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)



def get_design_matrix_structure(design_matrix, name_list):

    # Get Matrix Chunk Structure
    coef_group_sizes = []
    coef_group_starts = []
    coef_group_stops = []
    coefs_names = []
    coef_index_count = 0
    coef_group_count = 0

    for coef_group in design_matrix:
        group_size = np.shape(coef_group)[1]
        group_start = coef_index_count
        group_stop = group_start + group_size

        coef_group_sizes.append(group_size)
        coef_group_starts.append(group_start)
        coef_group_stops.append(group_stop)

        for regressor_index in range(group_size):
            coefs_names.append(name_list[coef_group_count] + "_" + str(regressor_index).zfill(3))

        coef_index_count += group_size
        coef_group_count += 1

    number_of_regressor_groups = len(coef_group_sizes)

    return number_of_regressor_groups, coef_group_sizes, coef_group_starts, coef_group_stops, coefs_names


def create_lagged_matrix(matrix, n_lags=14):
    """
    :param matrix: Matrix of shape (n_samples, n_dimensionns)
    :param n_lags: Number Of steps to include lagged versions of the matrix
    :return: Matrix with duplicated shifted version of origional matrix with shape (n_samples, n_dimensions * n_lags)
    """

    lagged_combined_matrix = []
    lagged_combined_matrix.append(matrix)

    for lag_index in range(1, n_lags):
        original_matrix = np.copy(matrix)
        shifted_matrix = np.roll(a=original_matrix, axis=0, shift=lag_index)
        shifted_matrix[0:lag_index] = 0
        lagged_combined_matrix.append(shifted_matrix)

    lagged_combined_matrix = np.hstack(lagged_combined_matrix)

    return lagged_combined_matrix


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def get_common_length(design_matrix):

    # Add Number Of Timepoints For Each Regressor
    timepoints_list = []
    for regresor in design_matrix:
        number_of_timepoints = np.shape(regresor)[0]
        timepoints_list.append(number_of_timepoints)

    # Get Smallest Number Of Timepoints
    common_length = np.min(timepoints_list)

    # Trim Design Matrix To This Length
    trimmed_design_matrix = []
    for regressor in design_matrix:
        trimmed_design_matrix.append(regressor[:common_length])

    return common_length, trimmed_design_matrix


def plot_design_matrix(design_matrix, regressor_names, save_directory):

    number_of_regressors = np.shape(design_matrix)[1]

    figure_1 = plt.figure(figsize=(20,20))
    axis_1 = figure_1.add_subplot(1,1,1)


    design_matrix_magnitude = np.max(np.abs(design_matrix))
    axis_1.imshow(np.transpose(design_matrix[3000:4000]), cmap="seismic", vmin=-design_matrix_magnitude, vmax=design_matrix_magnitude)

    axis_1.set_yticks(list(range(0, number_of_regressors)))
    axis_1.set_yticklabels(regressor_names)

    figure_1.suptitle("Design Matrix Sample")

    forceAspect(plt.gca())
    plt.savefig(os.path.join(save_directory, "Design_Matrix_Sample.svg"))
    plt.close()


def scale_continous_regressors(regressor_matrix):

    # Subtract Mean
    regressor_mean = np.mean(regressor_matrix, axis=0)
    regressor_sd = np.std(regressor_matrix, axis=0)

    # Devide By 2x SD
    regressor_matrix = np.subtract(regressor_matrix, regressor_mean)
    regressor_matrix = np.divide(regressor_matrix, 2 * regressor_sd)

    #plt.hist(np.ndarray.flatten(regressor_matrix), bins=100)
    #plt.show()
    return regressor_matrix



def create_design_matrix(base_directory):

    """

    Design Matrix Stucture

    Binarised Lick Trace
    Lagged Binarised Lick Trace Upto 500ms

    Running Trace
    Lagged Running Trace upto 500ms

    Top 18 Face Motion PCs

    Design Matrix Is Z Scored

    """

    # Settings
    lick_lags = 14
    running_lags = 14
    n_face_components = 18

    # Load Downsampled AI
    downsampled_ai_file = os.path.join(base_directory, "Downsampled_AI_Matrix_Framewise.npy")
    downsampled_ai_matrix = np.load(downsampled_ai_file)

    # Create Stimuli Dictionary
    stimuli_dictionary = widefield_utils.create_stimuli_dictionary()

    # Extract Lick and Running Traces
    lick_trace = downsampled_ai_matrix[stimuli_dictionary["Lick"]]
    running_trace = downsampled_ai_matrix[stimuli_dictionary["Running"]]

    # Binarise Lick Trace
    lick_threshold = np.load(os.path.join(base_directory, "Lick_Threshold.npy"))
    binarised_lick_trace = np.where(lick_trace > lick_threshold, 1, 0)

    # Get Lagged Lick and Running Traces
    binarised_lick_trace = np.expand_dims(binarised_lick_trace, 1)
    running_trace = np.expand_dims(running_trace, 1)

    lick_regressors = create_lagged_matrix(binarised_lick_trace, n_lags=lick_lags)
    running_regressors = create_lagged_matrix(running_trace, n_lags=running_lags)

    # Load Face Motion Data
    face_motion_components = np.load(os.path.join(base_directory, "Mousecam_Analysis", "matched_face_data.npy"))
    face_motion_components = face_motion_components[:, 0:n_face_components]

    # Scale Regressors
    #scale_continous_regressors(running_regressors)
    #scale_continous_regressors(face_motion_components)

    # Create Design Matrix
    coef_names = ["lick_Lag", "Running_Lag", "Face_Motion_PC"]

    print("Lick", np.shape(lick_regressors))
    print("running", np.shape(running_regressors))
    print("Face", np.shape(face_motion_components))

    design_matrix = [
        lick_regressors,
        running_regressors,
        face_motion_components,
    ]

    # Sometimes Mousecam Turned Off Before Widefield Cam - So We Stop Prematurely,
    common_length, design_matrix = get_common_length(design_matrix)

    number_of_regressor_groups, coef_group_sizes, coef_group_starts, coef_group_stops, coefs_names = get_design_matrix_structure(design_matrix, coef_names)

    design_matrix = np.hstack(design_matrix)

    # Save These
    save_directory = os.path.join(base_directory, "Behaviour_Ridge_Regression")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Create Dictionary Key
    design_matrix_key_dict = {}
    design_matrix_key_dict["number_of_regressor_groups"] = number_of_regressor_groups
    design_matrix_key_dict["coef_group_sizes"] = coef_group_sizes
    design_matrix_key_dict["coef_group_starts"] = coef_group_starts
    design_matrix_key_dict["coef_group_stops"] = coef_group_stops
    design_matrix_key_dict["coefs_names"] = coefs_names

    np.save(os.path.join(save_directory, "Behaviour_Design_Matrix.npy"), design_matrix)
    np.save(os.path.join(save_directory, "Behaviour_design_matrix_key_dict.npy"), design_matrix_key_dict)
    np.save(os.path.join(save_directory, "Behaviour_Design_Matrix_Common_Length.npy"), common_length)

    # Plot Design Matrix
    plot_design_matrix(design_matrix, coefs_names, save_directory)

