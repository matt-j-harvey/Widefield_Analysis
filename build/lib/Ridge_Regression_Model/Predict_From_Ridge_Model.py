import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD, PCA
import tables
from matplotlib.cm import ScalarMappable
from matplotlib.pyplot import Normalize
from tqdm import tqdm
from scipy import ndimage, signal
import cv2

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

def get_lagged_matrix(matrix, n_lags=14):
    """
    :param matrix: Matrix of shape (n_dimensionns, n_samples)
    :param n_lags: Number Of steps to include lagged versions of the matrix
    :return: Matrix with duplicated shifted version of origional matrix with shape (n_dimensions * n_lages, n_samples)
    """
    print("Getting Lagged Matrix Shape", np.shape(matrix))

    lagged_combined_matrix = []
    for lag_index in range(n_lags):
        lagged_matrix = np.copy(matrix)
        lagged_matrix = np.roll(a=lagged_matrix, axis=1, shift=lag_index)
        lagged_matrix[:, 0:lag_index] = 0
        lagged_combined_matrix.append(lagged_matrix)

    lagged_combined_matrix = np.vstack(lagged_combined_matrix)
    return lagged_combined_matrix


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

def denoise_data(sample_data):

    # Remove NaNS
    sample_data = np.nan_to_num(sample_data)

    # Filter
    sampling_frequency = 28  # In Hertz
    cutoff_frequency = 12  # In Hertz
    w = cutoff_frequency / (sampling_frequency / 2)  # Normalised frequency
    b, a = signal.butter(1, w, 'lowpass')
    sample_data = signal.filtfilt(b, a, sample_data, axis=0)

    # Denoise with dimensionality reduction
    model = PCA(n_components=150)
    transformed_data = model.fit_transform(sample_data)
    sample_data = model.inverse_transform(transformed_data)

    return sample_data


def predict_activity(base_directory, early_cutoff=10000, sample_size=5000):

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

    # Get Lagged Lick and Running Traces
    print("Lick Trace Shape", np.shape(lick_trace))
    lick_trace = create_lagged_matrix(lick_trace)
    print("Lagged Lick Trace Shape", np.shape(lick_trace))
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

    # Load Face Motion Data
    face_motion_components = np.load(os.path.join(base_directory, "Mousecam_Analysis", "matched_face_data.npy"))
    print("Face  Motion Components", np.shape(whisker_pad_motion_components))

    # Get Lagged Versions
    whisker_pad_motion_components = create_lagged_matrix(whisker_pad_motion_components)
    limb_movements = create_lagged_matrix(limb_movements)
    face_motion_components = create_lagged_matrix(face_motion_components)
    print("Whisker Pad Motion Components", np.shape(whisker_pad_motion_components))

    # Create Design Matrix
    coef_names = ["Lick", "Running", "Face_Motion", "Eye_Movements", "Blinks", "Whisking", "Limbs"]


    print("Design Matrix:")
    print("Lick Regressors: ", np.shape(lick_regressors))
    print("Running Regressors: ", np.shape(running_regressors))
    print("face_motion_components", np.shape(face_motion_components))
    print("eye_movement_event_kernel", np.shape(eye_movement_event_kernel))
    print("blink_event_kernel", np.shape(blink_event_kernel))
    print("whisker_pad_motion_components", np.shape(whisker_pad_motion_components))
    print("limb_movements", np.shape(limb_movements))

    design_matrix = [
        lick_regressors,
        running_regressors,
        face_motion_components,
        eye_movement_event_kernel,
        blink_event_kernel,
        whisker_pad_motion_components,
        limb_movements
    ]
    design_matrix = np.hstack(design_matrix)
    print("Deisgn Matrix", np.shape(design_matrix))
    design_matrix = design_matrix[early_cutoff:early_cutoff + sample_size]

    # Load Regression Dict
    regression_dictionary = np.load(os.path.join(base_directory, "Regression_Coefs",  "Regression_Dictionary_Simple.npy"), allow_pickle=True)[()]
    regression_intercepts = regression_dictionary["Intercepts"]
    print("Intercepts Shape", np.shape(regression_intercepts))

    regression_coefs = regression_dictionary["Coefs"]
    regression_coefs = np.transpose(regression_coefs)
    print("Regression Coefs Shape", np.shape(regression_coefs))

    # Get Model Prediction
    model_prediction = np.dot(design_matrix, regression_coefs)
    model_prediction = np.add(model_prediction, regression_intercepts)
    print("Model Prediction Shape", np.shape(model_prediction))

    # Get Real Sample
    real_data = delta_f_matrix[early_cutoff:early_cutoff + sample_size]

    # Denoise Data
    print("Denoising Real Data")
    real_data = denoise_data(real_data)
    print("Denoising Model Prediction")
    model_prediction = denoise_data(model_prediction)

    # View Model Prediction
    indicies, image_height, image_width = Regression_Utils.load_downsampled_mask(base_directory)

    # Create Video File
    reconstructed_file = os.path.join(base_directory, "Ridge_Model_Prediction.avi")
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(reconstructed_file, video_codec, frameSize=(image_width * 3, image_height), fps=30)  # 0, 12

    colourmap = ScalarMappable(cmap=Regression_Utils.get_musall_cmap(), norm=Normalize(vmin=-0.05, vmax=0.05))
    for frame_index in tqdm(range(sample_size)):

        predicted_frame = model_prediction[frame_index]
        real_frame = real_data[frame_index]
        residual = np.subtract(real_frame, predicted_frame)

        predicted_frame = Regression_Utils.create_image_from_data(predicted_frame, indicies, image_height, image_width)
        real_frame = Regression_Utils.create_image_from_data(real_frame, indicies, image_height, image_width)
        residual = Regression_Utils.create_image_from_data(residual, indicies, image_height, image_width)

        #real_frame = ndimage.gaussian_filter(real_frame, sigma=1)

        """
        figure_1 = plt.figure()
        rows = 1
        columns = 3
        real_axis = figure_1.add_subplot(rows, columns, 1)
        predicited_axis = figure_1.add_subplot(rows, columns, 2)
        residual_axis = figure_1.add_subplot(rows, columns, 3)

        real_axis.imshow(real_frame, vmin=-0.05, vmax=0.05, cmap=Regression_Utils.get_musall_cmap())
        predicited_axis.imshow(predicted_frame, vmin=-0.05, vmax=0.05, cmap=Regression_Utils.get_musall_cmap())
        residual_axis.imshow(residual, vmin=-0.05, vmax=0.05, cmap=Regression_Utils.get_musall_cmap())
        plt.show()
        """

        frame = np.hstack([real_frame, predicted_frame, residual])

        frame = colourmap.to_rgba(frame)
        frame = frame * 255
        frame = np.ndarray.astype(frame, np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()



# Fit Model
session_list = [
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_17_Switching_Imaging",
]

for session in session_list:
    predict_activity(session)


