import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
import os
import math
import scipy
import tables
from bisect import bisect_left
import cv2
from sklearn.decomposition import TruncatedSVD
from pathlib import Path
import joblib
from scipy import signal, ndimage, stats
from skimage.transform import resize
from scipy.interpolate import interp1d
import sys
from sklearn.linear_model import LinearRegression
from mpl_toolkits.axes_grid1 import make_axes_locatable


sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions





def view_cortical_vector(base_directory, vector, plot_name, save_directory):

    # Load Mask
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    map_image = np.zeros(image_width * image_height)
    map_image[indicies] = vector
    map_image = np.ndarray.reshape(map_image, (image_height, image_width))

    map_magnitude = np.max(np.abs(vector))

    ax = plt.subplot()
    im = ax.imshow(map_image, cmap='bwr', vmin=-1 * map_magnitude, vmax=map_magnitude)

    session_name = base_directory.split('/')[-2]
    ax.set_title(session_name + " " + plot_name)
    ax.axis('off')

    # Add Colourbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.savefig(os.path.join(save_directory, plot_name + ".png"))
    plt.close()


def get_running_tensor(base_directory, onset_list, start_window, stop_window):

    # Load Downsampled Running Trace
    downsampled_running_trace = np.load(os.path.join(base_directory, "Movement_Controls", "Downsampled_Running_Trace.npy"))

    running_tensor = []
    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window

        trial_running_trace = downsampled_running_trace[trial_start:trial_stop]
        running_tensor.append(trial_running_trace)

    running_tensor = np.array(running_tensor)

    return running_tensor




def get_running_regression_coefficients(base_directory, concatenated_running_tensor, concatenated_activity_tensor, coefficient_save_directory):

    # Transpose Activity Tensor For Faster Indexing (I reckon)
    concatenated_activity_tensor = np.transpose(concatenated_activity_tensor)

    # Perform Regression
    model = LinearRegression(fit_intercept=True)

    coefficients = []
    number_of_pixels = np.shape(concatenated_activity_tensor)[0]
    print("Number Of Pixels", number_of_pixels)

    print("Concantenated Running Tensor Shape", np.shape(concatenated_running_tensor))
    print("Concatenated Activity Tnesor Shape", np.shape(concatenated_activity_tensor))
    for pixel_index in range(number_of_pixels):
        model.fit(X=concatenated_running_tensor.reshape(-1, 1), y=concatenated_activity_tensor[pixel_index].reshape(-1, 1))
        coefficients.append(model.coef_[0][0])

    coefficients = np.array(coefficients)
    print("Coefs", np.shape(coefficients))

    view_cortical_vector(base_directory, coefficients, "Running_Coefficients", coefficient_save_directory)
    np.save(os.path.join(coefficient_save_directory, "Coefficient_Vector.npy"), coefficients)

    return coefficients


def flatten_activity_tensor(activity_tensor):

    number_of_trials = np.shape(activity_tensor)[0]
    trial_length = np.shape(activity_tensor)[1]
    number_of_pixels = np.shape(activity_tensor)[2]

    activity_tensor = np.reshape(activity_tensor, (number_of_trials * trial_length, number_of_pixels))

    return activity_tensor



def flatten_running_tensor(running_tensor):

    number_of_trials = np.shape(running_tensor)[0]
    trial_length = np.shape(running_tensor)[1]
    running_tensor = np.reshape(running_tensor, (number_of_trials * trial_length))

    return running_tensor



def regress_out_activity(activity_tensor, running_tensor, running_coefs):

    # Get Tensor Structure
    number_of_trials = np.shape(activity_tensor)[0]
    trial_length = np.shape(activity_tensor)[1]
    number_of_pixels = np.shape(activity_tensor)[2]

    # Flatten Each Tensor
    flat_activity_tensor = flatten_activity_tensor(activity_tensor)
    flat_running_tensor = flatten_running_tensor(running_tensor)

    print("Flat Activity Tensor", np.shape(flat_activity_tensor))
    print("Flat Running tensor", np.shape(flat_running_tensor))

    flat_running_tensor = np.reshape(flat_running_tensor, (number_of_trials * trial_length, 1))
    flat_activity_tensor = np.transpose(flat_activity_tensor)
    running_coefs = np.reshape(running_coefs, (number_of_pixels, 1))

    #running_coefs = np.transpose(running_coefs)
    flat_running_tensor = np.transpose(flat_running_tensor)

    print("Flat Activity Tensor", np.shape(flat_activity_tensor))
    print("Flat Running tensor", np.shape(flat_running_tensor))
    print("Running Coefs", np.shape(running_coefs))

    # Get Predicited_Activity
    predicited_activity = np.dot(running_coefs, flat_running_tensor)
    print("Predicited Activity Shape", np.shape(predicited_activity))

    # Subtract Predicted Activity
    print("Subtracting Activity")
    predicited_activity = np.transpose(predicited_activity)
    flat_activity_tensor = np.transpose(flat_activity_tensor)

    print("Predicited Activit Shape", np.shape(predicited_activity))
    print("Flat Activity Shape", np.shape(flat_activity_tensor))

    corrected_activity = np.subtract(flat_activity_tensor, predicited_activity)
    print("Corrected Activity Shape", np.shape(corrected_activity))

    # Reshape Predicited And Corrected Tensors
    predicited_activity_tensor = np.ndarray.reshape(predicited_activity, (number_of_trials, trial_length, number_of_pixels))
    corrected_activity_tensor = np.ndarray.reshape(corrected_activity, (number_of_trials, trial_length, number_of_pixels))

    return predicited_activity_tensor, corrected_activity_tensor


def regress_out_running(base_directory, activity_tensor_list, onset_meta_list, start_window, stop_window, experiment_name):


    number_of_conditions = len(activity_tensor_list)


    # Get Running Tensors
    running_tensor_list = []
    for onset_list in onset_meta_list:
        running_tensor = get_running_tensor(base_directory, onset_list, start_window, stop_window)
        running_tensor_list.append(running_tensor)

    # Flatten Running Tensors
    flat_running_tensor_list = []
    for running_tensor in running_tensor_list:
        flat_running_tensor = flatten_running_tensor(running_tensor)
        flat_running_tensor_list.append(flat_running_tensor)

    # Flatten Activity Tensors
    flat_activity_tensor_list = []
    for activity_tensor in activity_tensor_list:
        flat_activity_tensor = flatten_activity_tensor(activity_tensor)
        flat_activity_tensor_list.append(flat_activity_tensor)

    # Concatenate Flat Tensors
    concatenated_activity_tensor = np.concatenate(flat_activity_tensor_list)
    concatenated_running_tensor = np.concatenate(flat_running_tensor_list)

    # Get Regression Coefficients
    movement_control_directory = os.path.join(base_directory, "Movement_Controls")
    running_coef_directory = os.path.join(movement_control_directory, "Running_Regression_Coefficients")
    coefficient_save_directory = os.path.join(running_coef_directory, experiment_name)

    Widefield_General_Functions.check_directory(movement_control_directory)
    Widefield_General_Functions.check_directory(running_coef_directory)
    Widefield_General_Functions.check_directory(coefficient_save_directory)

    running_coefficients = get_running_regression_coefficients(base_directory, concatenated_running_tensor, concatenated_activity_tensor, coefficient_save_directory)


    # Regress Out Activity
    predicted_tensor_list = []
    corrected_tensor_list = []

    for condition_index in range(number_of_conditions):
        activity_tensor = activity_tensor_list[condition_index]
        running_tensor = running_tensor_list[condition_index]

        predicted_activity_tensor, corrected_activity_tensor = regress_out_activity(activity_tensor, running_tensor, running_coefficients)

        predicted_tensor_list.append(predicted_activity_tensor)
        corrected_tensor_list.append(corrected_activity_tensor)

    return predicted_tensor_list, corrected_tensor_list



"""

if running_correction == True:

    # Check Apropriate Directoriex Exist
    movement_control_directory = os.path.join(base_directory, "Movement_Controls")
    running_coef_directory = os.path.join(movement_control_directory, "Running_Regression_Coefficients")
    coefficient_save_directory = os.path.join(running_coef_directory, tensor_name)

    check_directory(movement_control_directory)
    check_directory(running_coef_directory)
    check_directory(coefficient_save_directory)

    # Get Running Regression coefficients
    running_coefficients = Get_Running_Linear_Regression_Coefficients.get_running_regression_coefficients(base_directory, activity_tensor, onsets, trial_start, trial_stop, coefficient_save_directory)

    predicted_tensor, corrected_tensor = correct_running_activity_tensor(base_directory, onsets, trial_start, trial_stop, activity_tensor, running_coefficients)
    np.save(os.path.join(save_directory, tensor_name + "_Activity_Tensor.npy"), activity_tensor)
    np.save(os.path.join(save_directory, tensor_name + "_Predicted_Tensor.npy"), predicted_tensor)
    np.save(os.path.join(save_directory, tensor_name + "_Corrected_Tensor.npy"), corrected_tensor)
else:

    """