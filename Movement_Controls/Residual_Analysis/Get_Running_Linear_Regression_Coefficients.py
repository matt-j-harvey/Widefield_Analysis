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




def get_running_regression_coefficients(base_directory, activity_tensor, onset_list, start_window, stop_window, coefficient_save_directory):

    # Get Running Tensor
    running_tensor = get_running_tensor(base_directory, onset_list, start_window, stop_window)

    # Concatenate Tensors
    number_of_trials = len(onset_list)
    trial_length = stop_window - start_window
    number_of_pixels = np.shape(activity_tensor)[2]

    running_tensor = np.reshape(running_tensor, (number_of_trials * trial_length))
    activity_tensor = np.reshape(activity_tensor, (number_of_trials * trial_length, number_of_pixels))
    activity_tensor = np.transpose(activity_tensor)

    # Perform Regression
    model = LinearRegression(fit_intercept=True)
    print("Running Tensor Shape", np.shape(running_tensor))

    print("Performing Running Regression")
    coefficients = []
    for pixel_index in range(number_of_pixels):
        model.fit(X=running_tensor.reshape(-1, 1), y=activity_tensor[pixel_index].reshape(-1, 1))
        coefficients.append(model.coef_[0][0])

    print("Coeffiecients", coefficients)

    view_cortical_vector(base_directory, coefficients, "Running_Coefficients", coefficient_save_directory)
    np.save(os.path.join(coefficient_save_directory, "Coefficient_Vector.npy"), coefficients)

    return coefficients

