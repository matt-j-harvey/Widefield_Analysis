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


def perform_regression(running_traces, activity_traces):

    # Number of Pixels
    number_of_pixels = np.shape(activity_traces)[0]

    # Get Regression Coefficients
    coefficient_vector = []
    intercept_vector = []
    for pixel_index in range(number_of_pixels):
        model = LinearRegression(fit_intercept=True)
        model.fit(X=running_traces.reshape(-1, 1), y=activity_traces[pixel_index].reshape(-1, 1))
        coef = model.coef_[0][0]
        intercept = model.intercept_[0]
        coefficient_vector.append(coef)
        intercept_vector.append(intercept)

    return coefficient_vector, intercept_vector


def get_running_regression_coefficients(base_directory, coefficient_save_name):

    # Load Delta F Matrix
    delta_f_matrix_filepath = os.path.join(base_directory, "Delta_F.h5")
    delta_f_matrix_container = tables.open_file(delta_f_matrix_filepath, mode='r')
    delta_f_matrix = delta_f_matrix_container.root['Data']

    # Load Downsampled Running Trace
    downsampled_running_trace = np.load(os.path.join(base_directory, "Downsampled_Running_Trace.npy"))

    # Get Data Structure
    number_of_pixels = np.shape(delta_f_matrix)[1]

    preferred_chunk_size = 10000
    coefficient_vector = []
    intercept_vector = []
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = Widefield_General_Functions.get_chunk_structure(preferred_chunk_size, number_of_pixels)
    for chunk_index in range(number_of_chunks):
        print("Chunk" , chunk_index, " of ", number_of_chunks)
        start = chunk_starts[chunk_index]
        stop = chunk_stops[chunk_index]
        activity_traces = delta_f_matrix[:, start:stop]
        activity_traces = np.transpose(activity_traces)
        activity_traces = np.nan_to_num(activity_traces)

        chunk_coefficients, chunk_intercepts = perform_regression(downsampled_running_trace, activity_traces)
        coefficient_vector = coefficient_vector + chunk_coefficients
        intercept_vector = intercept_vector + chunk_intercepts

    # Save These Coefficients
    coefficient_save_directory = os.path.join(base_directory, "Running_Regression_Coefficients")
    if not os.path.exists(coefficient_save_directory):
        os.mkdir(coefficient_save_directory)

    intercept_vector = np.array(intercept_vector)
    coefficient_vector = np.array(coefficient_vector)
    intercept_vector = np.nan_to_num(intercept_vector)
    coefficient_vector = np.nan_to_num(coefficient_vector)

    view_cortical_vector(base_directory, coefficient_vector, "Running_Coefficients", coefficient_save_directory)
    view_cortical_vector(base_directory, intercept_vector, "Running_Intercepts", coefficient_save_directory)

    np.save(os.path.join(coefficient_save_directory, coefficient_save_name + "_Coefficient_Vector.npy"), coefficient_vector)
    np.save(os.path.join(coefficient_save_directory, coefficient_save_name + "_Intercept_Vector.npy"),   intercept_vector)



