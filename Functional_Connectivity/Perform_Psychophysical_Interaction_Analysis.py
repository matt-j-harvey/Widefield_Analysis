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
import sys
from sklearn.linear_model import LinearRegression, Ridge

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Analysis/Trial_Aligned_Analysis")
sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Create_Activity_Tensor
import Widefield_General_Functions


def create_stimulus_regressor(tensor):

    number_of_trials = np.shape(tensor)[0]
    trial_length = np.shape(tensor)[1]

    stimuli_regressors = np.zeros((number_of_trials, trial_length, trial_length))
    for timepoint_index in range(trial_length):
        stimuli_regressors[:, timepoint_index, timepoint_index] = 1

    stimuli_regressors = np.reshape(stimuli_regressors, (number_of_trials * trial_length, trial_length))

    return stimuli_regressors



def view_design_matrix(design_matrix):
    pass


def flatten_and_concatenate_region_traces(context_1_region_tensor_list, context_2_region_tensor_list, number_of_conditions):

    flat_region_trace_list = []
    for condition_index in range(number_of_conditions[0]):
        print("Context 1, Condition: ", condition_index)

        region_mean_trace = context_1_region_tensor_list[condition_index]
        number_of_trials = np.shape(region_mean_trace)[0]
        trial_length = np.shape(region_mean_trace)[1]
        print("Region mean Trace", np.shape(region_mean_trace), "Number Of Trials: ", number_of_trials, "Trial Length", trial_length)

        region_mean_trace = np.ndarray.reshape(region_mean_trace, (number_of_trials * trial_length))
        print("Region mean Trace", np.shape(region_mean_trace))
        flat_region_trace_list.append(region_mean_trace)

    for condition_index in range(number_of_conditions[1]):
        print("Context 1, Condition: ", condition_index)

        region_mean_trace = context_2_region_tensor_list[condition_index]
        number_of_trials = np.shape(region_mean_trace)[0]
        trial_length = np.shape(region_mean_trace)[1]
        print("Region mean Trace", np.shape(region_mean_trace), "Number Of Trials: ", number_of_trials, "Trial Length", trial_length)

        region_mean_trace = np.ndarray.reshape(region_mean_trace, (number_of_trials * trial_length))
        print("Region mean Trace", np.shape(region_mean_trace))
        flat_region_trace_list.append(region_mean_trace)

    region_trace = np.concatenate(flat_region_trace_list)
    #region_trace = np.expand_dims(region_trace, axis=1)

    return region_trace



def flatten_and_concatenate_activity_tensors(context_1_activity_tensor_list, context_2_activity_tensor_list, number_of_conditions):

    number_of_pixels = np.shape(context_2_activity_tensor_list[0])[2]


    flat_activity_tensor_list = []
    for condition_index in range(number_of_conditions[0]):
        activity_tensor = context_1_activity_tensor_list[condition_index]
        number_of_trials = np.shape(activity_tensor)[0]
        trial_length = np.shape(activity_tensor)[1]
        activity_tensor = np.ndarray.reshape(activity_tensor, (number_of_trials * trial_length, number_of_pixels))
        flat_activity_tensor_list.append(activity_tensor)


    for condition_index in range(number_of_conditions[1]):
        activity_tensor = context_2_activity_tensor_list[condition_index]
        number_of_trials = np.shape(activity_tensor)[0]
        trial_length = np.shape(activity_tensor)[1]
        activity_tensor = np.ndarray.reshape(activity_tensor, (number_of_trials * trial_length, number_of_pixels))
        flat_activity_tensor_list.append(activity_tensor)


    flat_activity_tensor = np.concatenate(flat_activity_tensor_list)

    return flat_activity_tensor


def create_stimuli_regressors(context_1_activity_tensor_list, context_2_activity_tensor_list, number_of_conditions, number_of_trials_list, trial_length):


    # Create Stimuli Regressors
    stimuli_regressors_list = []

    for condition_index in range(number_of_conditions[0]):
        stimuli_regressors = create_stimulus_regressor(context_1_activity_tensor_list[condition_index])
        stimuli_regressors_list.append(stimuli_regressors)

    for condition_index in range(number_of_conditions[1]):
        stimuli_regressors = create_stimulus_regressor(context_2_activity_tensor_list[condition_index])
        stimuli_regressors_list.append(stimuli_regressors)


    # Concatenate Regressors
    total_number_of_conditions = np.sum(number_of_conditions)
    total_number_of_trials = np.sum(number_of_trials_list)

    stimuli_regressors = np.zeros((total_number_of_trials * trial_length, total_number_of_conditions * trial_length))

    print("empty stimuli regresors shape", np.shape(stimuli_regressors))

    current_position = 0
    for condition_index in range(total_number_of_conditions):

        condition_regressor = stimuli_regressors_list[condition_index]
        print("Condition regressor shape", np.shape(condition_regressor))

        condition_size = np.shape(condition_regressor)[0]

        x_start = current_position
        x_stop = x_start + condition_size
        y_start = condition_index * trial_length
        y_stop = y_start + trial_length

        print("x start", x_start)
        print("x stop", x_stop)
        print("y start", y_start)
        print("y stop", y_stop)

        stimuli_regressors[x_start:x_stop, y_start:y_stop] = condition_regressor
        current_position += condition_size

    print("Stimuli regressors shape", np.shape(stimuli_regressors))

    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.imshow(np.transpose(stimuli_regressors))
    forceAspect(ax, aspect=3)
    plt.show()
    """

    return stimuli_regressors

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def create_boxcar_funcctions(number_of_conditions, trial_length, number_of_trials_list):

    # Get Total Number Of Trials
    total_number_of_trials = np.sum(number_of_trials_list)

    # Create Empty Boxcar Functions
    condition_1_boxcar_function = np.zeros(total_number_of_trials * trial_length)
    condition_2_boxcar_function = np.zeros(total_number_of_trials * trial_length)

    # Get Boxcar Structure
    number_of_context_1_conditions = np.sum(number_of_conditions[0])
    number_of_condition_1_trials = np.sum(number_of_trials_list[0: number_of_context_1_conditions])
    condition_1_duration = number_of_condition_1_trials * trial_length

    # Fill Boxcar Function
    condition_1_boxcar_function[0:condition_1_duration] = 1
    condition_2_boxcar_function[condition_1_duration:] = 1

    return condition_1_boxcar_function, condition_2_boxcar_function






def create_ppi_model(context_1_activity_tensor_list, context_2_activity_tensor_list, context_1_region_tensor_list, context_2_region_tensor_list):


    # Get Data Structure
    number_of_conditions = [len(context_1_activity_tensor_list), len(context_2_activity_tensor_list)]
    trial_length         = np.shape(context_1_activity_tensor_list[0])[1]
    number_of_pixels     = np.shape(context_1_activity_tensor_list[0])[2]

    number_of_trials_list = []

    for condition_index in range(number_of_conditions[0]):
        activity_tensor = context_1_activity_tensor_list[condition_index]
        condition_trials = np.shape(activity_tensor)[0]
        number_of_trials_list.append(condition_trials)

    for condition_index in range(number_of_conditions[0]):
        activity_tensor = context_2_activity_tensor_list[condition_index]
        condition_trials = np.shape(activity_tensor)[0]
        number_of_trials_list.append(condition_trials)

    print("Number Of Trials List: ", number_of_trials_list)
    total_number_of_trials = np.sum(number_of_trials_list)
    print("Total Number Of Trials", total_number_of_trials)
    print("Trial Length", trial_length)


    # Flatten and Concatenate Region Tensors
    flat_region_trace = flatten_and_concatenate_region_traces(context_1_region_tensor_list, context_2_region_tensor_list, number_of_conditions)
    print("Post functiion flat region trace", np.shape(flat_region_trace))

    # Flatten and Concatenate Activity Tensors
    flat_activity_tensor = flatten_and_concatenate_activity_tensors(context_1_activity_tensor_list, context_2_activity_tensor_list, number_of_conditions)
    print("Flat Activiity Tensor Shape", np.shape(flat_activity_tensor))

    # Create Stimuli Regressors
    stimuli_regressors = create_stimuli_regressors(context_1_activity_tensor_list, context_2_activity_tensor_list, number_of_conditions, number_of_trials_list, trial_length)

    # Create Boxcar Functions
    context_1_boxcar, context_2_boxcar = create_boxcar_funcctions(number_of_conditions, trial_length, number_of_trials_list)

    # Create Model
    model = Ridge()
    #model = LinearRegression()

    # Create Variable Placeholders
    number_of_samples = total_number_of_trials * trial_length
    placeholder_trace = np.zeros((number_of_samples, 1))
    baseline_regressor = np.ones((number_of_samples, 1))

    # Expand Dims
    """
    pixel_trace = np.expand_dims(pixel_trace, axis=1)
    condition_1_pixel_trace_boxcar = np.expand_dims(condition_1_pixel_trace_boxcar, axis=1)
    condition_2_pixel_trace_boxcar = np.expand_dims(condition_2_pixel_trace_boxcar, axis=1)
    baseline_regressor = np.expand_dims(baseline_regressor, axis=1)
    print(np.shape(pixel_trace))
    """

    print("Placeholder trace", np.shape(placeholder_trace))
    print("Stimuli Regressors", np.shape(stimuli_regressors))
    print("Baseline regressor", np.shape(baseline_regressor))


    design_matrix = np.hstack([

        placeholder_trace,
        placeholder_trace,
        placeholder_trace,

        stimuli_regressors,
        baseline_regressor

    ])

    print("Dsgn Matrix Shape", np.shape(design_matrix))

    coefficient_tensor = []

    flat_activity_tensor = np.transpose(flat_activity_tensor)
    for pixel_index in range(number_of_pixels):

        if pixel_index % 1000 == 0:
            print("Pixel: ", pixel_index)

        # Create Pixel Trace
        pixel_trace = flat_activity_tensor[pixel_index]

        # Get Boxcar Traces
        condition_1_pixel_trace_boxcar = np.multiply(pixel_trace, context_1_boxcar)
        condition_2_pixel_trace_boxcar = np.multiply(pixel_trace, context_2_boxcar)

        # Expand Dims
        #pixel_trace = np.expand_dims(pixel_trace, axis=1)
        #condition_1_pixel_trace_boxcar = np.expand_dims(condition_1_pixel_trace_boxcar, axis=1)
        #condition_2_pixel_trace_boxcar = np.expand_dims(condition_2_pixel_trace_boxcar, axis=1)

        # Create Full Design Matrix
        design_matrix[:, 0] = pixel_trace
        design_matrix[:, 1] = condition_1_pixel_trace_boxcar
        design_matrix[:, 2] = condition_2_pixel_trace_boxcar

        #print("Deisgn Matrix Shape", np.shape(design_matrix))
        model.fit(X=design_matrix, y=flat_region_trace)

        coefficient_tensor.append(model.coef_)

    coefficient_tensor = np.array(coefficient_tensor)

    return coefficient_tensor







