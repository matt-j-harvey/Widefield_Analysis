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
from sklearn.preprocessing import StandardScaler

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
        region_mean_trace = context_1_region_tensor_list[condition_index]
        number_of_trials = np.shape(region_mean_trace)[0]
        trial_length = np.shape(region_mean_trace)[1]

        region_mean_trace = np.ndarray.reshape(region_mean_trace, (number_of_trials * trial_length))
        flat_region_trace_list.append(region_mean_trace)


    for condition_index in range(number_of_conditions[1]):
        region_mean_trace = context_2_region_tensor_list[condition_index]
        number_of_trials = np.shape(region_mean_trace)[0]
        trial_length = np.shape(region_mean_trace)[1]

        region_mean_trace = np.ndarray.reshape(region_mean_trace, (number_of_trials * trial_length))

        flat_region_trace_list.append(region_mean_trace)

    region_trace = np.concatenate(flat_region_trace_list)

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

    current_position = 0
    for condition_index in range(total_number_of_conditions):

        condition_regressor = stimuli_regressors_list[condition_index]

        condition_size = np.shape(condition_regressor)[0]

        x_start = current_position
        x_stop = x_start + condition_size
        y_start = condition_index * trial_length
        y_stop = y_start + trial_length

        stimuli_regressors[x_start:x_stop, y_start:y_stop] = condition_regressor
        current_position += condition_size

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






def create_ppi_model(context_1_activity_tensor_list, context_2_activity_tensor_list, context_1_region_tensor_list, context_2_region_tensor_list, context_1_bodycam_tensor, context_2_bodycam_tensor):


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

    #print("Number Of Trials List: ", number_of_trials_list)
    total_number_of_trials = np.sum(number_of_trials_list)
    #print("Total Number Of Trials", total_number_of_trials)
    #print("Trial Length", trial_length)


    # Flatten and Concatenate Region Tensors
    flat_region_trace = flatten_and_concatenate_region_traces(context_1_region_tensor_list, context_2_region_tensor_list, number_of_conditions)

    # Flatten and Concatenate Activity Tensors
    flat_activity_tensor = flatten_and_concatenate_activity_tensors(context_1_activity_tensor_list, context_2_activity_tensor_list, number_of_conditions)

    # Scale Traces
    """
    print("Region Trace Shape", np.shape(flat_region_trace))
    print("Activity Tensor Shape", np.shape(flat_activity_tensor))

    #standard_scaler = StandardScaler()
    #flat_region_trace = standard_scaler.fit_transform(flat_region_trace.reshape(-1, 1))
    #flat_region_trace = flat_region_trace[:, 0]
    #flat_activity_tensor = standard_scaler.fit_transform(flat_activity_tensor)

    # Scale Traces
    print("Region Trace Shape", np.shape(flat_region_trace))
    print("Activity Tensor Shape", np.shape(flat_activity_tensor))
    """
    # Create Stimuli Regressors
    stimuli_regressors = create_stimuli_regressors(context_1_activity_tensor_list, context_2_activity_tensor_list, number_of_conditions, number_of_trials_list, trial_length)

    print("Stimullli Regressors Shaoe", np.shape(stimuli_regressors))
    #print("Condition 1 bodycam Regressors", np.shape(context_1_bodycam_tensor))
    #print("Condition 2 bodycam Regressors", np.shape(context_2_bodycam_tensor))

    # Create Boxcar Functions
    context_1_boxcar, context_2_boxcar = create_boxcar_funcctions(number_of_conditions, trial_length, number_of_trials_list)

    # Create Model
    model = Ridge()
    #model = LinearRegression()


    # Rehsape Bodycam Tensors
    combined_bodycam_tensor = np.vstack([context_1_bodycam_tensor, context_2_bodycam_tensor])
    number_of_trials = np.shape(combined_bodycam_tensor)[0]
    trial_length = np.shape(combined_bodycam_tensor)[1]
    bodycam_components = np.shape(combined_bodycam_tensor)[2]
    combined_bodycam_tensor = np.reshape(combined_bodycam_tensor, (number_of_trials * trial_length, bodycam_components))
    print("Combined BOdycam Tensor", np.shape(combined_bodycam_tensor))


    # Create Variable Placeholders
    number_of_samples = total_number_of_trials * trial_length
    placeholder_trace = np.zeros((number_of_samples, 1))
    baseline_regressor = np.ones((number_of_samples, 1))

    design_matrix = np.hstack([

        placeholder_trace,
        placeholder_trace,
        placeholder_trace,

        stimuli_regressors,
        baseline_regressor,
        combined_bodycam_tensor

    ])


    coefficient_tensor = []





    flat_activity_tensor = np.transpose(flat_activity_tensor)
    for pixel_index in range(number_of_pixels):


        # Create Pixel Trace
        pixel_trace = flat_activity_tensor[pixel_index]

        # Get Boxcar Traces
        condition_1_pixel_trace_boxcar = np.multiply(pixel_trace, context_1_boxcar)
        condition_2_pixel_trace_boxcar = np.multiply(pixel_trace, context_2_boxcar)

        # Create Full Design Matrix
        design_matrix[:, 0] = pixel_trace
        design_matrix[:, 1] = condition_1_pixel_trace_boxcar
        design_matrix[:, 2] = condition_2_pixel_trace_boxcar

        """
        figure_1 = plt.figure()
        rows = 3
        columns = 1
        axis_1 = figure_1.add_subplot(rows, columns, 1)
        axis_2 = figure_1.add_subplot(rows, columns, 2)
        axis_3 = figure_1.add_subplot(rows, columns, 3)
        axis_1.plot(pixel_trace, c='b',  alpha=0.5)
        axis_2.plot(condition_1_pixel_trace_boxcar, c='m',  alpha=0.5)
        axis_2.plot(condition_2_pixel_trace_boxcar, c='k',  alpha=0.5)
        axis_3.plot(flat_region_trace, c='g', alpha=0.5)
        plt.show()
        """
        """
        print("Design Matrix Shape", np.shape(design_matrix))
        plt.imshow(design_matrix)
        plt.show()
        """

        # Centre Traces
        model.fit(X=design_matrix, y=flat_region_trace)

        coefficient_tensor.append(model.coef_)

    coefficient_tensor = np.array(coefficient_tensor)
    #print("Coeffficient Tensor Shape", np.shape(coefficient_tensor))

    return coefficient_tensor







