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

def get_stimuli_regressors(tensor):

    number_of_trials = np.shape(tensor)[0]
    trial_length = np.shape(tensor)[1]

    stimuli_regressors = np.zeros((number_of_trials, trial_length, trial_length))
    for timepoint_index in range(trial_length):
        stimuli_regressors[:, timepoint_index, timepoint_index] = 1

    stimuli_regressors = np.reshape(stimuli_regressors, (number_of_trials * trial_length, trial_length))

    return stimuli_regressors


def create_ppi_model(condition_1_tensor, condition_2_tensor, condition_1_region_mean_trace, condition_2_region_mean_trace):

    # Get Data Structure
    condition_1_trials = np.shape(condition_1_tensor)[0]
    condition_2_trials = np.shape(condition_2_tensor)[0]
    trial_length = np.shape(condition_1_tensor)[1]
    number_of_pixels = np.shape(condition_1_tensor)[2]
    print("trial length", trial_length)

    # Create Region Trace
    condition_1_region_mean_trace = np.ndarray.reshape(condition_1_region_mean_trace, (condition_1_trials * trial_length))
    condition_2_region_mean_trace = np.ndarray.reshape(condition_2_region_mean_trace, (condition_2_trials * trial_length))
    region_trace = np.concatenate([condition_1_region_mean_trace, condition_2_region_mean_trace])
    region_trace = np.expand_dims(region_trace, axis=1)

    # Create Boxcar Functions
    condition_1_boxcar_function = np.concatenate([np.ones(condition_1_trials * trial_length), np.zeros(condition_2_trials * trial_length)])
    condition_2_boxcar_function = np.concatenate([np.zeros(condition_1_trials * trial_length), np.ones(condition_2_trials * trial_length)])

    # Create Stimuli Regressors
    condition_1_stimuli_regressors = get_stimuli_regressors(condition_1_tensor)
    condition_2_stimuli_regressors = get_stimuli_regressors(condition_2_tensor)
    print("condition 1 stimuli regressors", np.shape(condition_1_stimuli_regressors))
    print("condition 2 stimuli regressors", np.shape(condition_2_stimuli_regressors))

    condition_1_stimuli_regressors = np.vstack([condition_1_stimuli_regressors, np.zeros((condition_2_trials * trial_length, trial_length))])
    condition_2_stimuli_regressors = np.vstack([np.zeros((condition_1_trials * trial_length, trial_length)), condition_2_stimuli_regressors])
    print("condition 1 stimuli regressors", np.shape(condition_1_stimuli_regressors))
    print("condition 2 stimuli regressors", np.shape(condition_2_stimuli_regressors))


    # Reshape Tensors
    condition_1_tensor = np.reshape(condition_1_tensor, (condition_1_trials * trial_length, number_of_pixels))
    condition_2_tensor = np.reshape(condition_2_tensor, (condition_2_trials * trial_length, number_of_pixels))

    coefficient_tensor = np.zeros((number_of_pixels, (trial_length*2)+4))
    #coefficient_tensor = np.zeros((number_of_pixels, 3))
    model = Ridge(fit_intercept=True)

    concatenated_tensor = np.vstack([condition_1_tensor, condition_2_tensor])
    concatenated_tensor = np.transpose(concatenated_tensor)

    # Create Variable Placeholders
    number_of_samples = (condition_1_trials * trial_length) + (condition_2_trials * trial_length)
    pixel_trace = np.zeros(number_of_samples)
    condition_1_pixel_trace_boxcar = np.zeros(number_of_samples)
    condition_2_pixel_trace_boxcar = np.zeros(number_of_samples)
    baseline_regressor = np.ones(number_of_samples)

    # Expand Dims
    pixel_trace = np.expand_dims(pixel_trace, axis=1)
    condition_1_pixel_trace_boxcar = np.expand_dims(condition_1_pixel_trace_boxcar, axis=1)
    condition_2_pixel_trace_boxcar = np.expand_dims(condition_2_pixel_trace_boxcar, axis=1)
    baseline_regressor = np.expand_dims(baseline_regressor, axis=1)
    print(np.shape(pixel_trace))

    design_matrix = np.hstack([

        pixel_trace,
        condition_1_pixel_trace_boxcar,
        condition_2_pixel_trace_boxcar,

        condition_1_stimuli_regressors,
        condition_2_stimuli_regressors,
        baseline_regressor,

    ])



    for pixel_index in range(number_of_pixels):

        if pixel_index % 1000 == 0:
            print("Pixel: ", pixel_index)

        # Create Pixel Trace
        pixel_trace = concatenated_tensor[pixel_index]

        # Get Boxcar Traces
        condition_1_pixel_trace_boxcar = np.multiply(pixel_trace, condition_1_boxcar_function)
        condition_2_pixel_trace_boxcar = np.multiply(pixel_trace, condition_2_boxcar_function)

        # Expand Dims
        #pixel_trace = np.expand_dims(pixel_trace, axis=1)
        #condition_1_pixel_trace_boxcar = np.expand_dims(condition_1_pixel_trace_boxcar, axis=1)
        #condition_2_pixel_trace_boxcar = np.expand_dims(condition_2_pixel_trace_boxcar, axis=1)

        # Create Full Design Matrix
        design_matrix[:, 0] = pixel_trace
        design_matrix[:, 1] = condition_1_pixel_trace_boxcar
        design_matrix[:, 2] = condition_2_pixel_trace_boxcar

        #print("Deisgn Matrix Shape", np.shape(design_matrix))
        model.fit(X=design_matrix, y=region_trace)

        coefficient_tensor[pixel_index] = model.coef_

        """
        print("Region Trace", np.shape(region_trace))
        print("Pixel Trace", np.shape(pixel_trace))
        print("Condition 1 stimuli regressors", np.shape(condition_1_stimuli_regressors))
        print("Condition 2 stimuli regressors", np.shape(condition_2_stimuli_regressors))
        print("Condition 1 boxcar", np.shape(condition_1_pixel_trace_boxcar))
        print("Condition 2 boxcar", np.shape(condition_2_pixel_trace_boxcar))

        plt.plot(region_trace)
        plt.plot(pixel_trace)
        plt.plot(condition_1_pixel_trace_boxcar, alpha=0.2)
        plt.plot(condition_2_pixel_trace_boxcar, alpha=0.2)
        plt.show()

        plt.plot(region_trace)
        plt.plot(condition_1_stimuli_regressors[:, 0])
        plt.plot(condition_2_stimuli_regressors[:, 0])
        plt.show()

        predicited = model.predict(design_matrix)
        plt.plot(region_trace)
        plt.plot(predicited)
        plt.show()
        """

    return coefficient_tensor




def perform_ppi_analysis(base_directory, tensor_names, onset_names, region_dictionary, selected_regions, trial_start=-5, trial_stop=20):

    # Get File Details
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)


    # Get Activity Tensors
    context_one_tensor_filepath = os.path.join(base_directory, "Activity_Tensors", tensor_names[0] + "_Activity_Tensor.npy")
    context_two_tensor_filepath = os.path.join(base_directory, "Activity_Tensors", tensor_names[1] + "_Activity_Tensor.npy")

    if not os.path.isfile(context_one_tensor_filepath):
        Create_Activity_Tensor.create_activity_tensor(base_directory, [onset_names[0]], trial_start, trial_stop, tensor_names[0], running_correction=False, spatial_smoothing=True)

    if not os.path.isfile(context_two_tensor_filepath):
        Create_Activity_Tensor.create_activity_tensor(base_directory, ["odour_context_stable_vis_2_onsets.npy"],  trial_start, trial_stop, tensor_names[1], running_correction=False, spatial_smoothing=True)

    visual_context_tensor = np.load(os.path.join(base_directory, "Activity_Tensors", tensor_names[0] + "_Activity_Tensor.npy"))
    odour_context_tensor = np.load(os.path.join(base_directory, "Activity_Tensors", tensor_names[1] + "_Activity_Tensor.npy"))


    # Load All Regions
    pixel_assignment_image = np.load(os.path.join(base_directory, "Pixel_Assignmnets_Image.npy"))
    plt.imshow(pixel_assignment_image)
    plt.show()

    # Select Regions
    region_map = np.load(os.path.join(base_directory, "Pixel_Assignmnets.npy"))

    for region in selected_regions:

        visual_context_region_tensor = Create_Activity_Tensor.get_region_activity(visual_context_tensor, region_map, selected_regions)
        odour_context_region_tensor = Create_Activity_Tensor.get_region_activity(odour_context_tensor, region_map, selected_regions)
        ppi_coefs = create_ppi_model(visual_context_tensor, odour_context_tensor, visual_context_region_tensor, odour_context_region_tensor)
        np.save(os.path.join(base_directory, region_name + "_PPI_Coefs.npy"), ppi_coefs)


        ppi_coefs = np.load(os.path.join(base_directory, region_name + "_PPI_Coefs.npy"))
        ppi_coefs = np.nan_to_num(ppi_coefs)
        vmax = np.percentile(np.abs(ppi_coefs), 99)

        print(np.shape(ppi_coefs))

        condition_1_coefs = ppi_coefs[:, 1]
        condition_2_coefs = ppi_coefs[:, 2]

        condition_1_image = Widefield_General_Functions.create_image_from_data(condition_1_coefs, indicies, image_height, image_width)
        condition_2_image = Widefield_General_Functions.create_image_from_data(condition_2_coefs, indicies, image_height, image_width)

        figure_1 = plt.figure()
        condition_1_axis = figure_1.add_subplot(1, 3, 1)
        condition_2_axis = figure_1.add_subplot(1, 3, 2)
        difference_axis = figure_1.add_subplot(1, 3, 3)

        condition_1_axis.imshow(condition_1_image, cmap='bwr', vmin=-1 * vmax, vmax=vmax)
        condition_2_axis.imshow(condition_2_image, cmap='bwr', vmin=-1 * vmax, vmax=vmax)

        difference_image = np.subtract(condition_1_image, condition_2_image)
        difference_magnitude = np.max(np.abs(difference_image))
        difference_axis.imshow(difference_image, cmap='bwr', vmin=-1 * difference_magnitude, vmax=difference_magnitude)

        plt.show()


        plt.ion()
        for x in range(123):
            plt.title(str(x))
            data = ppi_coefs[:, x]
            image = Widefield_General_Functions.create_image_from_data(data, indicies, image_height, image_width)
            plt.imshow(image, cmap='bwr', vmin=-1 * vmax, vmax=vmax)

            plt.draw()
            plt.pause(1)
            plt.clf()



region_dictionary = {
    "V1":[45, 46],
    "Retrosplenial":[28],
    "Secondary_Motor":[8, 9],
    "Somatosensory":[24, 21]}


selected_regions = ["V1", "Retrosplenial", "Secondary_Motor", "Somatosensory"]
base_directory = r"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging"
tensor_names = ['visual_context_stable_vis_2_tensor', 'odour_context_stable_vis_2_tensor']
onset_names = ["visual_context_stable_vis_2_onsets.npy", "odour_context_stable_vis_2_onsets.npy"]
