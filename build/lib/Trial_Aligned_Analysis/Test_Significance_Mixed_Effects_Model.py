import os

number_of_threads = 1
os.environ["OMP_NUM_THREADS"] = str(number_of_threads) # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = str(number_of_threads) # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = str(number_of_threads) # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(number_of_threads) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(number_of_threads) # export NUMEXPR_NUM_THREADS=1


import h5py
from tqdm import tqdm
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import tables
from datetime import datetime

from Widefield_Utils import widefield_utils

# Remove This Later
import warnings
warnings.filterwarnings("ignore")




def repackage_data_into_dataframe(pixel_activity, pixel_metadata):

    # Combine_Into Dataframe
    dataframe = pd.DataFrame(dtype=np.float64)
    dataframe["Data_Value"] = pixel_activity
    dataframe["Group"] = pixel_metadata[:, 0]
    dataframe["Mouse"] = pixel_metadata[:, 1]
    dataframe["Session"] = pixel_metadata[:, 2]
    dataframe["Condition"] = pixel_metadata[:, 3]

    return dataframe


def mixed_effects_random_slope_and_intercept(dataframe):

    model = sm.MixedLM.from_formula("Data_Value ~ Condition", dataframe, re_formula="Condition", groups=dataframe["Mouse"])
    model_fit = model.fit()
    parameters = model_fit.params
    group_slope = parameters[1]
    p_value = model_fit.pvalues["Condition"]

    return p_value, group_slope



def view_learning_raw_difference(tensor_directory, analysis_name, vmin=-0.05, vmax=0.05):

    # Open Analysis Dataframe
    analysis_file = tables.open_file(os.path.join(tensor_directory, analysis_name + "_Trialwise_.h5"), mode="r")
    activity_dataset = analysis_file.root["Data"]
    metadata_dataset = analysis_file.root["Trial_Details"]

    number_of_trials, number_of_timepoints, number_of_pixels = np.shape(activity_dataset)

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

    print("metadata_dataset", np.shape(metadata_dataset))
    print("activity_dataset", np.shape(activity_dataset))

    # Load AS Array
    print("Starting opening", datetime.now())
    activity_dataset = np.array(activity_dataset)
    print("Finished opening", datetime.now())

    # Split By Condition
    condition_details = metadata_dataset[:, 2]
    condition_1_indicies = np.where(condition_details == 0)[0]
    condition_2_indicies = np.where(condition_details == 1)[0]
    condition_3_indicies = np.where(condition_details == 2)[0]

    condition_1_data = activity_dataset[condition_1_indicies]
    condition_2_data = activity_dataset[condition_2_indicies]
    condition_3_data = activity_dataset[condition_3_indicies]
    print("Condition 1 data", np.shape(condition_1_data))
    print("condition 2 data", np.shape(condition_2_data))
    print("condition 3 data", np.shape(condition_3_data))

    # Get MEans
    condition_1_data = np.mean(condition_1_data, axis=0)
    condition_2_data = np.mean(condition_2_data, axis=0)
    condition_3_data = np.mean(condition_3_data, axis=0)

    # Load Colourmap
    colourmap = widefield_utils.get_musall_cmap()


    plt.ion()
    figure_1 = plt.figure()
    for timepoint_index in tqdm(range(number_of_timepoints), position=0, desc="Timepoint"):

        condition_1_axis = figure_1.add_subplot(1, 4, 1)
        condition_2_axis = figure_1.add_subplot(1, 4, 2)
        condition_3_axis = figure_1.add_subplot(1, 4, 3)
        diff_axis = figure_1.add_subplot(1, 4, 4)

        # Recreate Images
        condition_1_image = widefield_utils.create_image_from_data(condition_1_data[timepoint_index], indicies, image_height, image_width)
        condition_2_image = widefield_utils.create_image_from_data(condition_2_data[timepoint_index], indicies, image_height, image_width)
        condition_3_image = widefield_utils.create_image_from_data(condition_3_data[timepoint_index], indicies, image_height, image_width)
        diff_image = np.subtract(condition_3_image, condition_2_image)

        # Plot These
        condition_1_axis.imshow(condition_1_image, cmap=colourmap, vmin=vmin, vmax=vmax)
        condition_2_axis.imshow(condition_2_image, cmap=colourmap, vmin=vmin, vmax=vmax)
        condition_3_axis.imshow(condition_3_image, cmap=colourmap, vmin=vmin, vmax=vmax)
        diff_axis.imshow(diff_image, cmap=colourmap, vmin=vmin*0.5, vmax=vmax*0.5)

        plt.title(str(timepoint_index))
        plt.draw()
        plt.pause(0.1)
        plt.clf()

    plt.ioff()


    window = list(range(100,114))
    figure_1 = plt.figure()

    condition_1_axis = figure_1.add_subplot(1, 3, 1)
    condition_2_axis = figure_1.add_subplot(1, 3, 2)
    diff_axis = figure_1.add_subplot(1, 3, 3)

    # Get Average
    condition_1_average = np.mean(condition_2_data[window], axis=0)
    condition_2_average = np.mean(condition_3_data[window], axis=0)

    # Recreate Images
    condition_1_image = widefield_utils.create_image_from_data(condition_1_average, indicies, image_height, image_width)
    condition_2_image = widefield_utils.create_image_from_data(condition_2_average, indicies, image_height, image_width)
    diff_image = np.subtract(condition_2_image, condition_1_image)

    # Plot These
    vmin=-0.02
    vmax=0.02
    condition_1_axis.imshow(condition_1_image, cmap=colourmap, vmin=vmin, vmax=vmax)
    condition_2_axis.imshow(condition_2_image, cmap=colourmap, vmin=vmin, vmax=vmax)
    diff_axis.imshow(diff_image, cmap=colourmap, vmin=vmin * 0.5, vmax=vmax * 0.5)

    plt.show()

def view_raw_difference(tensor_directory, analysis_name, vmin=-0.05, vmax=0.05):

    # Open Analysis Dataframe
    analysis_file = tables.open_file(os.path.join(tensor_directory, analysis_name + "_Trialwise_.h5"), mode="r")
    activity_dataset = analysis_file.root["Data"]
    metadata_dataset = analysis_file.root["Trial_Details"]

    number_of_trials, number_of_timepoints, number_of_pixels = np.shape(activity_dataset)

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further( indicies, image_height, image_width)

    print("metadata_dataset", np.shape(metadata_dataset))
    print("activity_dataset", np.shape(activity_dataset))

    # Load AS Array
    print("Starting opening", datetime.now())
    activity_dataset = np.array(activity_dataset)
    print("Finished opening", datetime.now())

    # Split By Condition
    condition_details = metadata_dataset[:, 3]
    condition_1_indicies = np.where(condition_details == 0)[0]
    condition_2_indicies = np.where(condition_details == 1)[0]

    condition_1_data = activity_dataset[condition_1_indicies]
    condition_2_data = activity_dataset[condition_2_indicies]
    print("Condition 1 data", np.shape(condition_1_data))
    print("condition 2 data", np.shape(condition_2_data))

    # Get MEans
    condition_1_data = np.mean(condition_1_data, axis=0)
    condition_2_data = np.mean(condition_2_data, axis=0)

    # Load Colourmap
    colourmap = widefield_utils.get_musall_cmap()

    for timepoint_index in tqdm(range(number_of_timepoints), position=0, desc="Timepoint"):
        figure_1 = plt.figure()

        condition_1_axis = figure_1.add_subplot(1,3,1)
        condition_2_axis = figure_1.add_subplot(1, 3, 2)
        diff_axis = figure_1.add_subplot(1, 3, 3)

        # Recreate Images
        condition_1_image = widefield_utils.create_image_from_data(condition_1_data[timepoint_index],  indicies, image_height, image_width)
        condition_2_image = widefield_utils.create_image_from_data(condition_2_data[timepoint_index], indicies, image_height, image_width)

        # Plot These
        condition_1_axis.imshow(condition_1_image, cmap=colourmap, vmin=vmin, vmax=vmax)
        condition_2_axis.imshow(condition_2_image, cmap=colourmap, vmin=vmin, vmax=vmax)
        diff_axis.imshow(np.subtract(condition_1_image, condition_2_image), cmap=colourmap, vmin=-0.02, vmax=0.02)


        plt.title(str(timepoint_index))
        plt.show()


def test_significance_individual_timepoints(tensor_directory, analysis_name):

    """
    This Test Is Run Pixelwise - All Brains Must Be In Same Pixel Space

    :return:
    Tensor of P Values
    """

    """
    # Open Analysis Dataframe
    analysis_file = h5py.File(os.path.join(tensor_directory, analysis_name + ".hdf5"), "r")
    activity_dataset = analysis_file["Data"]
    metadata_dataset = analysis_file["metadata"]
    number_of_timepoints, number_of_trials, number_of_pixels = np.shape(activity_dataset)
    print("metadata_dataset", np.shape(metadata_dataset))
    """
    # Open Analysis Dataframe
    analysis_file = tables.open_file(os.path.join(tensor_directory, analysis_name + "_Trialwise_.h5"), mode="r")
    activity_dataset = analysis_file.root["Data"]
    metadata_dataset = analysis_file.root["Trial_Details"]

    # Create P and Slope Tensors
    p_value_tensor = np.ones((number_of_timepoints, number_of_pixels))
    slope_tensor = np.zeros((number_of_timepoints, number_of_pixels))

    for timepoint_index in tqdm(range(number_of_timepoints), position=0, desc="Timepoint"):

        # Get Timepoint Data
        timepoint_activity = activity_dataset[timepoint_index]

        for pixel_index in tqdm(range(number_of_pixels), position=1, desc="Pixel", leave=True):

            # Package Into Dataframe
            pixel_activity = timepoint_activity[:, pixel_index]
            pixel_dataframe = repackage_data_into_dataframe(pixel_activity, metadata_dataset)

            # Fit Mixed Effects Model
            p_value, slope = mixed_effects_random_slope_and_intercept(pixel_dataframe)
            p_value_tensor[timepoint_index, pixel_index] = p_value
            slope_tensor[timepoint_index, pixel_index] = slope


    # Save These Tensors
    np.save(os.path.join(tensor_directory, analysis_name + "_p_value_tensor.npy"), p_value_tensor)
    np.save(os.path.join(tensor_directory, analysis_name + "_slope_tensor.npy"), slope_tensor)



def test_significance_window(tensor_directory, analysis_name, window):

    """
    This Test Is Run Pixelwise - All Brains Must Be In Same Pixel Space

    :return:
    Tensor of P Values
    """

    """
    # Open Analysis Dataframe
    analysis_file = h5py.File(os.path.join(tensor_directory, analysis_name + ".hdf5"), "r")
    activity_dataset = analysis_file["Data"]
    metadata_dataset = analysis_file["metadata"]
    number_of_timepoints, number_of_trials, number_of_pixels = np.shape(activity_dataset)
    print("metadata_dataset", np.shape(metadata_dataset))
    """


    # Open Analysis Dataframe
    analysis_file = tables.open_file(os.path.join(tensor_directory, analysis_name + "_Trialwise_.h5"), mode="r")
    activity_dataset = analysis_file.root["Data"]
    metadata_dataset = analysis_file.root["Trial_Details"]
    activity_dataset = np.array(activity_dataset)
    metadata_dataset = np.array(metadata_dataset)

    activity_dataset = np.nan_to_num(activity_dataset)

    number_of_trials, number_of_timepoints, number_of_pixels = np.shape(activity_dataset)


    print("Number of timepoints", number_of_timepoints)
    print("number of pixels", number_of_pixels)
    print("number of trials", number_of_trials)

    # Create P and Slope Tensors
    p_value_tensor = np.ones(number_of_pixels)
    slope_tensor = np.zeros(number_of_pixels)

    # Get Timepoint Data
    timepoint_activity = activity_dataset[:, window]
    print("Timepoint activity shape", np.shape(timepoint_activity))
    timepoint_activity = np.mean(timepoint_activity, axis=1)

    for pixel_index in tqdm(range(number_of_pixels), position=1, desc="Pixel", leave=False):

        # Package Into Dataframe
        pixel_activity = timepoint_activity[:, pixel_index]
        pixel_dataframe = repackage_data_into_dataframe(pixel_activity, metadata_dataset)

        # Fit Mixed Effects Model
        p_value, slope = mixed_effects_random_slope_and_intercept(pixel_dataframe)
        p_value_tensor[pixel_index] = p_value
        slope_tensor[pixel_index] = slope

    return p_value_tensor, slope_tensor

"""
# Load Analysis Details
analysis_name = "Unrewarded_Contextual_Modulation"
tensor_directory = r"/media/matthew/External_Harddrive_2/Control_Switching_Tensors_100"

window = list(range(10,14))
p_value_tensor, slope_tensor = test_significance_window(tensor_directory, analysis_name, window)

indicies, image_height, image_width = widefield_utils.load_tight_mask()
indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

slope_map = widefield_utils.create_image_from_data(slope_tensor, indicies, image_height, image_width)
plt.imshow(slope_map)
plt.show()

p_map = widefield_utils.create_image_from_data(p_value_tensor, indicies, image_height, image_width)
p_map = np.nan_to_num(p_map)
plt.imshow(p_map)
plt.show()
"""