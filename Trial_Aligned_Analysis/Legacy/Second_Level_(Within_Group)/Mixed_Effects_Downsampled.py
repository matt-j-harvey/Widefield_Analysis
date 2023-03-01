import h5py
import tables
import numpy as np
import os
from tqdm import tqdm
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
from skimage.transform import resize
import resource
import sys
import shutil

from Widefield_Utils import widefield_utils
from Files import Session_List

# Remove This Later
import warnings
warnings.filterwarnings("ignore")


def downsample_mask_further(indicies, image_height, image_width, new_size=100):

    # Reconstruct To 2D
    template = np.zeros(image_height * image_width)
    template[indicies] = 1
    template = np.reshape(template, (image_height, image_width))

    # Downsample
    template = resize(template, (new_size, new_size))

    template = np.reshape(template, new_size * new_size)
    template = np.where(template > 0.5, 1, 0)
    indicies = np.nonzero(template)

    return indicies, new_size, new_size




def downsample_tensor(tensor, full_indicies, full_image_height, full_image_width, downsampled_indicies, downsampled_image_height, downsampled_image_width):

    tensor = np.nan_to_num(tensor)

    downsampled_tensor = []
    for trial in tensor:

        downsampled_trial = []
        for frame in trial:

            frame = widefield_utils.create_image_from_data(frame, full_indicies, full_image_height, full_image_width)
            frame = resize(frame, (downsampled_image_height, downsampled_image_width))
            frame = np.reshape(frame, downsampled_image_height * downsampled_image_width)
            frame = frame[downsampled_indicies]
            downsampled_trial.append(frame)
        downsampled_tensor.append(downsampled_trial)

    downsampled_tensor = np.array(downsampled_tensor)

    return downsampled_tensor



def smooth_tensor(tensor, indicies, image_height, image_width):

    tensor = np.nan_to_num(tensor)

    smoothed_tensor = []
    for trial in tensor:

        smoothed_trial = []
        for frame in trial:

            frame = widefield_utils.create_image_from_data(frame, indicies, image_height, image_width)
            frame = ndimage.gaussian_filter(frame, sigma=2)
            frame = np.reshape(frame, image_height * image_width)
            frame = frame[indicies]
            smoothed_trial.append(frame)
        smoothed_tensor.append(smoothed_trial)

    smoothed_tensor = np.array(smoothed_tensor)

    return smoothed_tensor

def mixed_effects_random_slope_and_intercept(dataframe):
    model = sm.MixedLM.from_formula("Data_Value ~ Condition", dataframe, re_formula="Condition", groups=dataframe["Mouse"])
    model_fit = model.fit(method='lbfgs')
    parameters = model_fit.params
    group_slope = parameters[1]
    p_value = model_fit.pvalues["Condition"]
    z_stat = model_fit.tvalues["Condition"]
    return p_value, group_slope, z_stat



def repackage_data_into_dataframe(condition_1_pixel_data, condition_2_pixel_data, condition_1_metadata, condition_2_metadata, start_window, stop_window):

    # Combine_Into Dataframe
    dataframe = pd.DataFrame(dtype=np.float64)

    # Combine Lists
    condition_1_data = condition_1_pixel_data[start_window:stop_window]
    condition_2_data = condition_2_pixel_data[start_window:stop_window]


    condition_1_data = np.mean(condition_1_data, axis=0)
    condition_2_data = np.mean(condition_2_data, axis=0)

    datapoints_list = np.concatenate([condition_1_data, condition_2_data])
    mouse_list = np.concatenate([condition_1_metadata, condition_2_metadata])
    condition_list = np.concatenate([np.zeros(len(condition_1_data)), np.ones(len(condition_2_data))])

    #print("Datapoint list", datapoints_list)
    dataframe["Data_Value"] = datapoints_list
    dataframe["Mouse"] = mouse_list
    dataframe["Condition"] = condition_list

    return dataframe



def load_initial_trial_and_get_shape(base_directory, tensor_save_directory, condition_name):

    # Get Path Details
    mouse_name, session_name = widefield_utils.get_mouse_name_and_session_name(base_directory)

    # Load Activity Tensor
    activity_tensor = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, condition_name + ".npy"), allow_pickle=True)

    # Get Shape
    number_of_trials, trial_length,  number_of_pixels = np.shape(activity_tensor)

    return trial_length, number_of_pixels


def add_mouse_data_to_dataframe(mouse_sesssions, mouse_index, data_container, trial_detail_container, tensor_save_directory, condition_name, full_indicies, full_image_height, full_image_width, downsampled_indicies, downsampled_image_height, downsampled_image_width):


    print("Mouse Sessions", mouse_sesssions)
    for base_directory in mouse_sesssions:

        # Get Path Details
        mouse_name, session_name = widefield_utils.get_mouse_name_and_session_name(base_directory)
        print("Mouse name", mouse_name, "session", session_name)

        # Load Activity Tensor
        activity_tensor = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, condition_name + ".npy"), allow_pickle=True)

        # Smooth Activity Tensor
        activity_tensor = downsample_tensor(activity_tensor, full_indicies, full_image_height, full_image_width, downsampled_indicies, downsampled_image_height, downsampled_image_width)

        # Add Data
        for trial in activity_tensor:
            data_container.append([trial])
            trial_detail_container.append([mouse_index])

        data_container.flush()
        trial_detail_container.flush()


def create_intermediate_dataset(nested_session_list, tensor_save_directory, intermediate_file_location, condition_name, trial_length, group_names, full_indicies, full_image_height, full_image_width, downsampled_indicies, downsampled_image_height, downsampled_image_width):
    print("Creating Intermediate Session List")

    number_of_pixels = np.shape(downsampled_indicies)[1]
    print("number of pixels", number_of_pixels)

    # Create Dataframes
    group_1_dataframe = tables.open_file(filename=os.path.join(intermediate_file_location, condition_name + "_" + group_names[0] + ".hdf5"), mode="w")
    group_1_data_container = group_1_dataframe.create_earray(name="Data", where=group_1_dataframe.root,shape=(0, trial_length, number_of_pixels), atom=tables.Float32Atom())
    group_1_trial_detail_container = group_1_dataframe.create_earray(name="Trial_Details", where=group_1_dataframe.root, shape=(0, ), atom=tables.UInt16Atom())

    group_2_dataframe = tables.open_file(filename=os.path.join(intermediate_file_location, condition_name + "_" + group_names[1] + ".hdf5"), mode="w")
    group_2_data_container = group_2_dataframe.create_earray(name="Data", where=group_2_dataframe.root, shape=(0, trial_length, number_of_pixels), atom=tables.Float32Atom())
    group_2_trial_detail_container = group_2_dataframe.create_earray(name="Trial_Details", where=group_2_dataframe.root, shape=(0, ), atom=tables.UInt16Atom())

    mouse_index = 0
    for mouse in tqdm(nested_session_list):
        add_mouse_data_to_dataframe(mouse[0], mouse_index, group_1_data_container, group_1_trial_detail_container, tensor_save_directory, condition_name, full_indicies, full_image_height, full_image_width, downsampled_indicies, downsampled_image_height, downsampled_image_width)
        add_mouse_data_to_dataframe(mouse[1], mouse_index, group_2_data_container, group_2_trial_detail_container, tensor_save_directory, condition_name, full_indicies, full_image_height, full_image_width, downsampled_indicies, downsampled_image_height, downsampled_image_width)
        mouse_index += 1

    group_1_dataframe.close()
    group_2_dataframe.close()



def reshape_dataset_to_tables(intermediate_file_location, condition_name, group_name):
    print("Reshaping Data")

    # Open File
    data_file = tables.open_file(filename=os.path.join(intermediate_file_location, condition_name + "_" + group_name + ".hdf5"), mode="r")
    data = data_file.root["Data"]
    data = np.array(data)
    trial_metadata = data_file.root["Trial_Details"]
    number_of_trials, number_of_timepoints, number_of_pixels = np.shape(data)
    print("Number of trials", number_of_trials, "Number of timepoints", number_of_timepoints)

    # Reshape Data
    print("Data Shape", np.shape(data))
    data = np.moveaxis(data, [0,1,2], [2,1,0])
    print("New Data Shape", np.shape(data))

    # Create New Datafile
    reshaped_data_file = tables.open_file(filename=os.path.join(intermediate_file_location, condition_name + "_" + group_name + "_Reshaped.h5"), mode="w")
    reshaped_data_container = reshaped_data_file.create_earray(name="Data", where=reshaped_data_file.root,shape=(0, number_of_timepoints, number_of_trials), atom=tables.Float32Atom())
    reshaped_trial_detail_container = reshaped_data_file.create_earray(name="Trial_Details", where=reshaped_data_file.root, shape=(0, ), atom=tables.UInt16Atom())

    # Copy Data
    for timepoint in tqdm(data):
        reshaped_data_container.append([timepoint])

    for trial in trial_metadata:
        reshaped_trial_detail_container.append([trial])

    # Close Files
    reshaped_data_file.close()
    data_file.close()



def paired_sessions_mixed_effects(nested_session_list, tensor_save_directory, condition_file, intermediate_file_location, analysis_name, group_names, full_indicies, full_image_height, full_image_width, downsampled_indicies, downsampled_image_height, downsampled_image_width):

    """
    Inputs - Nested List Of Structure:
    List - Mouse - Condition 1 sessions, Condition 2 sessions


    This Test Is Run Pixelwise - All Brains Must Be In Same Pixel Space

    :return:
    Tensor of P Values
    """

    # Get Condition Names
    condition_name = condition_file.replace('_onsets', '')
    condition_name = condition_name.replace('.npy', '')


    # Get Details
    trial_length, number_of_pixels = load_initial_trial_and_get_shape(nested_session_list[0][0][0], tensor_save_directory, condition_name)

    """
    # Step 1 - Combine Into A Single Dataframe
    # Dataframe Will Be Of The Shape (Trials, Timepoints, Pixels)
    # also include a second key for the random variables (Mouse)
    create_intermediate_dataset(nested_session_list, tensor_save_directory, intermediate_file_location, condition_name, trial_length, group_names, full_indicies, full_image_height, full_image_width, downsampled_indicies, downsampled_image_height, downsampled_image_width)
    reshape_dataset_to_tables(intermediate_file_location, condition_name, group_names[0])
    reshape_dataset_to_tables(intermediate_file_location, condition_name, group_names[1])
    """

    # Step 2 Iterate Through Each pixel

    # Load Reshaped Data
    group_1_file_container = tables.open_file(filename=os.path.join(intermediate_file_location, condition_name + "_" + group_names[0] + "_Reshaped.h5"), mode="r")
    group_1_data = group_1_file_container.root["Data"]
    group_1_trial_details = group_1_file_container.root["Trial_Details"]

    group_2_file_container = tables.open_file(filename=os.path.join(intermediate_file_location, condition_name + "_" + group_names[1] + "_Reshaped.h5"), mode="r")
    group_2_data = group_2_file_container.root["Data"]
    group_2_trial_details = group_2_file_container.root["Trial_Details"]

    print("Group 1 data shape", np.shape(group_1_data))
    print("Group 2 data shape", np.shape(group_2_data))

    # Iterate Through Each Pixel
    number_of_pixels, number_of_timepoints, condition_1_trials = np.shape(group_1_data)

    p_value_tensor = np.zeros(number_of_pixels)
    slope_tensor = np.zeros(number_of_pixels)
    t_stat_tensor = np.zeros(number_of_pixels)

    start_window = 10
    stop_window = 24 #52

    for pixel_index in tqdm(range(number_of_pixels)):

        condition_1_pixel_data = group_1_data[pixel_index]
        condition_2_pixel_data = group_2_data[pixel_index]

        condition_1_pixel_data = np.nan_to_num(condition_1_pixel_data)
        condition_2_pixel_data = np.nan_to_num(condition_2_pixel_data)

        dataframe = repackage_data_into_dataframe(condition_1_pixel_data, condition_2_pixel_data, group_1_trial_details, group_2_trial_details, start_window, stop_window)
        p_value, slope, t_stat = mixed_effects_random_slope_and_intercept(dataframe)

        p_value_tensor[pixel_index] = p_value
        slope_tensor[pixel_index] = slope
        t_stat_tensor[pixel_index] = t_stat

    # Save These Tensors
    np.save(os.path.join(intermediate_file_location, analysis_name + "_p_value_tensor.npy"), p_value_tensor)
    np.save(os.path.join(intermediate_file_location, analysis_name + "_slope_tensor.npy"), slope_tensor)
    np.save(os.path.join(intermediate_file_location, analysis_name + "_t_stat_tensor.npy"), t_stat_tensor)


def rename_sessions(nested_list, tensor_save_directory):

    for mouse in nested_list:
        for condition in mouse:
            for base_directory in condition:

                # Get Path Details
                mouse_name, session_name = widefield_utils.get_mouse_name_and_session_name(base_directory)
                print("Mouse name", mouse_name, "session", session_name)

                # Load Activity Tensor
                origional_filename = os.path.join(tensor_save_directory, mouse_name, session_name, "Mixed_Effects_Distribution_Matched_Onsets.npy_Activity_Tensor_Aligned_Across_Mice.npy")

                if os.path.exists(origional_filename):
                    target = os.path.join(tensor_save_directory, mouse_name, session_name, "Mixed_Effects_Distribution_Matched_Activity_Tensor_Aligned_Across_Mice.npy")
                    shutil.copy(origional_filename, target)
                    print(target)


# Load Mask
full_indicies, full_image_height, full_image_width = widefield_utils.load_tight_mask()

# Downsample Further
downsampled_indicies, downsampled_image_height, downsampled_image_width = downsample_mask_further(full_indicies, full_image_height, full_image_width)


# Load Analysis Details
analysis_name = "Hits_Pre_Post_Learning_response"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)
tensor_save_directory = r"//media/matthew/Expansion/Widefield_Analysis/Learning_Analysis"
group_names = ["Pre_Learning", "Post_Learning"]

onset_files = ["Mixed_Effects_Distribution_Matched_Activity_Tensor_Aligned_Across_Mice.npy"]


"""
### Controls Learning ###
significance_testing_folder = r"/media/matthew/29D46574463D2856/Significance_Testing/Control_Learning_Downsampled"

# Load Session List
control_learning_tuples = Session_List.expanded_controls_learning_nested

# Perform Mixed Effects Modelling
paired_sessions_mixed_effects(control_learning_tuples, tensor_save_directory, onset_files[0], significance_testing_folder, analysis_name, group_names, full_indicies, full_image_height, full_image_width, downsampled_indicies, downsampled_image_height, downsampled_image_width)
"""



### Mutants Learning ###
significance_testing_folder = r"/media/matthew/29D46574463D2856/Significance_Testing/Mutant_Learning_Downsampled"

# Load Session List
mutant_learning_tuples = Session_List.expanded_mutants_learning_nested

# Perform Mixed Effects Modelling
paired_sessions_mixed_effects(mutant_learning_tuples, tensor_save_directory, onset_files[0], significance_testing_folder, analysis_name, group_names, full_indicies, full_image_height, full_image_width, downsampled_indicies, downsampled_image_height, downsampled_image_width)
