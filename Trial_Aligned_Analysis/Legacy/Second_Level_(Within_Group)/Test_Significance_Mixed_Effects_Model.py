import h5py
import tables
import numpy as np
import os
from tqdm import tqdm
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

from Widefield_Utils import widefield_utils
from Files import Session_List

# Remove This Later
import warnings
warnings.filterwarnings("ignore")



def mixed_effects_random_slope_and_intercept(dataframe):

    model = sm.MixedLM.from_formula("Data_Value ~ Condition", dataframe, re_formula="Condition", groups=dataframe["Mouse"])
    model_fit = model.fit()
    parameters = model_fit.params
    group_slope = parameters[1]
    p_value = model_fit.pvalues["Condition"]

    return p_value, group_slope



def repackage_data_into_dataframe(condition_1_pixel_data, condition_2_pixel_data, condition_1_metadata, condition_2_metadata, start_window, stop_window):

    # Combine_Into Dataframe
    dataframe = pd.DataFrame(dtype=np.float64)

    # Combine Lists
    condition_1_data = condition_1_pixel_data[start_window:stop_window]
    condition_2_data = condition_2_pixel_data[start_window:stop_window]

    condition_1_data = np.mean(condition_1_data, axis=0)
    condition_2_data = np.mean(condition_2_data, axis=0)

    datapoints_list = np.concatenate([condition_1_data, condition_2_data])
    mouse_list = np.concatenate([condition_1_metadata[:, 0], condition_2_metadata[:, 0]])
    session_list = np.concatenate([condition_1_metadata[:, 1], condition_2_metadata[:, 1]])
    condition_list = np.concatenate([np.zeros(len(condition_1_data)), np.ones(len(condition_2_data))])

    #print("Datapoint list", datapoints_list)
    dataframe["Data_Value"] = datapoints_list
    dataframe["Mouse"] = mouse_list
    dataframe["Session"] = session_list
    dataframe["Condition"] = condition_list

    return dataframe




def load_initial_trial_and_get_shape(base_directory, tensor_save_directory, condition_name):

    # Get Path Details
    mouse_name, session_name = widefield_utils.get_mouse_name_and_session_name(base_directory)

    # Load Activity Tensor
    activity_tensor = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, condition_name + "_Activity_Tensor.npy"), allow_pickle=True)

    # Get Shape
    number_of_trials, trial_length,  number_of_pixels = np.shape(activity_tensor)

    return trial_length, number_of_pixels


def create_intermediate_dataset(nested_session_list, tensor_save_directory, intermediate_file_location, condition_name, trial_length, number_of_pixels):

    # Create Dataframe
    dataframe = tables.open_file(filename=intermediate_file_location + "_" + condition_name + ".hdf5", mode="w")
    data_container = dataframe.create_earray(name="Data", where=dataframe.root,shape=(0, trial_length, number_of_pixels), atom=tables.Float32Atom())
    trial_detail_container = dataframe.create_earray(name="Trial_Details", where=dataframe.root, shape=(0, 2), atom=tables.UInt16Atom())

    mouse_index = 0
    session_index = 0
    for mouse in nested_session_list:
        for base_directory in tqdm(mouse):

            # Get Path Details
            mouse_name, session_name = widefield_utils.get_mouse_name_and_session_name(base_directory)

            # Load Activity Tensor
            activity_tensor = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, condition_name + "_Activity_Tensor.npy"), allow_pickle=True)

            # Add Data
            for trial in activity_tensor:
                data_container.append([trial])
                trial_detail_container.append([np.array([mouse_index, session_index])])

            data_container.flush()
            trial_detail_container.flush()

            session_index += 1
        mouse_index += 1

    dataframe.close()


def reshape_dataset_to_tables(intermediate_file_location, condition_name):

    # Open File
    data_file = tables.open_file(filename=intermediate_file_location + "_" + condition_name + ".hdf5", mode="r")
    data = data_file.root["Data"]
    data = np.array(data)
    trial_metadata = data_file.root["Trial_Details"]
    number_of_trials, number_of_timepoints, number_of_pixels = np.shape(data)

    # Reshape Data
    print("Data Shape", np.shape(data))
    data = np.moveaxis(data, [0,1,2], [2,1,0])
    print("New Data Shape", np.shape(data))

    # Create New Datafile
    reshaped_data_file = tables.open_file(filename=intermediate_file_location + "_" + condition_name + "_Reshaped.h5", mode="w")
    reshaped_data_container = reshaped_data_file.create_earray(name="Data", where=reshaped_data_file.root,shape=(0, number_of_timepoints, number_of_trials), atom=tables.Float32Atom())
    reshaped_trial_detail_container = reshaped_data_file.create_earray(name="Trial_Details", where=reshaped_data_file.root, shape=(0, 2), atom=tables.UInt16Atom())

    # Copy Data
    for timepoint in tqdm(data):
        reshaped_data_container.append([timepoint])

    for trial in trial_metadata:
        reshaped_trial_detail_container.append([trial])

    # Close Files
    reshaped_data_file.close()
    data_file.close()


def within_group_mixed_effects(nested_session_list, tensor_save_directory, condition_1_file, condition_2_file, intermediate_file_location, analysis_name):

    """
    Inputs - Nested List Of Structure:
    Nested Session List - list of mice, within each mouse list of sessions
    Condition 1 file - filename of activity tensor of condition 1
    condition 2 filke - filename of activity tensor of condition 2

    This Test Is Run Pixelwise - All Brains Must Be In Same Pixel Space

    :return:
    Tensor of P Values
    """

    # Get Condition Names
    condition_1_name = condition_1_file.replace('_onsets', '')
    condition_1_name = condition_1_name.replace('.npy', '')

    condition_2_name = condition_2_file.replace('_onsets', '')
    condition_2_name = condition_2_name.replace('.npy', '')

    # Get Details
    trial_length, number_of_pixels = load_initial_trial_and_get_shape(nested_session_list[0][0], tensor_save_directory, condition_1_name)

    # Step 1 - Combine Into A Single Dataframe
    # Dataframe Will Be Of The Shape (Trials, Timepoints, Pixels)
    # also include a second key for the random variables (Mouse, Session)

    #create_intermediate_dataset(nested_session_list, tensor_save_directory, intermediate_file_location, condition_1_name, trial_length, number_of_pixels)
    #create_intermediate_dataset(nested_session_list, tensor_save_directory, intermediate_file_location, condition_2_name, trial_length, number_of_pixels)

    #reshape_dataset_to_tables(intermediate_file_location, condition_1_name)
    #reshape_dataset_to_tables(intermediate_file_location, condition_2_name)


    # Step 2 Iterate Through Each pixel


    # Load Reshaped Data
    condition_1_file_container = tables.open_file(filename=intermediate_file_location + "_" + condition_1_name + "_Reshaped.h5", mode="r")
    condition_1_data = condition_1_file_container.root["Data"]
    condition_1_trial_details = condition_1_file_container.root["Trial_Details"]

    condition_2_file_container = tables.open_file(filename=intermediate_file_location + "_" + condition_2_name + "_Reshaped.h5", mode="r")
    condition_2_data = condition_2_file_container.root["Data"]
    condition_2_trial_details = condition_2_file_container.root["Trial_Details"]


    # Iterate Through Each Pixel
    number_of_pixels, number_of_timepoints, condition_1_trials = np.shape(condition_1_data)

    p_value_tensor = np.zeros(number_of_pixels)
    slope_tensor = np.zeros(number_of_pixels)

    start_window = 10
    stop_window = 38

    for pixel_index in tqdm(range(number_of_pixels)):
        condition_1_pixel_data = condition_1_data[pixel_index]
        condition_2_pixel_data = condition_2_data[pixel_index]

        dataframe = repackage_data_into_dataframe(condition_1_pixel_data, condition_2_pixel_data, condition_1_trial_details, condition_2_trial_details, start_window, stop_window)
        p_value, slope = mixed_effects_random_slope_and_intercept(dataframe)

        p_value_tensor[pixel_index] = p_value
        slope_tensor[pixel_index] = slope


    # Save These Tensors
    np.save(os.path.join(intermediate_file_location, analysis_name + "p_value_tensor.npy"), p_value_tensor)
    np.save(os.path.join(intermediate_file_location, analysis_name + "slope_tensor.npy"), slope_tensor)




intermediate_file_location = r"/media/matthew/29D46574463D2856/Significance_Testing/Mixed_Effects_Modelling/Combined_Data"
analysis_name = "Unrewarded_Contextual_Modulation"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)
tensor_save_directory = r"//media/matthew/Expansion/Widefield_Analysis/Learning_Analysis"

# Load Session List
nested_session_list = Session_List.control_switching_nested

within_group_mixed_effects(nested_session_list, tensor_save_directory, onset_files[0], onset_files[1], intermediate_file_location, analysis_name)
