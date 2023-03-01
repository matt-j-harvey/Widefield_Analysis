import tables
import h5py
import os
import numpy as np
import pickle
from tqdm import tqdm
from Widefield_Utils import widefield_utils



def create_analysis_dataset(tensor_directory, nested_session_list, condition_list, analysis_name, start_window, stop_window):

    """
    Nested Session List Should Be The Following Structure
    group, mouse, learning_stage, condition

    Level 1 - Group
    Level 2 - Mouse
    Level 3 - Learning Stage
    Level 4 - Condition

    First Create An Intermediate HDF5 Dataset - With Trial As The First Dimension
    Then Reshape This to A h5 Dataset - With Timepoint As The First Dimension
    """

    # Get Tensor Details
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

    number_of_timepoints = stop_window - start_window
    number_of_pixels = np.shape(indicies)[1]
    print("Nubmer of pixels", number_of_pixels)
    print("Number of timepoints", number_of_timepoints)

    # Open File
    analysis_dataset_file = tables.open_file(filename=os.path.join(tensor_directory, analysis_name + "_Trialwise_.h5"), mode="w")
    activity_dataset = analysis_dataset_file.create_earray(analysis_dataset_file.root, 'Data', tables.Float32Atom(), shape=(0, number_of_timepoints, number_of_pixels))
    metadata_dataset = analysis_dataset_file.create_earray(analysis_dataset_file.root, 'Trial_Details', tables.UInt16Atom(), shape=(0, 4))

    # Get Regressor Data Structure
    number_of_regressors = len(condition_list)
    trial_length = stop_window - start_window

    # Iterate Through Each Session
    mouse_index = 0
    session_index = 0
    for group in nested_session_list:
        for mouse in group:
            learning_index = 0
            for learning_stage in mouse:
                for session in learning_stage:

                    # Load Regressor Dict
                    full_tensor_directory = os.path.join(tensor_directory, session)
                    print("Full tensor directory", full_tensor_directory, "Learning stage", learning_index)
                    regressor_dict = np.load(os.path.join(full_tensor_directory, "Regression_Dictionary_Simple.npy"), allow_pickle=True)[()]
                    regression_coefs = regressor_dict['Coefs']
                    regression_coefs = np.transpose(regression_coefs)

                    for condition_index in range(number_of_regressors):

                        # Get Stimuli Regressor For Each Condition
                        condition_regressor_start = condition_index * trial_length
                        condition_regressor_stop = condition_regressor_start + trial_length
                        condition_regressors = regression_coefs[condition_regressor_start:condition_regressor_stop]
                        print("Condition regressors", np.shape(condition_regressors))

                        activity_dataset.append([condition_regressors])
                        metadata_dataset.append([np.array([0, mouse_index, learning_index, condition_index])])

                        # Flush File
                        analysis_dataset_file.flush()

                        # Increment Counters
                    session_index += 1
                learning_index += 1

            mouse_index += 1
    analysis_dataset_file.close()
