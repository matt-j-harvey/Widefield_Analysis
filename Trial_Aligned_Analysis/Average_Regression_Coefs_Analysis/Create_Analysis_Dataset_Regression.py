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
    Level 1 - Group
    Level 2 - Mice
    Level 3 - Condition
    """

    # Get Tensor Details
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

    number_of_timepoints = stop_window - start_window
    number_of_pixels = np.shape(indicies)[1]
    print("Nubmer of pixels", number_of_pixels)

    # Open File
    analysis_dataset_file = tables.open_file(filename=os.path.join(tensor_directory, analysis_name + "_Trialwise_.h5"), mode="w")
    activity_dataset = analysis_dataset_file.create_earray(analysis_dataset_file.root, 'Data', tables.Float32Atom(), shape=(0, number_of_timepoints, number_of_pixels))
    metadata_dataset = analysis_dataset_file.create_earray(analysis_dataset_file.root, 'Trial_Details', tables.UInt16Atom(), shape=(0, 3))

    # Iterate Through Each Session
    group_index = 0
    mouse_index = 0

    # Get Regressor Data Structure
    number_of_regressors = len(condition_list)
    trial_length = stop_window - start_window

    for group in nested_session_list:
        for mouse in group:
            for session in mouse:

                # Load Regressor Dict
                full_tensor_directory = os.path.join(tensor_directory, session)
                regressor_dict = np.load(os.path.join(full_tensor_directory, "Regression_Dictionary_Simple.npy"), allow_pickle=True)[()]
                regression_coefs = regressor_dict['Coefs']
                regression_coefs = np.transpose(regression_coefs)
                print("Regression Coefs", np.shape(regression_coefs))

                for condition_index in range(number_of_regressors):

                    # Get Stimuli Regressor For Each Condition
                    condition_regressor_start = int(condition_index * trial_length)
                    condition_regressor_stop = int(condition_regressor_start + trial_length)

                    print("Condiion Regressor Start", condition_regressor_start, "Condition regressor stop", condition_regressor_stop)
                    condition_regressors = regression_coefs[condition_regressor_start:condition_regressor_stop]

                    print("Condition Regressors", np.shape(condition_regressors))

                    activity_dataset.append([condition_regressors])
                    metadata_dataset.append([np.array([group_index, mouse_index, condition_index])])

                    # Flush File
                    analysis_dataset_file.flush()

            mouse_index += 1
        group_index += 1
    analysis_dataset_file.close()


