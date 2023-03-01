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
    Level 1 - Mouse
    Level 2 - Condition
    Level 3 - Session

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
    metadata_dataset = analysis_dataset_file.create_earray(analysis_dataset_file.root, 'Trial_Details', tables.UInt16Atom(), shape=(0, 3))

    # Iterate Through Each Session
    mouse_index = 0
    session_index = 0

    for mouse in nested_session_list:
        learning_index = 0
        for learning_stage in mouse:
            for session in learning_stage:
                for condition in condition_list:

                    # Get Tensor Name
                    tensor_name = condition.replace("_onsets.npy", "")
                    tensor_name = tensor_name.replace("_onset_frames.npy", "")

                    # Open Trial Tensor
                    session_trial_tensor_dict_path = os.path.join(tensor_directory, session, tensor_name)
                    with open(session_trial_tensor_dict_path + ".pickle", 'rb') as handle:
                        session_trial_tensor_dict = pickle.load(handle)
                        activity_tensor = session_trial_tensor_dict["activity_tensor"]

                    # Add Data To Dataset
                    for trial in activity_tensor:
                        activity_dataset.append([trial])
                        metadata_dataset.append([np.array([mouse_index, session_index, learning_index])])

                    # Flush File
                    analysis_dataset_file.flush()

                    # Increment Counters
                session_index += 1
            learning_index += 1
        mouse_index += 1
    analysis_dataset_file.close()

    """
    # Reshape Into Timepoint Wide Dataframe
    trialwise_file = tables.open_file(filename=os.path.join(tensor_directory, analysis_name + "_Trialwise_.h5"), mode="r")
    trialwise_activity_dataset = trialwise_file.root['Data']
    trialwise_metadata_dataset = trialwise_file.root['Trial_Details']
    number_of_trials, number_of_timepoints, number_of_pixels = np.shape(trialwise_activity_dataset)

    with h5py.File(os.path.join(tensor_directory, analysis_name + ".hdf5"), "w") as f:
        activity_dataset = f.create_dataset("Data", (number_of_timepoints, number_of_trials, number_of_pixels), dtype=np.float32, chunks=True, compression="gzip")
        metadata_dataset = f.create_dataset("metadata", (number_of_trials, 4), dtype=np.uint16, chunks=True, compression="gzip")

        for trial_index in tqdm(range(number_of_trials)):
            trial_data = trialwise_activity_dataset[trial_index]
            trial_metadata = trialwise_metadata_dataset[trial_index]

            # Write To Dataset
            activity_dataset[:, trial_index] = trial_data
            metadata_dataset[trial_index] = trial_metadata
    """


