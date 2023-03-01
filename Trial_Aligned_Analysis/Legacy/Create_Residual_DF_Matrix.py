import h5py
import numpy as np
import tables
import os
from tqdm.auto import tqdm

from Widefield_Utils import widefield_utils
from Files import Session_List


def extract_only_running_and_licking(base_directory, design_matrix):

    design_matrix_dict = np.load(os.path.join(base_directory, "Ride_Regression", "design_matrix_key_dict.npy"), allow_pickle=True)[()]
    group_names = design_matrix_dict["Group Names"]
    group_sizes = design_matrix_dict["Group Sizes"]
    print("Group Names", group_names)
    print("Group sizes", group_sizes)

    only_lick_and_running = design_matrix[:, [0, group_sizes[0]]]
    print("Only lickig and running", np.shape(only_lick_and_running))

    return only_lick_and_running


def create_residual_matrix(base_directory, early_cutoff=3000):

    # Load Delta F Matrix
    delta_f_file = os.path.join(base_directory, "Downsampled_Delta_F.h5")
    delta_f_file_container = tables.open_file(delta_f_file, "r")
    delta_f_matrix = delta_f_file_container.root.Data
    delta_f_matrix = np.array(delta_f_matrix)
    number_of_frames, number_of_pixels = np.shape(delta_f_matrix)

    # Load Design Matrix
    design_matrix = np.load(os.path.join(base_directory, "Ride_Regression", "Design_Matrix.npy"))
    design_matrix = extract_only_running_and_licking(base_directory, design_matrix)

    # Load Ridge Coefs and Intercepts
    regression_dict = np.load(os.path.join(base_directory, "Ride_Regression", "Regression_Dictionary_Simple.npy"), allow_pickle=True)[()]
    regression_coefs = regression_dict["Coefs"]
    regression_intercepts = regression_dict["Intercepts"]

    # Get Chunk Structure
    chunk_size = 1000
    number_of_chunks, chunk_sizes, chunk_start_list, chunk_stop_list = widefield_utils.get_chunk_structure(chunk_size, number_of_pixels)

    # Create Output File
    residual_filepath = os.path.join(base_directory, "Residual_DF_Matrix.hdf5")
    with h5py.File(residual_filepath, "w") as f:
        residual_dataset = f.create_dataset("Residual_Data", (number_of_frames, number_of_pixels), dtype=np.float32, chunks=True, compression=True)

        # Get Residuals
        for chunk_index in tqdm(range(number_of_chunks), position=1, desc="Chunk: ", leave=True):

            # Get Chunk Data
            chunk_start = chunk_start_list[chunk_index]
            chunk_stop = chunk_stop_list[chunk_index]
            chunk_data = delta_f_matrix[:, chunk_start:chunk_stop]
            chunk_data = np.nan_to_num(chunk_data)

            # Get Predicted Data
            chunk_prediction = np.dot(design_matrix, np.transpose(regression_coefs[chunk_start:chunk_stop]))
            chunk_prediction = np.add(chunk_prediction, regression_intercepts[chunk_start:chunk_stop])

            # Get Residuals
            chunk_residuals = np.subtract(chunk_data, chunk_prediction)

            # Save Residuals
            residual_dataset[:, chunk_start:chunk_stop] = chunk_residuals

    delta_f_file_container.close()



selected_session_list = Session_List.mutant_transition_sessions
for x in range(10, len(selected_session_list)):
    create_residual_matrix(selected_session_list[x])
