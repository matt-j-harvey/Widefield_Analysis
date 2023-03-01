import os
from sklearn.decomposition import TruncatedSVD
import tables
import numpy as np
import pickle
import shutil
from distutils.dir_util import copy_tree
from datetime import datetime


def copy_mask_and_behaviour_data(base_directory, save_directory):

    # Copy Downsampled AI
    source_ai = os.path.join(base_directory, "Downsampled_AI_Matrix_Framewise.npy")
    destination_ai = os.path.join(save_directory, "Downsampled_AI_Matrix_Framewise.npy")
    shutil.copyfile(source_ai, destination_ai)

    # Copy Mask
    source_mask = os.path.join(base_directory, "Generous_Mask.npy")
    destination_mask = os.path.join(save_directory, "Generous_Mask.npy")
    shutil.copyfile(source_mask, destination_mask)

    # Copy Mask Dict
    source_mask_dict = os.path.join(base_directory, "Downsampled_mask_dict.npy")
    destination_mask_dict = os.path.join(save_directory, "Downsampled_mask_dict.npy")
    shutil.copyfile(source_mask_dict, destination_mask_dict)

    # Copy Stimuli Onsets
    source_stimuli_onsets = os.path.join(base_directory, "Stimuli_Onsets")
    destination_stimuli_onsets = os.path.join(save_directory, "Stimuli_Onsets")
    copy_tree(source_stimuli_onsets, destination_stimuli_onsets)

    # Copy Alignment Dict
    source_alignment_dict = os.path.join(base_directory, "Within_Mouse_Alignment_Dictionary.npy")
    destination_alignment_dict = os.path.join(save_directory, "Within_Mouse_Alignment_Dictionary.npy")
    shutil.copyfile(source_alignment_dict, destination_alignment_dict)


def perform_svd_decomposition(base_directory, save_directory, n_components=2000):
    time_start = datetime.now()

    # Load Delta F Matrix
    delta_f_matrix_file = os.path.join(base_directory, "Downsampled_Delta_F.h5")
    delta_f_container = tables.open_file(delta_f_matrix_file, mode='r')
    delta_f_matrix = delta_f_container.root["Data"]
    delta_f_matrix = np.nan_to_num(delta_f_matrix)

    # Create Model
    model = TruncatedSVD(n_components=n_components)

    # FIt and Transform Data
    transformed_data = model.fit_transform(delta_f_matrix)

    # Save Data
    np.save(os.path.join(save_directory, "SVD_Compressed_Delta_F.npy"), transformed_data)

    # Save Model
    model_filename = os.path.join(save_directory, "SVD_Model.sav")
    pickle.dump(model, open(model_filename, 'wb'))

    time_finished = datetime.now()
    print("Compressed Session: ", base_directory, "Start: ", time_start, "Finished", time_finished)


def get_session_list(mouse_directory):
    mouse_files = os.listdir(mouse_directory)

    mouse_session_list = []

    for filename in mouse_files:
        if "_Imaging" in filename:
            mouse_session_list.append(filename)

    return mouse_session_list


def run_svd_pipeline(source_root_direcory, destination_root_directory):

    # Iterate Through Each Mouse
    mouse_list = os.listdir(source_root_direcory)
    print("Mice", mouse_list)

    for mouse in mouse_list[1:]:

        # Get Mouse Save Root
        mouse_save_root = os.path.join(destination_root_directory, mouse)
        if not os.path.exists(mouse_save_root):
            os.mkdir(mouse_save_root)

        # Get Mouse Session List
        mouse_session_list = get_session_list(os.path.join(source_root_direcory, mouse))
        print("Mouse session list", mouse_session_list)

        for session in mouse_session_list:

            base_directory = os.path.join(source_root_direcory, mouse, session)
            save_directory = os.path.join(destination_root_directory, mouse, session)
            if not os.path.exists(save_directory):
                os.mkdir(save_directory)
            print("Base", base_directory, "Save", save_directory)

            # Copy Behaviour and Mask Info
            copy_mask_and_behaviour_data(base_directory, save_directory)

            # Perform SVD Compression
            perform_svd_decomposition(base_directory, save_directory)


source_root = r"//media/matthew/External_Harddrive_1/Neurexin_Data"
destination_root = r"//media/matthew/External_Harddrive_3/SVD_Compressed_Data/Neurexin_Data"
run_svd_pipeline(source_root, destination_root)