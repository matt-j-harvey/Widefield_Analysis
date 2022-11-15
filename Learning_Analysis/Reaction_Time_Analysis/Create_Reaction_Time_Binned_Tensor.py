import numpy as np
import matplotlib.pyplot as plt
import tables
from tqdm import tqdm
import os

import RT_Strat_Utils


def get_array_name(base_directory):
    split_base_directory = os.path.normpath(base_directory)
    split_base_directory = split_base_directory.split(os.sep)
    mouse_name = split_base_directory[-2]
    session_name = split_base_directory[-1]
    array_name = mouse_name + "_" + session_name
    return array_name



def reconstruct_activity_tensor(activity_tensor, indicies, image_height, image_width, alignment_dictionary):

    reconstructed_tensor = []

    for trial in activity_tensor:
        reconstructed_trial = []

        for frame in trial:

            # Reconstruct Image
            frame = RT_Strat_Utils.create_image_from_data(frame, indicies, image_height, image_width)

            # Align Image
            frame = RT_Strat_Utils.transform_image(frame, alignment_dictionary)

            reconstructed_trial.append(frame)
        reconstructed_tensor.append(reconstructed_trial)

    reconstructed_tensor = np.array(reconstructed_tensor)
    return reconstructed_tensor


def apply_shared_tight_mask(activity_tensor):

    # Load Tight Mask
    indicies, image_height, image_width = RT_Strat_Utils.load_tight_mask()

    transformed_tensor = []
    for trial in activity_tensor:
        transformed_trial = []

        for frame in trial:
            frame = np.ndarray.flatten(frame)
            frame = frame[indicies]
            transformed_trial.append(frame)
        transformed_tensor.append(transformed_trial)

    transformed_tensor = np.array(transformed_tensor)
    return transformed_tensor


def get_activity_tensor(activity_matrix, onsets, start_window, stop_window, start_cutoff=3000):

    number_of_pixels = np.shape(activity_matrix)[1]
    number_of_trials = np.shape(onsets)[0]
    number_of_timepoints = np.shape(activity_matrix)[0]

    # Create Empty Tensor To Hold Data
    activity_tensor = []

    # Get Correlation Matrix For Each Trial
    for trial_index in range(number_of_trials):

        # Get Trial Activity
        trial_start = onsets[trial_index] + start_window
        trial_stop = onsets[trial_index] + stop_window

        if trial_start > start_cutoff and trial_stop < number_of_timepoints:
            trial_activity = activity_matrix[trial_start:trial_stop]
            activity_tensor.append(trial_activity)

    activity_tensor = np.array(activity_tensor)
    return activity_tensor


def create_rt_stratified_tensor(base_directory, stratified_onsets, start_window, stop_window):

    # Load Mask
    indicies, image_height, image_width = RT_Strat_Utils.load_generous_mask(base_directory)

    # Load Alignment Dictionary
    alignment_dictionary = np.load(os.path.join(base_directory, "Cluster_Alignment_Dictionary.npy"), allow_pickle=True)[()]

    # Load Activity Matrix
    delta_f_file = os.path.join(base_directory, "Delta_F.h5")
    delta_f_container = tables.open_file(delta_f_file, "r")
    activity_matrix = delta_f_container.root.Data

    stratified_tensor = []
    for onset_group in stratified_onsets:

        if len(onset_group) > 0:

            # Get Activity Tensor
            activity_tensor = get_activity_tensor(activity_matrix, onset_group, start_window, stop_window)

            # Reconstruct Into Local Brain Space
            activity_tensor = reconstruct_activity_tensor(activity_tensor, indicies, image_height, image_width, alignment_dictionary)

            # Apply Shared Tight Mask
            activity_tensor = apply_shared_tight_mask(activity_tensor)

            stratified_tensor.append(activity_tensor)

        else:
            stratified_tensor.append(None)

    # Close Delta F File
    delta_f_container.close()

    return stratified_tensor



def create_reaction_time_binned_tensor(control_session_list, mutant_session_list, bin_list_file, control_binned_matrix, mutant_binned_matrix, start_window, stop_window, save_directory):

    # Load Bin List
    bin_list = np.load(bin_list_file)
    print("Bin List: ", bin_list)


    # Load Onset Binned Matricies
    control_binned_matrix = np.load(control_binned_matrix, allow_pickle=True)
    mutant_binned_matrix = np.load(mutant_binned_matrix, allow_pickle=True)

    number_of_time_bins, number_of_control_sessions = np.shape(control_binned_matrix)
    number_of_time_bins, number_of_mutant_sessions = np.shape(mutant_binned_matrix)

    print("Number of time bins", number_of_time_bins)
    print("Number of control sessions", number_of_control_sessions)
    print("Number Of Mutant Sessions", number_of_mutant_sessions)

    # Create Table Files
    file_list = []
    group_list = []
    for time_bin in bin_list:
        file_name = os.path.join(save_directory, str(time_bin).zfill(4) + ".h5")
        file_container = tables.open_file(file_name, mode='w')
        file_list.append(file_container)

        control_group = file_container.create_group(where="/", name="Controls")
        mutant_group = file_container.create_group(where="/", name="Mutants")
        group_list.append([control_group, mutant_group])

    # Iterate Through Control Mice
    for control_session_index in tqdm(range(number_of_control_sessions)):
        print("Control Session: ", control_session_index)

        # Get Onsets List
        session_onsets_stratified = control_binned_matrix[:, control_session_index]
        print("Stratified Onsets", session_onsets_stratified)

        # Get Stratified Tensor
        base_directory = control_session_list[control_session_index]
        stratified_tensor = create_rt_stratified_tensor(base_directory, session_onsets_stratified, start_window, stop_window)
        print("Stratified Tensr", np.shape(stratified_tensor))

        for time_bin_index in range(number_of_time_bins):
            array_name = get_array_name(base_directory)
            time_bin_array = stratified_tensor[time_bin_index]

            if time_bin_array is not None:
                print("Array Shape", np.shape(time_bin_array))
                selected_group = group_list[time_bin_index]
                tensor_storage = file_container.create_carray(selected_group[0], array_name, tables.UInt16Atom(), shape=(np.shape(time_bin_array)))
                tensor_storage[:] = time_bin_array
                tensor_storage.flush()

    # Iterate Through Mutant Mice
    for mutant_session_index in tqdm(range(number_of_mutant_sessions)):
        print("Mutant Session: ", mutant_session_index)

        # Get Onsets List
        session_onsets_stratified = mutant_binned_matrix[:, mutant_session_index]
        print("Stratified Onsets", session_onsets_stratified)

        # Get Stratified Tensor
        base_directory = mutant_session_list[mutant_session_index]
        stratified_tensor = create_rt_stratified_tensor(base_directory, session_onsets_stratified, start_window, stop_window)
        print("Stratified Tensor", np.shape(stratified_tensor))

        for time_bin_index in range(number_of_time_bins):
            array_name = get_array_name(base_directory)
            time_bin_array = stratified_tensor[time_bin_index]

            if time_bin_array is not None:
                print("Array Shape", np.shape(time_bin_array))
                selected_group = group_list[time_bin_index]
                tensor_storage = file_container.create_carray(selected_group[1], array_name, tables.UInt16Atom(), shape=(np.shape(time_bin_array)))
                tensor_storage[:] = time_bin_array
                tensor_storage.flush()


    # Close Open Files
    for file_container in file_list:
        file_container.close()






control_session_list = [r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_14_Discrimination_Imaging",
                        r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging",

                        r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_14_Discrimination_Imaging",
                        r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_15_Discrimination_Imaging",

                        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_04_Discrimination_Imaging",
                        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_06_Discrimination_Imaging",

                        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_01_Discrimination_Imaging",
                        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_03_Discrimination_Imaging",

                        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_04_29_Discrimination_Imaging",
                        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_01_Discrimination_Imaging",

                        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_25_Discrimination_Imaging",
                        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_29_Discrimination_Imaging",

                        ]

mutant_session_list = [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_13_Discrimination_Imaging",
                       r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_14_Discrimination_Imaging",

                       r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_02_Discrimination_Imaging",
                       r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_04_Discrimination_Imaging",

                       r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_04_30_Discrimination_Imaging",
                       r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_02_Discrimination_Imaging",

                       r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_04_30_Discrimination_Imaging",
                       r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_02_Discrimination_Imaging",

                       r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_28_Discrimination_Imaging",
                       r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_30_Discrimination_Imaging",

                       r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_20_Discrimination_Imaging",
                       r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_22_Discrimination_Imaging",
                       ]


control_binned_matrix = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Early_Learning/Control_Vis_1_RT_Binned.npy"
mutant_binned_matrix = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Early_Learning/Mutant_Vis_1_RT_Binned.npy"
bin_list_file = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Early_Learning/Bin_List.npy"
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Early_Learning/RT_Stratified_Tensors"

start_window = -10
stop_window = 60
create_reaction_time_binned_tensor(control_session_list, mutant_session_list, bin_list_file, control_binned_matrix, mutant_binned_matrix, start_window, stop_window, save_directory)