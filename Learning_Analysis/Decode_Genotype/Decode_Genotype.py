import numpy as np
import os
import tables
import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import Decoding_Utils
import Balanced_Sampling


def get_activity_tensors(session_list, onset_file, start_window, stop_window, tensor_root_directory, remake_activity_tensor):

    # Create List To Hold Activity Tensors
    activity_tensor_list = []

    # Get Activity Tensor Name
    activity_tensor_name = onset_file.replace("_onsets.npy", "")
    activity_tensor_name = activity_tensor_name + "_Activity_Tensor.npy"
    print("Activity tensor name", activity_tensor_name)

    for session in session_list:
        print("session", session)

        session_tensor_directory = Decoding_Utils.check_save_directory(session, tensor_root_directory)
        session_tensor_file = os.path.join(session_tensor_directory, activity_tensor_name)
        print("Session tensor file", session_tensor_file)
        if not os.path.exists(session_tensor_file) or remake_activity_tensor == True:
            print("Getting activitry tensor")
            activity_tensor = Decoding_Utils.get_activity_tensor(session, onset_file, start_window, stop_window, tensor_root_directory)
        else:
            activity_tensor = np.load(session_tensor_file)

        activity_tensor_list.append(activity_tensor)

    return activity_tensor_list



def get_data_structure(combined_file):

    # Open Combined File
    file_container = tables.open_file(combined_file, "r")

    # Get Trial Numbers
    control_trials = []
    mutant_trials = []
    print("file container", file_container)
    for array in file_container.list_nodes(where="/Controls"):
        trials = np.shape(array)[0]
        control_trials.append(trials)

    for array in file_container.list_nodes(where="/Mutants"):
        trials = np.shape(array)[0]
        trial_length = np.shape(array)[1]
        mutant_trials.append(trials)

    print("Control Trials", control_trials)
    print("Mutant Trials", mutant_trials)

    file_container.close()

    return control_trials, mutant_trials, trial_length



def get_mean_response(array, baseline_correction=True):

    if baseline_correction == False:
        mean_response = np.mean(array, axis=0)

    elif baseline_correction == True:

        corrected_tensor = []
        for trial in array:
            trial_baseline = trial[0:10]
            trial_baseline = np.mean(trial_baseline, axis=0)
            trial = np.subtract(trial, trial_baseline)
            corrected_tensor.append(trial)

        corrected_tensor = np.array(corrected_tensor)
        mean_response = np.mean(corrected_tensor, axis=0)

    return mean_response


def visualise_data(combined_file, baseline_correction=True):

    save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Decoding_Analysis/Sanity_Check"

    # Get Drawing Functions
    indicies, image_height, image_width = Decoding_Utils.load_tight_mask()
    plt.ion()

    # Open File
    file_container = tables.open_file(combined_file, "r")

    control_mean_vectors = []
    mutant_mean_vectors = []

    for array in tqdm(file_container.list_nodes(where="/Controls")):
        mean_response = get_mean_response(array, baseline_correction)
        control_mean_vectors.append(mean_response)

    for array in tqdm(file_container.list_nodes(where="/Mutants")):
        mean_response = get_mean_response(array, baseline_correction)
        mutant_mean_vectors.append(mean_response)

    # Visualise Responses
    number_of_control_mice = len(control_mean_vectors)
    number_of_mutant_mice = len(mutant_mean_vectors)
    number_of_mice = np.max([number_of_control_mice, number_of_mutant_mice])
    number_of_timepoints = np.shape(control_mean_vectors[0])[0]
    print("Number Of Timepoints", number_of_timepoints)

    figure_1 = plt.figure()
    gridspec_1 = GridSpec(ncols=number_of_mice, nrows=2, figure=figure_1)
    vmin=0
    vmax = np.percentile(control_mean_vectors, 99)

    plt.ion()
    for timepoint_index in range(number_of_timepoints):

        for mouse_index in range(number_of_control_mice):
            data = control_mean_vectors[mouse_index][timepoint_index]
            data = Decoding_Utils.create_image_from_data(data, indicies, image_height, image_width)
            axis = figure_1.add_subplot(gridspec_1[0, mouse_index])
            axis = figure_1.add_subplot(gridspec_1[0, mouse_index])
            axis.imshow(data, vmin=vmin, vmax=vmax, cmap='inferno')
            axis.axis('off')

        for mouse_index in range(number_of_mutant_mice):
            data = mutant_mean_vectors[mouse_index][timepoint_index]
            data = Decoding_Utils.create_image_from_data(data, indicies, image_height, image_width)
            axis = figure_1.add_subplot(gridspec_1[1, mouse_index])
            axis.imshow(data, vmin=vmin, vmax=vmax, cmap='inferno')
            axis.axis('off')

        figure_1.suptitle(str(timepoint_index))
        plt.draw()
        plt.savefig(os.path.join(save_directory, str(timepoint_index).zfill(3) + ".png"))
        plt.pause(0.1)
        plt.clf()

    # Close File
    file_container.close()


def decode_genotype(combined_file, number_of_folds=5):

    # Get Data Structure
    control_trials, mutant_trials, number_of_timepoints = get_data_structure(combined_file)

    # Get Drawing Functions
    indicies, image_height, image_width = Decoding_Utils.load_tight_mask()
    cmap = Decoding_Utils.get_mussal_cmap()
    plt.ion()

    # Iterate Through Each Timepoint
    timepoint_score_list = []
    timepoint_wights_list = []
    for timepoint in range(number_of_timepoints):

        # Get A Sample Of The Data With Trials Balanced For Genotype and Mouse
        data, labels = Balanced_Sampling.get_data_sample(combined_file, control_trials, mutant_trials, timepoint)

        # Perform K Fold Cross Validation
        k_fold_object = StratifiedKFold(n_splits=number_of_folds, random_state=42, shuffle=True)

        # Iterate Through Each Split
        score_list = []
        weight_list = []
        for train_index, test_index in k_fold_object.split(data, y=labels):

            # Split Data Into Train and Test Sets
            data_train, data_test = data[train_index], data[test_index]
            labels_train, labels_test = labels[train_index], labels[test_index]

            # Train Model
            model = LogisticRegression(penalty='l2')
            model.fit(data_train, labels_train)

            # Test Model
            model_score = model.score(data_test, labels_test)

            # Add Score To Score List
            score_list.append(model_score)

            # Get Model Weights
            model_weights = model.coef_
            weight_list.append(model_weights)

        # Get Mean Score and Mean Model Weights
        mean_score = np.mean(score_list)

        weight_list = np.array(weight_list)
        mean_weights = np.mean(weight_list, axis=0)

        # View Weights
        weight_map = Decoding_Utils.create_image_from_data(mean_weights, indicies, image_height, image_width)
        weight_magnitude = np.max(np.abs(weight_map))
        plt.imshow(weight_map, vmax=weight_magnitude, vmin=-1 * weight_magnitude, cmap=cmap)
        plt.title("Timepoint: " + str(timepoint) + " Mean Score: " + str(np.around(mean_score, 2)))
        plt.savefig(os.path.join(r"/media/matthew/Expansion/Widefield_Analysis/Decoding_Analysis/Correct_Rejections_Genotype", str(timepoint).zfill(3) + ".png"))
        plt.draw()
        plt.pause(0.1)
        plt.clf()

        # Add To Timepoint Lists
        timepoint_score_list.append(mean_score)
        timepoint_wights_list.append(mean_weights)

        print("Timepoint: ", timepoint, "Mean Score: ", mean_score)

    plt.plot(timepoint_score_list)
    plt.show()


control_session_list = [r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_24_Discrimination_Imaging",
                        r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_25_Discrimination_Imaging",
                        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_22_Discrimination_Imaging",
                        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_24_Discrimination_Imaging",
                        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_09_Discrimination_Imaging",
                        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging"]

mutant_session_list = [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_09_Discrimination_Imaging",
                        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging",
                        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_14_Discrimination_Imaging",
                        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_15_Discrimination_Imaging",
                        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging",
                        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_08_Discrimination_Imaging"]


combined_file = r"/media/matthew/Expansion/Widefield_Analysis/Decoding_Analysis/Correct_Rejections_Combined_Activity_Tensor.h5"

# Visualsie Data As A Sanity Check
visualise_data(combined_file)

analysis_name = r"Correct_Rejections"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Decoding_Utils.load_analysis_container(analysis_name)
tensor_root_directory = r"/media/matthew/Expansion/Widefield_Analysis/Activity_Tensors"
print("Onset files", onset_files)
decode_genotype(combined_file)
