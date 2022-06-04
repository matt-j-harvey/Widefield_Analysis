import numpy as np
import sklearn.svm
from sklearn.decomposition import NMF
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import os
import matplotlib.pyplot as plt
import sys
from matplotlib import cm
import h5py

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")
import Widefield_General_Functions


def perform_dimensionality_reduction(trial_tensor, n_components=3):

    # Get Tensor Shape
    number_of_trials = np.shape(trial_tensor)[0]
    trial_length = np.shape(trial_tensor)[1]
    number_of_neurons = np.shape(trial_tensor)[2]

    # Flatten Tensor To Perform Dimensionality Reduction
    reshaped_tensor = np.reshape(trial_tensor, (number_of_trials * trial_length, number_of_neurons))

    # Perform Dimensionality Reduction
    model = NMF(n_components=n_components)
    model.fit(reshaped_tensor)

    transformed_data = model.transform(reshaped_tensor)
    components = model.components_

    # Put Transformed Data Back Into Tensor Shape
    transformed_data = np.reshape(transformed_data, (number_of_trials, trial_length, n_components))

    return components, transformed_data



def get_trial_tensor(delta_f_matrix, onset_list, start_window, stop_window):

    trial_tensor = []
    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window
        trial_data = delta_f_matrix[trial_start:trial_stop]
        trial_tensor.append(trial_data)

    trial_tensor = np.array(trial_tensor)
    return trial_tensor




def load_data(base_directory, start_window=-10, stop_window=50):

    # Load Delta F Matrix
    delta_f_matrix_filepath = os.path.join(base_directory, "Delta_F.hdf5")
    delta_f_matrix_container = h5py.File(delta_f_matrix_filepath, 'r')
    delta_f_matrix = delta_f_matrix_container['Data']

    # Load Onsets
    vis_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "visual_1_all_onsets.npy"))
    vis_2_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "visual_2_all_onsets.npy"))

    condition_1_data = get_trial_tensor(delta_f_matrix, vis_1_onsets, start_window, stop_window)
    condition_2_data = get_trial_tensor(delta_f_matrix, vis_2_onsets, start_window, stop_window)
    combined_data = np.concatenate([condition_1_data, condition_2_data], axis=0)

    condition_1_labels = np.zeros(np.shape(condition_1_data)[0])
    condition_2_labels = np.ones(np.shape(condition_2_data)[0])
    data_labels = np.concatenate([condition_1_labels, condition_2_labels], axis=0)

    return combined_data, data_labels


def perform_k_fold_cross_validation(data, labels, number_of_folds=5):

    score_list = []
    weight_list = []

    # Get Indicies To Split Data Into N Train Test Splits
    #k_fold_object = KFold(n_splits=number_of_folds, random_state=None, shuffle=True)
    k_fold_object = StratifiedKFold(n_splits=number_of_folds, random_state=42, shuffle=True)

    # Iterate Through Each Split
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

    # Return Mean Score and Mean Model Weights
    print(score_list)
    mean_score = np.mean(score_list)

    weight_list = np.array(weight_list)
    mean_weights = np.mean(weight_list, axis=0)
    return mean_score, mean_weights


def perform_decoding(transformed_data, labels):
    trial_length = np.shape(transformed_data)[1]
    score_list = []
    weight_matrix = []

    for timepoint in range(trial_length):
        timepoint_data = transformed_data[:, timepoint]


        mean_score, mean_weights = perform_k_fold_cross_validation(timepoint_data, labels)
        print("TImepoint: ", timepoint, "Mean Score: ", mean_score)
        score_list.append(mean_score)
        weight_matrix.append(mean_weights)

    return score_list, weight_matrix


def visualise_weight_matrix(weight_matrix,  components, base_directory):
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    number_of_timepoints = np.shape(weight_matrix)[0]

    figure_1 = plt.figure()
    [rows, columns] = Widefield_General_Functions.get_best_grid(number_of_timepoints)
    print(rows, columns)
    axes_list = []

    for timepoint in range(number_of_timepoints):
        weights = weight_matrix[timepoint]
        pixel_loadings = np.dot(weights, components)
        pixel_loadings = np.nan_to_num(pixel_loadings)
        pixel_loadings = np.abs(pixel_loadings)
        reconstructed_image = Widefield_General_Functions.create_image_from_data(pixel_loadings, indicies, image_height, image_width)

        axes_list.append(figure_1.add_subplot(rows, columns, timepoint+1))
        axes_list[timepoint].set_title(str(timepoint))
        axes_list[timepoint].axis('off')
        axes_list[timepoint].imshow(reconstructed_image, cmap='jet')

    plt.show()



def reconstruct_weight_matricies(weight_matrix,  components, base_directory):

    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)
    number_of_timepoints = np.shape(weight_matrix)[0]
    reconstructed_matrix_list = []

    for timepoint in range(number_of_timepoints):
        weights = weight_matrix[timepoint]
        pixel_loadings = np.dot(weights, components)
        pixel_loadings = np.abs(pixel_loadings)
        reconstructed_image = Widefield_General_Functions.create_image_from_data(pixel_loadings, indicies, image_height, image_width)
        reconstructed_matrix_list.append(reconstructed_image)

    return reconstructed_matrix_list




def visualise_decoding_over_learning(base_directory, session_list):

    figure_1 = plt.figure()

    # Load Decoding Scores and Weight Matricies
    decoding_scores_list = []
    weight_matrix_list = []
    number_of_sessions = len(session_list)
    for session_index in range(number_of_sessions):
        output_directory = base_directory + session_list[session_index] + "/Decoding_Analysis"

        decoding_scores = np.load(output_directory + "/score_list.npy")
        weight_matrix = np.load(output_directory + "/weight_matrix.npy")
        components = np.load(output_directory + "/Components.npy")
        weight_matrix = reconstruct_weight_matricies(weight_matrix, components, base_directory + session_list[session_index])

        decoding_scores_list.append(decoding_scores)
        weight_matrix_list.append(weight_matrix)


    # Plot Decoding For Each Timestep Across Learning
    number_of_timepoints = np.shape(decoding_scores_list[0])[0]
    rows = 1
    columns = number_of_sessions

    for timepoint in range(number_of_timepoints):
        figure_1 = plt.figure()
        plt.suptitle("Timepoint: " + str(timepoint))
        for session_index in range(number_of_sessions):
            axis = figure_1.add_subplot(rows, columns, session_index + 1)
            image = weight_matrix_list[session_index][timepoint]
            axis.imshow(image, cmap='jet', vmax=np.percentile(image, 99))
            axis.set_title(str(np.around(decoding_scores_list[session_index][timepoint],2)))
            axis.axis('off')
        plt.show()










session_list = [
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging"
]


for session_index in range(len(session_list)):
    print("Session: ", session_index, " of ", len(session_list))

    base_directory = session_list[session_index]

    # Create Output Directory
    output_directory = os.path.join(base_directory, "Decoding_Analysis")
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # Load Data
    combined_data, data_labels = load_data(base_directory)
    print("Loaded Data", np.shape(combined_data))

    # Perform Decoding
    score_list, weight_matrix = perform_decoding(combined_data, data_labels)

    # Save Score List and Weight Matrix
    np.save(output_directory + "/score_list.npy", score_list)
    np.save(output_directory + "/weight_matrix.npy", weight_matrix)

# Visualise Decoding Over Learning:
visualise_decoding_over_learning(base_directory, session_list)


