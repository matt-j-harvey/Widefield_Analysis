import numpy as np
import sklearn.svm
from sklearn.decomposition import NMF
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
import os
import matplotlib.pyplot as plt
import sys
from matplotlib import cm

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



def load_data(output_directory, session_index, remake=False):

    if os.path.exists(output_directory + "/Transformed_Data.npy") and not remake:
        transformed_data = np.load(output_directory + "/Transformed_Data.npy")
        components = np.load(output_directory + "/Components.npy")
        data_labels = np.load(output_directory + "/Labels.npy")

    else:
        condition_1_data = np.load(base_directory + session_list[session_index] + "/Stimuli_Evoked_Responses/All Vis 1/All Vis 1_Activity_Matrix_All_Trials.npy")
        condition_2_data = np.load(base_directory + session_list[session_index] + "/Stimuli_Evoked_Responses/All Vis 2/All Vis 2_Activity_Matrix_All_Trials.npy")
        combined_data = np.concatenate([condition_1_data, condition_2_data], axis=0)

        components, transformed_data = perform_dimensionality_reduction(combined_data, n_components=30)
        condition_1_labels = np.zeros(np.shape(condition_1_data)[0])
        condition_2_labels = np.ones(np.shape(condition_2_data)[0])
        data_labels = np.concatenate([condition_1_labels, condition_2_labels], axis=0)


        # Save Transformed Data
        np.save(output_directory + "/Transformed_Data.npy", transformed_data)
        np.save(output_directory + "/Components.npy", components)
        np.save(output_directory + "/Labels.npy", data_labels)



    return transformed_data, components, data_labels


def perform_k_fold_cross_validation(data, labels, model, number_of_folds=5):

    score_list = []
    weight_list = []

    # Get Indicies To Split Data Into N Train Test Splits
    k_fold_object = KFold(n_splits=number_of_folds, random_state=None, shuffle=True)

    # Iterate Through Each Split
    for train_index, test_index in k_fold_object.split(data):

        # Split Data Into Train and Test Sets
        data_train, data_test = data[train_index], data[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

        # Train Model
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

        model = LogisticRegression()
        mean_score, mean_weights = perform_k_fold_cross_validation(timepoint_data, labels, model)

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










#base_directory = "/media/matthew/Seagate Expansion Drive2/Longitudinal_Analysis/NXAK14.1A/"
base_directory = "/media/matthew/Seagate Expansion Drive2/Longitudinal_Analysis/NXAK4.1B/"
session_list = ["2021_02_04_Discrimination_Imaging",
                "2021_02_06_Discrimination_Imaging",
                "2021_02_08_Discrimination_Imaging",
                "2021_02_10_Discrimination_Imaging",
                "2021_02_12_Discrimination_Imaging",
                "2021_02_14_Discrimination_Imaging",
                "2021_02_22_Discrimination_Imaging"]

"""
for session_index in range(len(session_list)):
    print("Session: ", session_index, " of ", len(session_list))

    # Create Output Directory
    output_directory = base_directory + session_list[session_index] + "/Decoding_Analysis"
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # Load Data
    transformed_data, components, data_labels = load_data(output_directory, session_index, remake=False)

    # Perform Decoding
    score_list, weight_matrix = perform_decoding(transformed_data, data_labels)

    # Save Score List and Weight Matrix
    np.save(output_directory + "/score_list.npy", score_list)
    np.save(output_directory + "/weight_matrix.npy", weight_matrix)
"""

# Visualise Decoding Over Learning:
visualise_decoding_over_learning(base_directory, session_list)


