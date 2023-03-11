import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


import Ridge_Model_Seperate_Penalties_Class


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def divide_data_into_chunks(input_data, output_data, n_chunks=5):

    # Get Number Of Data points
    number_of_datapoints = np.shape(input_data)[0]

    # Get Size Of Individual Chunk
    chunk_size = int(number_of_datapoints / chunks)

    nested_input_data = []
    nested_output_data = []

    for chunk_index in range(n_chunks):
        start = chunk_index * chunk_size
        stop = start + chunk_size

        chunk_output_data = output_data[start:stop]
        chunk_input_data = input_data[start:stop]

        nested_output_data.append(chunk_output_data)
        nested_input_data.append(chunk_input_data)

    nested_input_data = np.array(nested_input_data)
    nested_output_data = np.array(nested_output_data)

    return nested_input_data, nested_output_data



def get_cross_validated_ridge_penalties(design_matrix, target_data, Nstim, Nt, Nbehv, n_folds=5):

    # Get Selection Of Potential Ridge Penalties
    ridge_penalty_selection = np.logspace(start=-2, stop=5, base=10, num=7)
    number_of_possible_penalties = len(ridge_penalty_selection)
    print("Number oof possible penalties", number_of_possible_penalties)

    # Create Cross Fold Object
    cross_fold_object = KFold(n_splits=n_folds, random_state=None, shuffle=False)

    # Create Matrix To Hold Results
    n_pixels = np.shape(target_data)[1]
    penalty_error_matrix = np.zeros((number_of_possible_penalties, number_of_possible_penalties, n_pixels))

    # Iterate Through Each Ridge Penalty Pair
    for stimulus_penalty_index in tqdm(range(number_of_possible_penalties), desc="Stim Penalty Index", position=0):
        for behaviour_penalty_index in tqdm(range(number_of_possible_penalties), desc="Behaviour Penalty Index", position=1):

            # Select Ridge Penalties
            stimulus_ridge_penalty = ridge_penalty_selection[stimulus_penalty_index]
            behaviour_ridge_penalty = ridge_penalty_selection[behaviour_penalty_index]

            error_list = []

            # Enumerate Through Each Fold
            for i, (train_indices, test_indices) in enumerate(cross_fold_object.split(design_matrix)):

                # Get Training and Test Data
                x_train = design_matrix[train_indices]
                y_train = target_data[train_indices]
                x_test = design_matrix[test_indices]
                y_test = target_data[test_indices]


                """
                plt.title("X Train")
                plt.imshow(x_train)
                forceAspect(plt.gca())
                plt.show()

                plt.title("X Test")
                plt.imshow(x_test)
                forceAspect(plt.gca())
                plt.show()

                plt.title("Y Train")
                plt.imshow(y_train)
                forceAspect(plt.gca())
                plt.show()

                plt.title("Y Test")
                plt.imshow(y_test)
                forceAspect(plt.gca())
                plt.show()
                """


                # Create Model
                model = Ridge_Model_Seperate_Penalties_Class.ridge_model(Nstim, Nt, Nbehv, stimulus_ridge_penalty, behaviour_ridge_penalty)

                # Fit Model
                model.fit(x_train, np.transpose(y_train))

                # Predict Data
                y_pred = model.predict(x_test)
                y_pred = np.transpose(y_pred)

                # Score Prediction
                fold_error = mean_squared_error(y_true=y_test, y_pred=y_pred, multioutput='raw_values')
                error_list.append(fold_error)

            # Get Average Error Across Folds
            error_list = np.array(error_list)
            mean_error = np.mean(error_list, axis=0)

            print("Mean eorr", np.mean(mean_error))

            penalty_error_matrix[stimulus_penalty_index, behaviour_penalty_index] = mean_error


    # Return The Ridge Penalties Associated With The Smallest Error For Each Pixel
    stim_ridge_coef_vector = []
    behaviour_ridge_coef_vector = []
    for pixel_index in range(n_pixels):
        pixel_errors = penalty_error_matrix[:, :, pixel_index]
        min_error = np.min(pixel_errors)
        best_combination_index = np.where(pixel_errors == min_error)
        print("Best combination Index", best_combination_index)

        stim_penalty = best_combination_index[0][0]
        behaviour_penalty = best_combination_index[0][1]

    ridge_coef_vector = np.array(ridge_coef_vector)
    return ridge_coef_vector


