import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm
import os



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


def get_cross_validated_ridge_penalties(design_matrix, target_data, n_folds=5):

    # Get Selection Of Potential Ridge Penalties
    ridge_penalty_selection = np.logspace(start=-2, stop=5, base=10, num=36)

    # Create Cross Fold Object
    cross_fold_object = KFold(n_splits=n_folds, random_state=None, shuffle=False)

    penalty_error_matrix = []

    # Iterate Through Each Ridge Penalty
    for penalty in ridge_penalty_selection:
        error_list = []

        # Enumerate Through Each Fold
        for i, (train_indices, test_indices) in enumerate(cross_fold_object.split(design_matrix)):

            # Get Training and Test Data
            x_train = design_matrix[train_indices]
            y_train = target_data[train_indices]
            x_test = design_matrix[test_indices]
            y_test = target_data[test_indices]

            # Create Model
            model = Ridge(alpha=penalty, solver='auto')

            # Fit Model
            model.fit(X=x_train, y=y_train)

            # Predict Data
            y_pred = model.predict(X=x_test)

            # Score Prediction
            fold_error = mean_squared_error(y_true=y_test, y_pred=y_pred, multioutput='raw_values')
            error_list.append(fold_error)

        # Get Average Error Across Folds
        error_list = np.array(error_list)
        mean_error = np.mean(error_list, axis=0)
        penalty_error_matrix.append(mean_error)


    # Return The Ridge Penalties Associated With The Smallest Error For Each Pixel
    penalty_error_matrix = np.array(penalty_error_matrix)

    penalty_error_matrix = np.transpose(penalty_error_matrix)
    ridge_coef_vector = []
    for pixel_errors in penalty_error_matrix:
        min_error = np.min(pixel_errors)
        min_index = list(pixel_errors).index(min_error)
        selected_ridge_penalty = ridge_penalty_selection[min_index]
        ridge_coef_vector.append(selected_ridge_penalty)

    ridge_coef_vector = np.array(ridge_coef_vector)
    return ridge_coef_vector




"""
#base_directory = r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_12_Transition_Imaging"

# Load Design Matrix
#design_matrix = np.load(os.path.join(base_directory, "Ride_Regression", "Design_Matrix.npy"))
#get_cross_validated_ridge_penalties(design_matrix, design_matrix)
"""