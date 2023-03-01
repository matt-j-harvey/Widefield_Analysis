import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import os

from Widefield_Utils import widefield_utils


def view_brain_map(map_values, plot_name, indicies, image_height, image_width, colourmap, save_directory, vmin=None, vmax=None):

    coef_map = widefield_utils.create_image_from_data(map_values, indicies, image_height, image_width)

    if vmax==None:
        vmax = np.max(np.abs(coef_map))

    if vmin==None:
        vmin = -1*np.max(np.abs(coef_map))

    plt.title(plot_name)
    plt.imshow(coef_map, cmap=colourmap, vmax=vmax, vmin=vmin)
    plt.axis('off')
    plt.colorbar()

    plt.savefig(os.path.join(save_directory, plot_name + ".svg"))
    plt.close()


def view_multiple_maps(map_values_list, plot_names, indicies, image_height, image_width, colourmap, save_directory, plot_name, vmin=None, vmax=None):

    # Get Grid
    number_of_maps = len(map_values_list)
    n_columns, n_rows = widefield_utils.get_best_grid(number_of_maps)

    # Create Figure
    figure_1 = plt.figure(figsize=(50, 50))

    for map_index in range(number_of_maps):

        # Reconstruct Image
        map_values = map_values_list[map_index]
        coef_map = widefield_utils.create_image_from_data(map_values, indicies, image_height, image_width)

        # Create Axis
        axis = figure_1.add_subplot(n_rows, n_columns, map_index + 1)

        vmax = np.max(np.abs(coef_map))
        vmin = -1 * np.max(np.abs(coef_map))

        image_handle = axis.imshow(coef_map, cmap=colourmap, vmax=vmax, vmin=vmin)
        axis.axis('off')
        axis.set_title(plot_names[map_index])
        #figure_1.colorbar(image_handle, cax=axis)

    plt.savefig(os.path.join(save_directory, plot_name + ".svg"))
    plt.close()


def view_ridge_penalty_maps(regression_directory, regression_dictionary, indicies, image_height, image_width ):

    # Extract Ridge Penalties
    ridge_penalties = regression_dictionary["Ridge_Penalties"]
    ridge_penalties = np.log(ridge_penalties)

    # Create Map
    view_brain_map(ridge_penalties, "Best_Ridge_Penalties_Log_10", indicies, image_height, image_width, widefield_utils.get_musall_cmap(), regression_directory, vmin=-7.5, vmax=7.5)


def calculate_r2(design_matrix, delta_f_matrix, regression_dictionary, regression_directory, indicies, image_height, image_width, early_cutoff=3000):

    """
    # Load Delta F
    delta_f_matrix = np.load(os.path.join(base_directory, "Delta_F_Matrix_100_by_100_SVD.npy"))
    print("Delta F Matrix Shape", np.shape(delta_f_matrix))
    print("Design Mateix Shape", np.shape(design_matrix))

    # Remove Early Cutoff
    delta_f_matrix = delta_f_matrix[early_cutoff:]
    design_matrix = design_matrix[early_cutoff:]
    """

    # Extract Regression Coefs and Intercepts
    regression_coefs = regression_dictionary["Coefs"]
    regression_intercepts = regression_dictionary["Intercepts"]

    # Preidcit Activity
    predicted_activity = get_model_prediction(design_matrix, regression_coefs, regression_intercepts)

    # Calculate R2
    r2_vector = r2_score(y_true=delta_f_matrix, y_pred=predicted_activity, multioutput='raw_values')
    np.save(os.path.join(regression_directory, "R2_Vector.npy"), r2_vector)

    # Create Map
    view_brain_map(r2_vector, "R2_Score", indicies, image_height, image_width, "inferno", regression_directory, vmin=0, vmax=1)



def get_model_prediction(design_matrix, regression_coefs, regression_intercepts):
    predicted_activity = np.matmul(design_matrix, np.transpose(regression_coefs))
    predicted_activity = np.add(predicted_activity, regression_intercepts)
    return predicted_activity

def calculate_coefficient_of_partial_determination(sum_sqaure_error_full, sum_sqaure_error_reduced):
    coefficient_of_partial_determination = np.subtract(sum_sqaure_error_reduced, sum_sqaure_error_full)
    sum_sqaure_error_reduced = np.add(sum_sqaure_error_reduced, 0.000001) # Ensure We Do Not Divide By Zero
    coefficient_of_partial_determination = np.divide(coefficient_of_partial_determination, sum_sqaure_error_reduced)
    coefficient_of_partial_determination = np.nan_to_num(coefficient_of_partial_determination)
    return coefficient_of_partial_determination



def view_cpds(delta_f_matrix, design_matrix, design_matrix_key_dict, regression_dictionary, regression_directory, indicies, image_height, image_width):

    # Load Delta F
    """
    delta_f_matrix = np.load(os.path.join(base_directory, "Delta_F_Matrix_100_by_100_SVD.npy"))
    print("Delta F Matrix Shape", np.shape(delta_f_matrix))
    print("Design Mateix Shape", np.shape(design_matrix))

    # Remove Early Cutoff
    delta_f_matrix = delta_f_matrix[early_cutoff:]
    design_matrix = design_matrix[early_cutoff:]
    """

    # Extract Regression Coefs and Intercepts
    regression_coefs = regression_dictionary["Coefs"]
    regression_intercepts = regression_dictionary["Intercepts"]
    ridge_penalties = regression_dictionary["Ridge_Penalties"]

    # Preidict Activity
    predicted_activity = get_model_prediction(design_matrix, regression_coefs, regression_intercepts)

    # Get Full SSE
    sum_square_errors = np.subtract(delta_f_matrix, predicted_activity)
    sum_square_errors = np.square(sum_square_errors)
    sum_square_errors = np.sum(sum_square_errors, axis=0)

    # Load Design Matrix Key
    #design_matrix_key_dict = np.load(os.path.join(regression_directory, "design_matrix_key_dict.npy"), allow_pickle=True)[()]
    number_of_regressor_groups = design_matrix_key_dict["number_of_regressor_groups"]
    regressor_group_starts = design_matrix_key_dict["coef_group_starts"]
    regressor_group_stops = design_matrix_key_dict["coef_group_stops"]
    regressor_names = design_matrix_key_dict["coefs_names"]

    for regressor_group_index in range(number_of_regressor_groups):

        partial_design_matrix = np.copy(design_matrix)
        regressor_group_start = regressor_group_starts[regressor_group_index]
        regressor_group_stop = regressor_group_stops[regressor_group_index]
        np.random.shuffle(partial_design_matrix[:, regressor_group_start:regressor_group_stop])

        regressor_group_name = regressor_names[regressor_group_start]
        regressor_group_name = regressor_group_name[0:-4]


        # Get Sum Of Squared Error Of This Prediction
        partial_model = Ridge(alpha=ridge_penalties)
        partial_model.fit(X=partial_design_matrix, y=delta_f_matrix)
        partial_prediction = partial_model.predict(partial_design_matrix)
        partial_sse = np.subtract(delta_f_matrix, partial_prediction)
        partial_sse = np.square(partial_sse)
        partial_sse = np.sum(partial_sse, axis=0)

        # Calculate Coefficient Of Partial Determination
        coefficient_of_partial_determination = calculate_coefficient_of_partial_determination(sum_square_errors, partial_sse)
        np.save(os.path.join(regression_directory, regressor_group_name + "_CPD.npy"), coefficient_of_partial_determination)

        # Create Map
        view_brain_map(coefficient_of_partial_determination, regressor_group_name + "_CPD", indicies, image_height, image_width, "inferno", regression_directory, vmin=0, vmax=0.2)


def visualise_regression_results(regression_directory, regression_dictionary, design_matrix, design_matrix_key_dict, delta_f_matrix):

    # Load Regression Dict
    #regression_dictionary = np.load(os.path.join(regression_directory,  "Regression_Dictionary_Simple.npy"), allow_pickle=True)[()]

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

    # View Ridge Penalty Map
    view_ridge_penalty_maps(regression_directory, regression_dictionary, indicies, image_height, image_width)

    # View Regression Coefs
    regression_coefs = regression_dictionary["Coefs"]
    regression_coefs = np.transpose(regression_coefs)


    # Load Design Matrix Key
    #design_matrix_key_dict = np.load(os.path.join(regression_directory, "design_matrix_key_dict.npy"), allow_pickle=True)[()]
    number_of_regressor_groups = design_matrix_key_dict["number_of_regressor_groups"]
    regressor_group_sizes = design_matrix_key_dict["coef_group_sizes"]
    regressor_group_starts = design_matrix_key_dict["coef_group_starts"]
    regressor_group_stops = design_matrix_key_dict["coef_group_stops"]
    regressor_names = design_matrix_key_dict["coefs_names"]

    # View Coefs
    difference_cmap = widefield_utils.get_musall_cmap()
    for regression_group_index in range(number_of_regressor_groups):
        group_start = regressor_group_starts[regression_group_index]
        group_stop = regressor_group_stops[regression_group_index]
        group_coefs = regression_coefs[group_start:group_stop]
        group_names = regressor_names[group_start:group_stop]
        plot_name = "regressor Grpup" + str(regression_group_index).zfill(3)
        view_multiple_maps(group_coefs, group_names, indicies, image_height, image_width, difference_cmap, regression_directory, plot_name)

    # Load Design Matrix
    #design_matrix = np.load(os.path.join(regression_directory, "Design_Matrix.npy"))

    # View Total Explained Variance
    calculate_r2(design_matrix, delta_f_matrix, regression_dictionary, regression_directory, indicies, image_height, image_width, early_cutoff=3000)

    # View CPDs
    view_cpds(delta_f_matrix, design_matrix, design_matrix_key_dict, regression_dictionary, regression_directory, indicies, image_height, image_width)

