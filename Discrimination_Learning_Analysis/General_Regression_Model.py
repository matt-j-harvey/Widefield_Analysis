import numpy as np
import sklearn.svm
from sklearn.decomposition import NMF
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
import os
import matplotlib.pyplot as plt
import sys
from matplotlib import cm
import h5py
from datetime import datetime

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")
import Widefield_General_Functions


def load_generous_mask(home_directory):

    # Loads the mask for a video, returns a list of which pixels are included, as well as the original image height and width
    mask = np.load(home_directory + "/Generous_Mask.npy")

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask>0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width


def ResampleLinear1D(original, targetLen):

    original = np.array(original, dtype=float)
    index_arr = np.linspace(0, len(original)-1, num=targetLen, dtype=float)
    index_floor = np.array(index_arr, dtype=int) #Round down
    index_ceil = index_floor + 1
    index_rem = index_arr - index_floor #Remain

    val1 = original[index_floor]
    val2 = original[index_ceil % len(original)]
    interp = val1 * (1.0-index_rem) + val2 * index_rem
    assert(len(interp) == targetLen)
    return interp



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

    onset_count = 0
    number_of_onsets = len(onset_list)
    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window
        trial_data = delta_f_matrix[trial_start:trial_stop]
        trial_tensor.append(trial_data)
        onset_count += 1
    trial_tensor = np.array(trial_tensor)
    trial_tensor = np.nan_to_num(trial_tensor)
    return trial_tensor




def load_neural_data(base_directory, condition_onsets_list, start_window, stop_window):

    # Load Delta F Matrix
    delta_f_matrix_filepath = os.path.join(base_directory, "Delta_F.hdf5")
    delta_f_matrix_container = h5py.File(delta_f_matrix_filepath, 'r')
    delta_f_matrix = delta_f_matrix_container['Data']

    neural_tensor = []
    for condition in condition_onsets_list:
        conditon_data = get_trial_tensor(delta_f_matrix, condition, start_window, stop_window)
        neural_tensor.append(conditon_data)

    neural_tensor = np.vstack(neural_tensor)
    return neural_tensor



def load_behavioural_data(downsampled_running_trace, downsampled_lick_trace, condition_onsets_list, start_window, stop_window):

    running_tensor = []
    lick_tensor = []

    for condition in condition_onsets_list:
        condition_running_data = get_trial_tensor(downsampled_running_trace, condition, start_window, stop_window)
        condition_lick_data = get_trial_tensor(downsampled_lick_trace, condition, start_window, stop_window)
        running_tensor.append(condition_running_data)
        lick_tensor.append(condition_lick_data)

    running_tensor = np.vstack(running_tensor)
    lick_tensor = np.vstack(lick_tensor)

    return running_tensor, lick_tensor



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



def downsample_ai_traces(base_directory, sanity_check=True):

    # Load Frame Times
    frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    frame_times = Widefield_General_Functions.invert_dictionary(frame_times)

    # Load AI Recorder File
    ai_filename = Widefield_General_Functions.get_ai_filename(base_directory)
    ai_data = Widefield_General_Functions.load_ai_recorder_file(base_directory + "/" + ai_filename)

    # Extract Relevant Traces
    stimuli_dictionary = Widefield_General_Functions.create_stimuli_dictionary()
    running_trace = ai_data[stimuli_dictionary["Running"]]
    lick_trace = ai_data[stimuli_dictionary["Lick"]]

    # Load Delta F Matrix
    delta_f_matrix_filepath = os.path.join(base_directory, "Delta_F.hdf5")
    delta_f_matrix_container = h5py.File(delta_f_matrix_filepath, 'r')
    delta_f_matrix = delta_f_matrix_container['Data']

    # Get Data Structure
    number_of_timepoints = np.shape(delta_f_matrix)[0]
    imaging_start = frame_times[0]
    imaging_stop = frame_times[number_of_timepoints - 1]

    # Get Traces Only While Imaging
    imaging_running_trace = running_trace[imaging_start:imaging_stop]
    imaging_lick_trace = lick_trace[imaging_start:imaging_stop]

    # Downsample Traces
    downsampled_running_trace = ResampleLinear1D(imaging_running_trace, number_of_timepoints)
    downsampled_lick_trace = ResampleLinear1D(imaging_lick_trace, number_of_timepoints)

    """
    # Check Movement Controls Directory Exists
    movement_controls_directory = os.path.join(base_directory, "Movement_Controls")
    if not os.path.exists(movement_controls_directory):
        os.mkdir(movement_controls_directory)

    # Save Downsampled Running Trace
    np.save(os.path.join(movement_controls_directory, "Downsampled_Running_Trace.npy"), downsampled_running_trace)

    # Sanity Check
    if sanity_check == True:
        figure_1 = plt.figure()
        real_axis = figure_1.add_subplot(2, 1, 1)
        down_axis = figure_1.add_subplot(2, 1, 2)

        real_stop = int(len(imaging_running_trace)/100)
        down_stop = int(len(downsampled_running_trace)/100)

        real_axis.plot(imaging_running_trace[0:real_stop])
        down_axis.plot(downsampled_running_trace[0:down_stop])

        real_axis.set_title("Real Running Trace")
        down_axis.set_title("Downsampled Running Trace")
        plt.show()
    """

    return downsampled_running_trace, downsampled_lick_trace


def create_design_matrix(number_of_conditions, trial_length, condition_trials, running_tensor, lick_tensor, neural_tensor):

    # Create Condition Regressors
    condition_regressor_list = []
    for condition in condition_trials:
        condition_regressor = create_stimuli_regressor(condition, trial_length)
        print("Condition Regressor", np.shape(condition_regressor))
        condition_regressor_list.append(condition_regressor)

    # Combine Stimuli Regressors Into Design Matrix
    number_of_trials = np.sum(condition_trials)
    number_of_timepoints = trial_length * number_of_trials
    number_of_regressors = number_of_conditions * trial_length
    stimuli_design_matrix = np.zeros((number_of_regressors, number_of_timepoints))
    print("Stimuli Design MAtrix", np.shape(stimuli_design_matrix))


    timepoint_start = 0
    for condition_index in range(number_of_conditions):

        regressor_start = condition_index * trial_length
        regressor_stop = regressor_start + trial_length

        condition_duration = trial_length * condition_trials[condition_index]
        timepoint_stop = timepoint_start + condition_duration

        condition_regressor = condition_regressor_list[condition_index]

        stimuli_design_matrix[regressor_start:regressor_stop, timepoint_start:timepoint_stop] = condition_regressor
        timepoint_start += condition_duration

    stimuli_design_matrix = np.transpose(stimuli_design_matrix)


    # Create Behavioural Design Matrix
    running_tensor = np.reshape(running_tensor, (np.shape(running_tensor)[0] * np.shape(running_tensor)[1], 1))
    print("running tensor", np.shape(running_tensor))

    lick_tensor = np.reshape(lick_tensor, (np.shape(lick_tensor)[0] * np.shape(lick_tensor)[1], 1))
    print("lick tensor", np.shape(lick_tensor))

    behavioural_design_matrix = np.hstack([running_tensor, lick_tensor])
    print("Behavioural_Deisgn Matrix", np.shape(behavioural_design_matrix))

    # Create Full Design Matrix
    design_matrix = np.hstack([stimuli_design_matrix, behavioural_design_matrix])
    print("Design Matrix Shape", np.shape(design_matrix))

    return design_matrix


def create_stimuli_regressor(number_of_trials, trial_length):

    design_matrix = []

    for trial_index in range(number_of_trials):

        trial_stimuli_regressor = np.eye((trial_length))

        design_matrix.append(trial_stimuli_regressor)

    design_matrix = np.hstack(design_matrix)

    return design_matrix



def perform_regression(neural_tensor, design_matrix):

    print("neural tensor shape", np.shape(neural_tensor))

    # Get Neural Data Shape
    number_of_trials, trial_length, number_of_pixels = np.shape(neural_tensor)

    # Reshape Neural Tensor to 2D
    neural_tensor = np.reshape(neural_tensor, (number_of_trials * trial_length, number_of_pixels))

    print("Neural Tensor Shape", np.shape(neural_tensor))
    print("Design Matrix Shape", np.shape(design_matrix))

    # Perform Regression
    model = Ridge()
    model.fit(X=design_matrix, y=neural_tensor)

    # Get Coefficients
    regression_coefficients = model.coef_

    # Get R2
    prediction = model.predict(X=design_matrix)
    r2 = r2_score(y_true=neural_tensor, y_pred=prediction)

    # Get Full Sum of Sqaured Error
    error = np.subtract(neural_tensor, prediction)
    sqaured_error = np.square(error)
    full_sum_squared_error = np.sum(sqaured_error, axis=0)

    return regression_coefficients, r2, full_sum_squared_error


def get_coefficients_of_partial_determination(neural_tensor, design_matrix, full_sum_squared_error):

    # Get Neural Data Shape
    number_of_trials, trial_length, number_of_pixels = np.shape(neural_tensor)

    # Reshape Neural Tensor to 2D
    neural_tensor = np.reshape(neural_tensor, (number_of_trials * trial_length, number_of_pixels))

    number_of_regressors = np.shape(design_matrix)[1]
    partial_determination_matrix = []
    for regresor_index in range(number_of_regressors):
        print("Regressor", regresor_index, " of ", number_of_regressors, "at ", datetime.now())

        # Get Partial Design Matrix
        partial_design_marix = np.copy(design_matrix)
        partial_design_marix[:, regresor_index] = 0

        # Fit Reduced Model
        model = Ridge()
        model.fit(X=partial_design_marix, y=neural_tensor)

        # Get Reduced Sum of Sqaured Error
        prediction = model.predict(partial_design_marix)
        error = np.subtract(neural_tensor, prediction)
        sqaured_error = np.square(error)
        reduced_sum_squared_error = np.sum(sqaured_error, axis=0)

        # Get Coefficient Of Partial Determination
        coefficient_of_partial_determination = np.subtract(reduced_sum_squared_error, full_sum_squared_error)
        coefficient_of_partial_determination = np.divide(coefficient_of_partial_determination, reduced_sum_squared_error)

        partial_determination_matrix.append(coefficient_of_partial_determination)

    partial_determination_matrix = np.array(partial_determination_matrix)
    partial_determination_matrix = np.transpose(partial_determination_matrix)
    return partial_determination_matrix


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


def visualise_coefficients(base_directory, coefficients):

    indicies, image_height, image_width = load_generous_mask(base_directory)

    number_of_dimensions = np.ndim(coefficients)

    if number_of_dimensions == 1:
        image = Widefield_General_Functions.create_image_from_data(coefficients, indicies, image_height, image_width)
        plt.imshow(image)
        plt.show()

    elif number_of_dimensions == 2:

        dim_1, dim_2 = np.shape(coefficients)

        if dim_1 > dim_2:
            coefficients = np.transpose(coefficients)

        nuber_of_samples = np.shape(coefficients)[0]
        plt.ion()

        for x in range(nuber_of_samples):
            image = Widefield_General_Functions.create_image_from_data(coefficients[x], indicies, image_height, image_width)
            plt.title(str(x))
            plt.imshow(image)
            plt.draw()
            plt.pause(0.1)
            plt.clf()

        plt.ioff()
    plt.close()




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

def remove_early_onsets(onsets, window=1500):

    thresholded_onsets = []

    for onset in onsets:
        if onset > window:
            thresholded_onsets.append(onset)

    return thresholded_onsets


def seperate_regression_coefficents_into_conditions(coefficients, trial_length, number_of_conditions):

    print("Coefficient Shape", np.shape(coefficients))
    condition_coefficients_list = []

    for condition_index in range(number_of_conditions):
        condition_start = condition_index * trial_length
        condition_stop = condition_start + trial_length
        condition_coefficients = coefficients[:, condition_start:condition_stop]
        condition_coefficients_list.append(condition_coefficients)

    return condition_coefficients_list



def perform_regression_analysis(session_list, start_window, stop_window, condition_onset_files, model_name):

    number_of_conditions = len(condition_onset_files)
    trial_length = stop_window - start_window
    print("Number of conditions: ", number_of_conditions)
    print("Trial Length: ", trial_length)

    for session_index in range(len(session_list)):
        print("Session: ", session_index, " of ", len(session_list))

        # Select Session Directory
        base_directory = session_list[session_index]

        # Create Output Directory
        output_directory = os.path.join(base_directory, "Simple_Regression")
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        # Load Onsets
        condition_onsets_list = []
        for onset_file in condition_onset_files:
            condition_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", onset_file))
            condition_onsets = remove_early_onsets(condition_onsets)

            condition_onsets = condition_onsets[0:3]

            condition_onsets_list.append(condition_onsets)

        # Get Trial Numbers
        condition_trial_numbers = []
        for condition in condition_onsets_list:
            condition_trials = len(condition)
            condition_trial_numbers.append(condition_trials)
        print("Condition Trials: ", condition_trial_numbers)

        # Load Behavioural Data
        print("Loading Behavioural Data")
        downsampled_running_trace, downsampled_lick_trace = downsample_ai_traces(base_directory)
        running_tensor, lick_tensor = load_behavioural_data(downsampled_running_trace, downsampled_lick_trace, condition_onsets_list, start_window, stop_window)
        print("Running Tensor Shape", np.shape(running_tensor))
        print("Lick Tensor Shape", np.shape(lick_tensor))

        # Load Neural Data
        print("Loading Neural Data")
        neural_tensor = load_neural_data(base_directory, condition_onsets_list, start_window, stop_window)
        print("Neural Tensor Shape", np.shape(neural_tensor))


        # Create Design Matrix
        print("Creating Design MAtrix")
        design_matrix = create_design_matrix(number_of_conditions, trial_length, condition_trial_numbers, running_tensor, lick_tensor, neural_tensor)

        # Perform Regression
        print("Performing Regression")
        regression_coefficients, r2, full_sum_squared_error = perform_regression(neural_tensor, design_matrix)
        print("Regression Coefficients", np.shape(regression_coefficients))

        # Get Coefficients Of Partial Determination
        print("Getting Coefficients of Partial Determination")
        partial_determination_matrix = get_coefficients_of_partial_determination(neural_tensor, design_matrix, full_sum_squared_error)
        print("Partial Determination matrix", np.shape(partial_determination_matrix))

        # Seperate Cofficeints Into Each Condition
        condition_coefficients_list = seperate_regression_coefficents_into_conditions(regression_coefficients, trial_length, number_of_conditions)
        condition_cpd_list = seperate_regression_coefficents_into_conditions(partial_determination_matrix, trial_length, number_of_conditions)

        # Save Regression Results
        regression_dictionary = {
            "Conditions": condition_onset_files,
            "R2": r2,
            "Full_Sum_Sqaure_Error": full_sum_squared_error,
            "Regression_Coefficients": condition_coefficients_list,
            "Coefficients_of_Partial_Determination":condition_cpd_list,
            "Start_Window": start_window,
            "Stop_Window": stop_window,
        }

        np.save(os.path.join(output_directory, model_name + "_Regression_Model.npy"), regression_dictionary)


session_list = [
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_02_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_04_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_06_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_08_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_10_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_12_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_14_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_16_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_18_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_23_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_25_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_27_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_03_01_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_03_03_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_03_05_Discrimination_Imaging",
]



start_window = -14
stop_window = 27
condition_onset_files = ["visual_1_all_onsets.npy", "visual_2_all_onsets.npy"]
model_name = "Test_General_Model"
perform_regression_analysis(session_list, start_window, stop_window, condition_onset_files, model_name)