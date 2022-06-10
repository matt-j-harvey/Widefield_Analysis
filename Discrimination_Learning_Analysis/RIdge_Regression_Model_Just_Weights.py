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




def load_neural_data(base_directory, condition_1_onsets, condition_2_onsets, start_window, stop_window):

    # Load Delta F Matrix
    delta_f_matrix_filepath = os.path.join(base_directory, "Delta_F.hdf5")
    delta_f_matrix_container = h5py.File(delta_f_matrix_filepath, 'r')
    delta_f_matrix = delta_f_matrix_container['Data']

    condition_1_data = get_trial_tensor(delta_f_matrix, condition_1_onsets, start_window, stop_window)
    condition_2_data = get_trial_tensor(delta_f_matrix, condition_2_onsets, start_window, stop_window)

    return condition_1_data, condition_2_data



def load_behavioural_data(base_directory, condition_1_onsets, condition_2_onsets, start_window, stop_window):

    downsampled_running_trace, downsampled_lick_trace = downsample_ai_traces(base_directory)

    condition_1_running_data = get_trial_tensor(downsampled_running_trace, condition_1_onsets, start_window, stop_window)
    condition_2_running_data = get_trial_tensor(downsampled_running_trace, condition_2_onsets, start_window, stop_window)

    condition_1_lick_data = get_trial_tensor(downsampled_lick_trace, condition_1_onsets, start_window, stop_window)
    condition_2_lick_data = get_trial_tensor(downsampled_lick_trace, condition_2_onsets, start_window, stop_window)

    return condition_1_running_data, condition_2_running_data, condition_1_lick_data, condition_2_lick_data



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


def create_design_matrix(vis_1_delta_f, vis_2_delta_f, vis_1_running_data, vis_2_running_data, vis_1_lick_data, vis_2_lick_data):

    # Get Tensor Shapes
    vis_1_trials, trial_length, number_of_pixels = np.shape(vis_1_delta_f)
    vis_2_trials, trial_length, number_of_pixels = np.shape(vis_2_delta_f)

    # Create Stimuli Regressor
    vis_1_stimuli_regressor = create_stimuli_regressor(vis_1_trials, trial_length)
    vis_2_stimuli_regressor = create_stimuli_regressor(vis_2_trials, trial_length)

    condition_2_start = vis_1_trials * trial_length
    stimuli_regressor = np.zeros((trial_length * 2, vis_1_trials*trial_length + vis_2_trials*trial_length))
    stimuli_regressor[0:trial_length, 0:condition_2_start] = vis_1_stimuli_regressor
    stimuli_regressor[trial_length:, condition_2_start:] = vis_2_stimuli_regressor

    # Create Behavioural Regressors
    vis_1_running_regressor = np.concatenate(vis_1_running_data)
    vis_1_lick_regressor = np.concatenate(vis_1_lick_data)

    vis_2_running_regressor = np.concatenate(vis_2_running_data)
    vis_2_lick_regressor = np.concatenate(vis_2_lick_data)

    running_regressor = np.concatenate([vis_1_running_regressor, vis_2_running_regressor])
    lick_regressor = np.concatenate([vis_1_lick_regressor, vis_2_lick_regressor])

    design_matrix = np.vstack([stimuli_regressor, running_regressor, lick_regressor])

    return design_matrix


def create_stimuli_regressor(number_of_trials, trial_length):

    design_matrix = []

    for trial_index in range(number_of_trials):

        trial_stimuli_regressor = np.eye((trial_length))

        design_matrix.append(trial_stimuli_regressor)

    design_matrix = np.hstack(design_matrix)

    return design_matrix



def perform_regression(vis_1_delta_f, vis_2_delta_f, design_matrix):

    # Get Neural Data Shape
    vis_1_trials, trial_length, number_of_pixels = np.shape(vis_1_delta_f)
    vis_2_trials, trial_length, number_of_pixels = np.shape(vis_2_delta_f)

    # Reshape Neural Tensors to 2D
    vis_1_delta_f = np.reshape(vis_1_delta_f, (vis_1_trials * trial_length, number_of_pixels))
    vis_2_delta_f = np.reshape(vis_2_delta_f, (vis_2_trials * trial_length, number_of_pixels))

    # Concatenate Neural Data
    neural_data = np.vstack([vis_1_delta_f, vis_2_delta_f])

    # Tranpose Both
    design_matrix = np.transpose(design_matrix)

    # Perform Regression
    model = Ridge()
    model.fit(X=design_matrix, y=neural_data)

    # Get Coefficients
    regression_coefficients = model.coef_

    # Get R2
    prediction = model.predict(X=design_matrix)
    r2 = r2_score(y_true=neural_data, y_pred=prediction)

    # Get Full Sum of Sqaured Error
    error = np.subtract(neural_data, prediction)
    sqaured_error = np.square(error)
    full_sum_squared_error = np.sum(sqaured_error, axis=0)

    return regression_coefficients, r2, full_sum_squared_error


def get_coefficients_of_partial_determination(vis_1_delta_f, vis_2_delta_f, design_matrix, full_sum_squared_error):

    # Get Neural Data Shape
    vis_1_trials, trial_length, number_of_pixels = np.shape(vis_1_delta_f)
    vis_2_trials, trial_length, number_of_pixels = np.shape(vis_2_delta_f)

    # Reshape Neural Tensors to 2D
    vis_1_delta_f = np.reshape(vis_1_delta_f, (vis_1_trials * trial_length, number_of_pixels))
    vis_2_delta_f = np.reshape(vis_2_delta_f, (vis_2_trials * trial_length, number_of_pixels))

    # Concatenate Neural Data
    neural_data = np.vstack([vis_1_delta_f, vis_2_delta_f])

    # Transpose Both
    design_matrix = np.transpose(design_matrix)


    number_of_regressors = np.shape(design_matrix)[1]
    partial_determination_matrix = []
    for regresor_index in range(number_of_regressors):
        print("Regressor", regresor_index, " of ", number_of_regressors, "at ", datetime.now())

        # Get Partial Design Matrix
        partial_design_marix = np.copy(design_matrix)
        partial_design_marix[:, regresor_index] = 0

        # Fit Reduced Model
        model = Ridge()
        model.fit(X=partial_design_marix, y=neural_data)

        # Get Reduced Sum of Sqaured Error
        prediction = model.predict(partial_design_marix)
        error = np.subtract(neural_data, prediction)
        sqaured_error = np.square(error)
        reduced_sum_squared_error = np.sum(sqaured_error, axis=0)

        # Get Coefficient Of Partial Determination
        coefficient_of_partial_determination = np.subtract(reduced_sum_squared_error, full_sum_squared_error)
        coefficient_of_partial_determination = np.divide(coefficient_of_partial_determination, reduced_sum_squared_error)

        partial_determination_matrix.append(coefficient_of_partial_determination)

    partial_determination_matrix = np.array(partial_determination_matrix)
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
condition_1 = "visual_1_all_onsets.npy"
condition_2 = "visual_2_all_onsets.npy"
model_name = "Discrimination_All_Onsets_Just_Weights"

for session_index in range(len(session_list)):
    print("Session: ", session_index, " of ", len(session_list))

    # Select Session Directory
    base_directory = session_list[session_index]

    # Create Output Directory
    output_directory = os.path.join(base_directory, "Simple_Regression")
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # Load Onsets
    #print("Loading onsets")
    vis_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_1))
    vis_2_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_2))

    # Remove Early Onsets
    vis_1_onsets = remove_early_onsets(vis_1_onsets)
    vis_2_onsets = remove_early_onsets(vis_2_onsets)

    # Load Behavioural Data
    #print("Loading Behavioural Data")
    vis_1_running_data, vis_2_running_data, vis_1_lick_data, vis_2_lick_data = load_behavioural_data(base_directory, vis_1_onsets, vis_2_onsets, start_window, stop_window)

    # Load Neural Data
    print("Loading Neural Data")
    vis_1_delta_f, vis_2_delta_f = load_neural_data(base_directory, vis_1_onsets, vis_2_onsets, start_window, stop_window)

    # Create Design Matrix
    print("Creating Design MAtrix")
    design_matrix = create_design_matrix(vis_1_delta_f, vis_2_delta_f, vis_1_running_data, vis_2_running_data, vis_1_lick_data, vis_2_lick_data)

    # Perform Regression
    print("Performing Regression")
    regression_coefficients, r2, full_sum_squared_error = perform_regression(vis_1_delta_f, vis_2_delta_f, design_matrix)

    # Get Coefficients Of Partial Determination
    print("Getting Coefficients of Partial Determination")
    #partial_determination_matrix = get_coefficients_of_partial_determination(vis_1_delta_f, vis_2_delta_f, design_matrix, full_sum_squared_error)

    # Save Regression Results
    regression_dictionary = {
        "Regression_Coefficients": regression_coefficients,
        "R2": r2,
        "Full_Sum_Sqaure_Error": full_sum_squared_error,
        #"Coefficients_of_Partial_Determination":partial_determination_matrix,
        "Condition_1": condition_1,
        "Condition_2": condition_2,
        "Start_Window": start_window,
        "Stop_Window": stop_window,
    }

    np.save(os.path.join(output_directory, model_name + "_Regression_Model.npy"), regression_dictionary)
