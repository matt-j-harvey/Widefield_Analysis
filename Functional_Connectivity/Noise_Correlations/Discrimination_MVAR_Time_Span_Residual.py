import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from scipy import stats
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist


def sort_matrix(matrix):

    # Cluster Matrix
    Z = ward(pdist(matrix))

    # Get Dendogram Leaf Order
    new_order = leaves_list(Z)

    # Sorted Matrix
    sorted_matrix = matrix[:, new_order][new_order]

    return sorted_matrix



def normalise_activity_matrix(activity_matrix):

    # Subtract Min
    min_vector = np.min(activity_matrix, axis=0)
    activity_matrix = np.subtract(activity_matrix, min_vector)

    # Divide By Max
    max_vector = np.max(activity_matrix, axis=0)
    activity_matrix = np.divide(activity_matrix, max_vector)

    return activity_matrix



def load_activity_tensors(delta_f_matrix, onset_list, start_window, stop_window, preceeding_window=5):

    # Create Empty Lists To Hold Data
    activity_tensor = []
    preceeding_activity_tensor = []

    # Iterate Through Each Trial Onset
    for onset in onset_list:

        # Get Trial Start and Stop Times
        trial_start = onset + start_window
        trial_stop = onset + stop_window

        # Extract Trial Data and Shifted Trial Data
        trial_data = delta_f_matrix[trial_start:trial_stop]
        lagged_preceeding_actiity_list = []
        for x in range(1, preceeding_window+1):
            preceeding_trial_data = delta_f_matrix[trial_start-x:trial_stop-x]
            lagged_preceeding_actiity_list.append(preceeding_trial_data)

        # Add These To Respective Lists
        activity_tensor.append(trial_data)
        preceeding_activity_tensor.append(lagged_preceeding_actiity_list)

    # Convert Lists To Arrays
    activity_tensor = np.array(activity_tensor)
    preceeding_activity_tensor = np.array(preceeding_activity_tensor)
    print("Activity tensor shape", np.shape(activity_tensor))

    # Flatten These
    number_of_trials = len(onset_list)
    trial_length = stop_window - start_window
    number_of_regions = np.shape(activity_tensor)[2]

    activity_tensor = np.reshape(activity_tensor, (number_of_trials * trial_length, number_of_regions))
    preceeding_activity_tensor = np.reshape(preceeding_activity_tensor, (number_of_trials * trial_length, preceeding_window * number_of_regions))

    return activity_tensor, preceeding_activity_tensor


def create_stimuli_regressors(stimuli_trials, trial_length, stimuli_regressor_matrix, start_index, stimuli_index):

    trial_start = start_index
    stimuli_start = stimuli_index * trial_length

    for trial_index in range(stimuli_trials):
        trial_stop = trial_start + trial_length
        stimuli_stop = stimuli_start + trial_length

        stimuli_regressor_matrix[trial_start:trial_stop, stimuli_start:stimuli_stop] = np.identity(trial_length)

        trial_start += trial_length

    return stimuli_regressor_matrix


def smooth_data(data, window_size=3):

    smoothed_array = []
    number_of_traces = np.shape(data)[1]

    for trace_index in range(number_of_traces):
        trace = data[:, trace_index]
        smoothed_data = np.convolve(trace, np.ones(window_size) / window_size, mode='same')

        """
        plt.plot(trace)
        plt.plot(smoothed_data)
        plt.show()
        """
        smoothed_array.append(smoothed_data)

    smoothed_array = np.array(smoothed_array)
    smoothed_array = np.transpose(smoothed_array)
    return smoothed_array


def fit_mvar_model(base_directory):

    # Load Residual Data
    activity_matrix = np.load(os.path.join(base_directory, "Movement_Correction", "Unexplained_Activity.npy"))
    print("Delta F Matrix Shape", np.shape(activity_matrix))

    # Normalise Activity Matrix
    activity_matrix = normalise_activity_matrix(activity_matrix)

    # Remove Background Activity
    activity_matrix = activity_matrix[:, 1:]
    print("Raw Data Shape", np.shape(activity_matrix))

    # Smooth Activity Matrix
    activity_matrix = smooth_data(activity_matrix, window_size=5)
    print("Raw Data Shape", np.shape(activity_matrix))

    # Remove Nans
    activity_matrix = np.nan_to_num(activity_matrix)

    window_size = 1
    following_data = activity_matrix[window_size:]
    preceeding_data = activity_matrix[0:-window_size]

    model = Ridge(alpha=2)
    model.fit(X=preceeding_data,y=following_data)
    coefs = model.coef_
    print("Coefs shape", np.shape(coefs))

    coefs = np.reshape(coefs, (58, 58))
    #np.fill_diagonal(coefs, 0)
    plt.imshow(coefs)
    plt.show()
    """

    # Get Preceeding Activity


    # Combine Tensors
    activity_tensor_list = np.vstack(activity_tensor_list)
    preceeding_activity_tensor_list = np.vstack(preceeding_activity_tensor_list)

    # Get Total Number Of Trials
    total_number_of_trials = 0
    number_of_stimuli = len(onset_group_list)
    trial_length = stop_window - start_window
    for stimuli_index in range(number_of_stimuli):
        stimuli_trials = len(onset_group_list[stimuli_index])
        total_number_of_trials += stimuli_trials

    # Create Stimuli Regressors
    stimuli_regressor_matrix = np.zeros((total_number_of_trials * trial_length, (number_of_stimuli * trial_length)))

    start_index = 0
    for stimuli_index in range(number_of_stimuli):
        stimuli_trials = len(onset_group_list[stimuli_index])
        stimuli_regressor_matrix = create_stimuli_regressors(stimuli_trials, trial_length, stimuli_regressor_matrix, start_index, stimuli_index)
        start_index += stimuli_trials * trial_length


    # Transpose These Tensors
    preceeding_activity_tensor_list = np.transpose(preceeding_activity_tensor_list)
    stimuli_regressor_matrix = np.transpose(stimuli_regressor_matrix)

    print("Preceeding activity tensor list", np.shape(preceeding_activity_tensor_list))
    print("Stimuli Regressor Matrix", np.shape(stimuli_regressor_matrix))
    number_of_stimuli_regressors = np.shape(stimuli_regressor_matrix)[0]

    # Create Design Matrix - Will be X
    design_matrix = np.vstack([preceeding_activity_tensor_list, stimuli_regressor_matrix])
    design_matrix = np.transpose(design_matrix)

    # Transpose Activity Tensor - Will be Y
    activity_tensor_list = np.transpose(activity_tensor_list)
    print("Activity tensor list", np.shape(activity_tensor_list))

    # Iterate Through Each Region
    number_of_regions = np.shape(activity_tensor_list)[0]

    # Create Model
    model = Ridge(fit_intercept=True)

    # Iterate Through Each Region
    coef_list = []
    for region_index in range(number_of_regions):

        region_trace = activity_tensor_list[region_index]

        model.fit(X=design_matrix, y=region_trace)
        coefs = model.coef_
        coef_list.append(coefs)

    coef_list = np.array(coef_list)
    print("Coef List Shape", np.shape(coef_list))

    # Get Recurrent Coefs
    recurrent_regressors = coef_list[:, 0:-number_of_stimuli_regressors]

    # Reshape These
    recurrent_regressors = np.reshape(recurrent_regressors, (number_of_regions, preceeding_window, number_of_regions))
    connectivity_matrix = np.mean(recurrent_regressors, axis=1)


    #connectivity_matrix = coef_list[:, 0:number_of_regions]
    """
    #return connectivity_matrix


"""
test_size = 20000
    train_end = number_of_timepoints - test_size

    following_data = activity_matrix[1:train_end + 1]
    preceeding_data = activity_matrix[0:train_end]

    test_following_data = activity_matrix[train_end:]
    test_preceeding_data = activity_matrix[train_end-1:-1]

    print("Following Data Shape", np.shape(following_data))
    print("Preceeding Data shape", np.shape(preceeding_data))
    print("Test Following Data Shape", np.shape(test_following_data))
    print("Test Preceeding Data", np.shape(test_preceeding_data))


    
    alpha_list = []
    score_list = []

    for x in range(0, 10):
        model = Ridge(alpha=x)
        model.fit(X=preceeding_data, y=following_data)

        score = model.score(X=test_preceeding_data, y=test_following_data)
        alpha_list.append(x)
        score_list.append(score)
    plt.plot(score_list)
    plt.show()
    print("Alphas")
    print(alpha_list)

    print("Scores")
    print(score_list)

"""



session_list = [
     "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
     #"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging"]


for session in session_list:
    fit_mvar_model(session)
