import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def create_design_matrix(activity_tensor, bodycam_tensor):

    # Get Data Structure
    number_of_trials = np.shape(activity_tensor)[0]
    trial_length = np.shape(activity_tensor)[1]
    number_of_bodycam_components = np.shape(bodycam_tensor)[2]
    total_length = number_of_trials * trial_length


    # Flatten Bodycam Tensor
    boddycam_tensor = np.reshape(bodycam_tensor, (number_of_trials * trial_length, number_of_bodycam_components))

    # Get Mean Trace
    trace_mean = np.mean(activity_tensor, axis=0)

    #plt.plot(trace_mean)
    #plt.show()

    design_matrix = []
    for trial_index in range(number_of_trials):
        trial_start = trial_index * trial_length
        trial_stop = trial_start + trial_length
        trial_regressor = np.zeros(total_length)
        trial_regressor[trial_start:trial_stop] = trace_mean
        design_matrix.append(trial_regressor)

    # Add Baseline Regressor
    baseline_regressor = np.ones(total_length)
    design_matrix.append(baseline_regressor)

    # Add Bodycam Regressors
    for component_index in range(number_of_bodycam_components):
        design_matrix.append(boddycam_tensor[:, component_index])

    #design_matrix.append(global_brain_signal)

    """
    design_matrix = np.array(design_matrix)
    plt.imshow(design_matrix)
    plt.show()
    """

    design_matrix = np.transpose(design_matrix)
    return design_matrix


def get_beta_weights(activity_tensor, bodycam_tensor):

    # Get Data Structure
    number_of_trials = np.shape(activity_tensor)[0]
    trial_length = np.shape(activity_tensor)[1]
    number_of_brain_regions = np.shape(activity_tensor)[2]
    number_of_bodycam_components = np.shape(bodycam_tensor)[2]

    # Create Regression Model
    model = LinearRegression()
    #model = Ridge()

    coefficient_vector_list = []

    # Get Global Brain Signal
    """
    global_mean = np.mean(activity_tensor, axis=0)
    global_noise_signal = np.subtract(activity_tensor, global_mean)
    global_noise_signal = np.reshape(global_noise_signal, (number_of_trials * trial_length, number_of_brain_regions))
    global_noise_signal = np.mean(global_noise_signal, axis=1)
    """
    # Get Activity Tensor For Each
    for roi_index in range(number_of_brain_regions):

        # Get ROI Tensor Trace
        roi_tensor = activity_tensor[:, :, roi_index]

        # Get ROI Design Matrix
        design_matrix = create_design_matrix(roi_tensor, bodycam_tensor)

        # Flatten ROI Tensor
        roi_tensor = np.reshape(roi_tensor, (number_of_trials * trial_length))

        # Perform Regression
        model.fit(X=design_matrix, y=roi_tensor)

        # Get Coefficients
        beta_weights = model.coef_
        beta_weights = beta_weights[0: -(number_of_bodycam_components + 1)]

        coefficient_vector_list.append(beta_weights)

    coefficient_vector_list = np.array(coefficient_vector_list)
    return coefficient_vector_list

    """
    beta_weight_correlation_matrix = np.corrcoef(coefficient_vector_list)
    print("Beta Weight Correlations", beta_weight_correlation_matrix)

    # Get Correlation Between Beta Weights
   
    for x in range(number_of_brain_regions-1):
        roi_1_vector = coefficient_vector_list[x]
        roi_2_vector = coefficient_vector_list[ + 1]

        plt.plot(roi_1_vector)
        plt.plot(roi_2_vector)
        plt.show()
    
    plt.title("Beta Weight Correlations")
    plt.imshow(beta_weight_correlation_matrix, cmap='bwr', vmin=-1, vmax=1)
    plt.show()
    """



    return beta_weight_correlation_matrix

def normalise_list(input_list):

    # Subtract Min
    list_min = np.min(input_list)
    input_list = np.subtract(input_list, list_min)

    # Divide Max
    list_max = np.max(input_list)
    input_list = np.divide(input_list, list_max)

    return input_list


def perform_beta_series_correlation_analysis(context_1_activity_tensor_list, context_2_activity_tensor_list, context_1_bodycam_tensor_list, context_2_bodycam_tensor_list):


    # Get Context 1 Beta Series
    number_of_context_1_stimuli = len(context_1_activity_tensor_list)
    context_1_beta_series = []

    for stimuli_index in range(number_of_context_1_stimuli):
        activity_tensor = context_1_activity_tensor_list[stimuli_index]
        bodycam_tensor = context_1_bodycam_tensor_list[stimuli_index]
        beta_series_tensor = get_beta_weights(activity_tensor, bodycam_tensor)
        context_1_beta_series.append(beta_series_tensor)

    context_1_beta_series_tensor = np.hstack(context_1_beta_series)



    # Get Context 2 Beta Series
    number_of_context_2_stimuli = len(context_2_activity_tensor_list)
    context_2_beta_series = []
    for stimuli_index in range(number_of_context_2_stimuli):
        activity_tensor = context_2_activity_tensor_list[stimuli_index]
        bodycam_tensor = context_2_bodycam_tensor_list[stimuli_index]
        beta_series_tensor = get_beta_weights(activity_tensor, bodycam_tensor)
        context_2_beta_series.append(beta_series_tensor)

    context_2_beta_series_tensor = np.array(context_2_beta_series[0])

    print("Context 1 beta series tensor", np.shape(context_1_beta_series_tensor))
    print("Context 2 beta series tensor", np.shape(context_2_beta_series_tensor))

    context_1_correlation_map = np.corrcoef(context_1_beta_series_tensor)
    context_2_correlation_map = np.corrcoef(context_2_beta_series_tensor)


    difference_map = np.subtract(context_1_correlation_map, context_2_correlation_map)

    plt.imshow(context_1_correlation_map, cmap='bwr', vmin=-1, vmax=1)
    plt.show()

    plt.imshow(context_2_correlation_map, cmap='bwr', vmin=-1, vmax=1)
    plt.show()

    plt.imshow(difference_map, cmap='bwr', vmin=-1, vmax=1)
    plt.show()


    return context_1_correlation_map, context_2_correlation_map

