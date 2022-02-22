import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def create_design_matrix(activity_tensor, global_brain_signal):

    # Get Data Structure
    number_of_trials = np.shape(activity_tensor)[0]
    trial_length = np.shape(activity_tensor)[1]
    total_length = number_of_trials * trial_length

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

    # Add Global Signal Regressor
    design_matrix.append(global_brain_signal)

    """
    design_matrix = np.array(design_matrix)
    plt.imshow(design_matrix)
    plt.show()
    """

    design_matrix = np.transpose(design_matrix)
    return design_matrix


def get_beta_weights(activity_tensor):

    # Get Data Structure
    number_of_trials = np.shape(activity_tensor)[0]
    trial_length = np.shape(activity_tensor)[1]
    number_of_brain_regions = np.shape(activity_tensor)[2]

    # Create Regression Model
    model = LinearRegression()

    coefficient_vector_list = []

    # Get Global Brain Signal
    global_mean = np.mean(activity_tensor, axis=0)
    global_noise_signal = np.subtract(activity_tensor, global_mean)
    global_noise_signal = np.reshape(global_noise_signal, (number_of_trials * trial_length, number_of_brain_regions))
    global_noise_signal = np.mean(global_noise_signal, axis=1)

    # Get Activity Tensor For Each
    for roi_index in range(number_of_brain_regions):

        # Get ROI Tensor Trace
        roi_tensor = activity_tensor[:, :, roi_index]

        # Get ROI Design Matrix
        design_matrix = create_design_matrix(roi_tensor, global_noise_signal)

        # Flatten ROI Tensor
        roi_tensor = np.reshape(roi_tensor, (number_of_trials * trial_length))

        # Perform Regression
        model.fit(X=design_matrix, y=roi_tensor)

        # Get Coefficients
        beta_weights = model.coef_
        beta_weights = beta_weights[0:-1]
        print("Beta weights", beta_weights)
        coefficient_vector_list.append(beta_weights)

    coefficient_vector_list = np.array(coefficient_vector_list)


    beta_weight_correlation_matrix = np.corrcoef(coefficient_vector_list)
    print("Beta Weight Correlations", beta_weight_correlation_matrix)

    # Get Correlation Between Beta Weights
    """
    for x in range(number_of_brain_regions-1):
        roi_1_vector = coefficient_vector_list[x]
        roi_2_vector = coefficient_vector_list[ + 1]

        plt.plot(roi_1_vector)
        plt.plot(roi_2_vector)
        plt.show()
    """
    plt.title("Beta Weight Correlations")
    plt.imshow(beta_weight_correlation_matrix, cmap='bwr', vmin=-1, vmax=1)
    plt.show()




    return beta_weight_correlation_matrix



def perform_beta_series_correlation_analysis(context_1_activity_tensor_list, context_2_activity_tensor_list):


    context_1_correlation_map_list = []
    for tensor in context_1_activity_tensor_list:
        beta_correlation_matrix = get_beta_weights(tensor)
        context_1_correlation_map_list.append(beta_correlation_matrix)

    context_2_correlation_map_list = []
    for tensor in context_2_activity_tensor_list:
        beta_correlation_matrix = get_beta_weights(tensor)
        context_2_correlation_map_list.append(beta_correlation_matrix)

    context_1_correlation_map_list = np.array(context_1_correlation_map_list)
    context_2_correlation_map_list = np.array(context_2_correlation_map_list)

    mean_context_1_map = np.mean(context_1_correlation_map_list, axis=0)
    mean_context_2_map = np.mean(context_2_correlation_map_list, axis=0)

    return mean_context_1_map, mean_context_2_map




