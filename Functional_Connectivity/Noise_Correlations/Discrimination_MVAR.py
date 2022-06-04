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



def load_activity_tensors(delta_f_matrix, onset_list, start_window, stop_window):

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
        preceeding_trial_data = delta_f_matrix[trial_start-1:trial_stop-1]

        # Add These To Respective Lists
        activity_tensor.append(trial_data)
        preceeding_activity_tensor.append(preceeding_trial_data)

    # Convert Lists To Arrays
    activity_tensor = np.array(activity_tensor)
    preceeding_activity_tensor = np.array(preceeding_activity_tensor)
    print("Activity tensor shape", np.shape(activity_tensor))
    # Flatten These
    number_of_trials = len(onset_list)
    trial_length = stop_window - start_window
    number_of_regions = np.shape(activity_tensor)[2]

    activity_tensor = np.reshape(activity_tensor, (number_of_trials * trial_length, number_of_regions))
    preceeding_activity_tensor = np.reshape(preceeding_activity_tensor, (number_of_trials * trial_length, number_of_regions))

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



def fit_mvar_model(delta_f_matrix, onset_group_list, start_window, stop_window):


    # Load Activity Tensors
    activity_tensor_list = []
    preceeding_activity_tensor_list = []
    for onset_group in onset_group_list:
        activity_tensor, preceeding_activity_tensor = load_activity_tensors(delta_f_matrix, onset_group, start_window, stop_window)
        activity_tensor_list.append(activity_tensor)
        preceeding_activity_tensor_list.append(preceeding_activity_tensor)

    # Combine Tensors
    activity_tensor_list = np.vstack(activity_tensor_list)
    preceeding_activity_tensor_list = np.vstack(preceeding_activity_tensor_list)
    delta_activity_tensor = np.subtract(activity_tensor_list, preceeding_activity_tensor_list)

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


    """
    # Get Downsampled Running Speed
    downsampled_running_trace = np.load(os.path.join(base_directory, "Movement_Controls", "Downsampled_Running_Trace.npy"))
    running_regressor_list = []
    for onset_group in onset_group_list:
        running_tensor = extract_running_tensor(downsampled_running_trace, onset_group, start_window, stop_window)
        running_regressor_list.append(running_tensor)
    running_regressor_list = np.vstack(running_regressor_list)
    """

    # Transpose These Tensors
    delta_activity_tensor = np.transpose(delta_activity_tensor)
    preceeding_activity_tensor_list = np.transpose(preceeding_activity_tensor_list)
    stimuli_regressor_matrix = np.transpose(stimuli_regressor_matrix)
    #running_regressor_list = np.transpose(running_regressor_list)

    # Create Design Matrix - Will be X
    design_matrix = np.vstack([preceeding_activity_tensor_list, stimuli_regressor_matrix])
    design_matrix = np.transpose(design_matrix)

    # Transpose Activity Tensor - Will be Y
    activity_tensor_list = np.transpose(activity_tensor_list)
    print("Activity tensor list", np.shape(activity_tensor_list))

    # Iterate Through Each Region
    number_of_regions = np.shape(delta_activity_tensor)[0]

    # Create Model
    model = LinearRegression()

    # Iterate Through Each Region
    coef_list = []
    for region_index in range(number_of_regions):

        region_trace = activity_tensor_list[region_index]

        model.fit(X=design_matrix, y=region_trace)
        coefs = model.coef_
        coef_list.append(coefs)

    coef_list = np.array(coef_list)


    connectivity_matrix = coef_list[:, 0:number_of_regions]

    return connectivity_matrix





# Settings
condition_1 = "visual_1_all_onsets.npy"
condition_2 = "visual_2_all_onsets.npy"
start_window = -14
stop_window = 40
trial_length = stop_window - start_window

# Load Neural Data
base_directory = r"C:\Users\matth\Documents\Functional-Connectivity_V2\Parcellated_Delta_F\NXAK7.1B\2021_02_22_Discrimination_Imaging"
activity_matrix = np.load(os.path.join(base_directory, "Cluster_Activity_Matrix.npy"))
print("Delta F Matrix Shape", np.shape(activity_matrix))

# Normalise Activity Matrix
activity_matrix = normalise_activity_matrix(activity_matrix)

# Remove Background Activity
activity_matrix = activity_matrix[:, 1:]

# Load Onsets
vis_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_1))
vis_2_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_2))


connectivity_matrix = fit_mvar_model(activity_matrix, [vis_1_onsets, vis_2_onsets], start_window, stop_window)
plt.imshow(connectivity_matrix)
plt.show()

connectivity_matrix = sort_matrix(connectivity_matrix)
plt.imshow(connectivity_matrix)
plt.show()