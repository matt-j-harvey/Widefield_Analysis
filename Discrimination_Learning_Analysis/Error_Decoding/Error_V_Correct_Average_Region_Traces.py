import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from scipy import stats
from scipy import signal


def get_trial_tensor(delta_f_matrix, onset_list, start_window, stop_window):

    number_of_timepoints = np.shape(delta_f_matrix)[0]

    trial_tensor = []
    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window

        if trial_start >= 0 and trial_stop < number_of_timepoints - 1:
            trial_data = delta_f_matrix[trial_start:trial_stop]
            print("Trial Data", np.shape(trial_data))

            """
            # Subtract Baseline
            trial_baseline = np.mean(delta_f_matrix[trial_start:0], axis=0)

            print("Trial Basaline", np.shape(trial_baseline))

            trial_data = np.subtract(trial_data, trial_baseline)
            print("Trial Data", np.shape(trial_data))
            """

            trial_tensor.append(trial_data)

    trial_tensor = np.array(trial_tensor)

    return trial_tensor


def remove_early_onsets(onset_list):

    curated_onsets = []
    for onset in onset_list:
        if onset > 3000:
            curated_onsets.append(onset)

    return curated_onsets



def get_lowcut_coefs(w=0.0033, fs=28.):
    b, a = signal.butter(2, w/(fs/2.), btype='highpass');
    return b, a

def perform_lowcut_filter(data, b, a):
    print("Data Shape", np.shape(data))
    filtered_data = signal.filtfilt(b, a, data, padlen=10000, axis=0)
    return filtered_data

def normalise_delta_f(activity_matrix):

    #activity_matrix = np.transpose(activity_matrix)

    # Subtract Min
    min_vector = np.min(activity_matrix, axis=0)
    activity_matrix = np.subtract(activity_matrix, min_vector)

    # Divide By New Max
    max_vector = np.max(activity_matrix, axis=0)
    activity_matrix = np.divide(activity_matrix, max_vector)

    # Remove Nans and Transpose
    activity_matrix = np.nan_to_num(activity_matrix)
    #activity_matrix = np.transpose(activity_matrix)

    return activity_matrix


def preprocess_activity_matrix(activity_matrix, early_cutoff=3000):

    # Lowcut Filter
    b, a = get_lowcut_coefs()

    # Get Sample Data
    usefull_data = activity_matrix[early_cutoff:]

    # Lowcut Filter
    usefull_data = perform_lowcut_filter(usefull_data, b, a)

    # Normalise
    usefull_data = normalise_delta_f(usefull_data)

    # Remove Early Frames
    activity_matrix[0:early_cutoff] = 0
    activity_matrix[early_cutoff:] = usefull_data

    return activity_matrix



def get_session_average_traces(base_directory, condition_1, condition_2, start_window, stop_window):

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(2, 1, 1)
    axis_2 = figure_1.add_subplot(2, 1, 2)


    # Load Clustered Activity Matrix
    activity_matrix = np.load(os.path.join(base_directory, "Cluster_Activity_Matrix.npy"))
    axis_1.imshow(np.transpose(activity_matrix[5000:6000]))

    # Preprocess Activity Matrix
    activity_matrix = preprocess_activity_matrix(activity_matrix)
    axis_2.imshow(np.transpose(activity_matrix[5000:6000]))

    plt.show()

    # Get Trial Tensors
    condition_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_1))
    condition_2_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_2))
    condition_1_onsets = remove_early_onsets(condition_1_onsets)
    condition_2_onsets = remove_early_onsets(condition_2_onsets)

    print("Correct trials", len(condition_1_onsets))
    print("Inorrect Trials", len(condition_2_onsets))

    # Get Trial Tensors
    condition_1_data = get_trial_tensor(activity_matrix, condition_1_onsets, start_window, stop_window)
    condition_2_data = get_trial_tensor(activity_matrix, condition_2_onsets, start_window, stop_window)

    # Get Means and SDs
    condition_1_mean = np.mean(condition_1_data, axis=0)
    condition_1_sd = np.std(condition_1_data, axis=0)

    condition_2_mean = np.mean(condition_2_data, axis=0)
    condition_2_sd = np.std(condition_2_data, axis=0)
    
    return condition_1_mean, condition_1_sd, condition_2_mean, condition_2_sd



def plot_region_average_traces(base_directory, condition_1, condition_2, start_window, stop_window):

    # Load Clustered Activity Matrix
    activity_matrix = np.load(os.path.join(base_directory, "Cluster_Activity_Matrix.npy"))

    # Preprocess Activity Matrix

    # Get Average Traces
    condition_1_mean, condition_1_sd, condition_2_mean, condition_2_sd = get_session_average_traces(base_directory, activity_matrix, condition_1, condition_2, start_window, stop_window)
  
    # Create Save Directory
    error_prediction_directory = os.path.join(base_directory, "Error_Prediction")
    if not os.path.exists(error_prediction_directory):
        os.mkdir(error_prediction_directory)

    plot_directory = os.path.join(error_prediction_directory, "Region_Average_Traces")
    if not os.path.exists(plot_directory):
        os.mkdir(plot_directory)

    # Plot Region Traces
    pixel_assignments = np.load("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Final_Consensus_Clusters.npy")
    cluster_outlines = np.load("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/cluster_outlines.npy")
    outline_pixels = np.nonzero(cluster_outlines)

    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, 36)
    
    number_of_regions = np.shape(activity_matrix)[1]
    for region_index in range(number_of_regions):

        figure_1 = plt.figure(figsize=(10, 5))
        gridspec_1 = GridSpec(nrows=1, ncols=4)
        trace_axis = figure_1.add_subplot(gridspec_1[0, 0:3])
        region_axis = figure_1.add_subplot(gridspec_1[0, 3])
        
        condition_1_region_trace = condition_1_mean[:, region_index]
        condition_1_region_sd = condition_1_sd[:, region_index]
        condition_1_region_upper_bound = np.add(condition_1_region_trace, condition_1_region_sd)
        condition_1_region_lower_bound = np.subtract(condition_1_region_trace, condition_1_region_sd)
        trace_axis.plot(x_values, condition_1_region_trace, c='b')
        trace_axis.fill_between(x=x_values, y1=condition_1_region_lower_bound, y2=condition_1_region_upper_bound, color='b', alpha=0.4)

        condition_2_region_trace = condition_2_mean[:, region_index]
        condition_2_region_sd = condition_2_sd[:, region_index]
        condition_2_region_upper_bound = np.add(condition_2_region_trace, condition_2_region_sd)
        condition_2_region_lower_bound = np.subtract(condition_2_region_trace, condition_2_region_sd)
        trace_axis.plot(x_values, condition_2_region_trace, c='r')
        trace_axis.fill_between(x=x_values, y1=condition_2_region_lower_bound, y2=condition_2_region_upper_bound, color='r', alpha=0.4)

        region_mask = np.where(pixel_assignments == region_index, 1, 0)
        region_mask[outline_pixels] = 1
        region_axis.imshow(region_mask)
        region_axis.axis('off')
        region_axis.set_title(str(region_index))

        trace_axis.axvline(0, c='k', linestyle='--')

        plt.show()
        



def get_group_averages(control_session_list, mutant_session_list, condition_1, condition_2, start_window, stop_window, selected_region):
    condition_1_mutant_averages = []
    condition_1_control_averages = []

    condition_2_mutant_averages = []
    condition_2_control_averages = []

    for base_directory in control_session_list:

        # Get Average Traces
        condition_1_mean, condition_1_sd, condition_2_mean, condition_2_sd = get_session_average_traces(base_directory, condition_1, condition_2, start_window, stop_window)
        condition_1_control_averages.append(condition_1_mean)
        condition_2_control_averages.append(condition_2_mean)

    for base_directory in mutant_session_list:
        # Get Average Traces
        condition_1_mean, condition_1_sd, condition_2_mean, condition_2_sd = get_session_average_traces(base_directory, condition_1, condition_2, start_window, stop_window)
        condition_1_mutant_averages.append(condition_1_mean)
        condition_2_mutant_averages.append(condition_2_mean)


    # Get Selected Region Trace
    condition_1_mutant_averages = np.array(condition_1_mutant_averages)
    condition_1_control_averages = np.array(condition_1_control_averages)
    condition_2_mutant_averages = np.array(condition_2_mutant_averages)
    condition_2_control_averages = np.array(condition_2_control_averages)
    condition_1_control_averages = condition_1_control_averages[:,:, selected_region]
    condition_2_control_averages = condition_2_control_averages[:,:, selected_region]
    condition_1_mutant_averages = condition_1_mutant_averages[:,:, selected_region]
    condition_2_mutant_averages = condition_2_mutant_averages[:,:, selected_region]

    # Get Group Means
    print("condition_1_control_averages", np.shape(condition_1_control_averages))
    control_condition_1_group_average = np.mean(condition_1_control_averages, axis=0)
    control_condition_2_group_average = np.mean(condition_2_control_averages, axis=0)
    mutant_condition_1_group_average = np.mean(condition_1_mutant_averages, axis=0)
    mutant_condition_2_group_average = np.mean(condition_2_mutant_averages, axis=0)

    control_condition_1_sems = stats.sem(condition_1_control_averages, axis=0)
    control_condition_2_sems = stats.sem(condition_2_control_averages, axis=0)
    mutant_condition_1_sems = stats.sem(condition_1_mutant_averages, axis=0)
    mutant_condition_2_sems = stats.sem(condition_2_mutant_averages, axis=0)
    
    control_condition_1_upper_limit = np.add(control_condition_1_group_average, control_condition_1_sems)
    control_condition_2_upper_limit = np.add(control_condition_2_group_average, control_condition_2_sems)
    mutant_condition_1_upper_limit = np.add(mutant_condition_1_group_average, mutant_condition_1_sems)
    mutant_condition_2_upper_limit = np.add(mutant_condition_2_group_average, mutant_condition_2_sems)
    
    control_condition_1_lower_limit = np.subtract(control_condition_1_group_average, control_condition_1_sems)
    control_condition_2_lower_limit = np.subtract(control_condition_2_group_average, control_condition_2_sems)
    mutant_condition_1_lower_limit = np.subtract(mutant_condition_1_group_average, mutant_condition_1_sems)
    mutant_condition_2_lower_limit = np.subtract(mutant_condition_2_group_average, mutant_condition_2_sems)


    # Plot These
    figure_1 = plt.figure()
    gridspec_1 = GridSpec(nrows=2, ncols=2, figure=figure_1)

    condition_1_all_mice_axis = figure_1.add_subplot(gridspec_1[0, 0])
    condition_2_all_mice_axis = figure_1.add_subplot(gridspec_1[0, 1])
    condition_1_group_average_axis = figure_1.add_subplot(gridspec_1[1, 0])
    condition_2_group_average_axis = figure_1.add_subplot(gridspec_1[1, 1])

    # Plot Group Averages
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, 36)
    condition_1_group_average_axis.plot(x_values, control_condition_1_group_average, c='b')
    condition_1_group_average_axis.plot(x_values, mutant_condition_1_group_average, c='g')
    condition_1_group_average_axis.fill_between(x=x_values, y1=control_condition_1_lower_limit, y2=control_condition_1_upper_limit, color='b', alpha=0.2)
    condition_1_group_average_axis.fill_between(x=x_values, y1=mutant_condition_1_lower_limit, y2=mutant_condition_1_upper_limit, color='g', alpha=0.2)

    condition_2_group_average_axis.plot(x_values, control_condition_2_group_average, c='b')
    condition_2_group_average_axis.plot(x_values, mutant_condition_2_group_average, c='g')
    condition_2_group_average_axis.fill_between(x=x_values, y1=control_condition_2_lower_limit, y2=control_condition_2_upper_limit, color='b', alpha=0.2)
    condition_2_group_average_axis.fill_between(x=x_values, y1=mutant_condition_2_lower_limit, y2=mutant_condition_2_upper_limit, color='g', alpha=0.2)


    # Plot Individual Mice
    for mouse in condition_1_control_averages:
        condition_1_all_mice_axis.plot(x_values, mouse, c='b')

    for mouse in condition_1_mutant_averages:
        condition_1_all_mice_axis.plot(x_values, mouse, c='g')

    for mouse in condition_2_control_averages:
        condition_2_all_mice_axis.plot(x_values, mouse, c='b')

    for mouse in condition_2_mutant_averages:
        condition_2_all_mice_axis.plot(x_values, mouse, c='g')


    plt.show()



control_session_list = [

    # Controls 46 sessions

    # 78.1A - 6
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_21_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_24_Discrimination_Imaging",

    # 78.1D - 8
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_23_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_25_Discrimination_Imaging",

    # 4.1B - 7
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging",

    # 22.1A - 7
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_07_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging",

    # 14.1A - 6
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_07_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_09_Discrimination_Imaging",

    # 7.1B - 12
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_22_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_24_Discrimination_Imaging",
]

mutant_session_list = [

    # Mutants

    # 4.1A - 15
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_03_03_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging",

    # 20.1B - 11
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_17_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging",

    # 24.1C - 10
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_10_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_10_08_Discrimination_Imaging",

    # NXAK16.1B - 16
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_06_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_06_15_Discrimination_Imaging",

    # 10.1A - 8
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_12_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_14_Discrimination_Imaging",


    # 71.2A - 16

]

pixel_assignments = np.load("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Final_Consensus_Clusters.npy")
plt.imshow(pixel_assignments)
plt.show()

# Decoding Parameters
start_window = -48
stop_window = 48
condition_1 = "visual_2_correct_onsets.npy"
condition_2 = "visual_2_incorrect_onsets.npy"

get_group_averages(control_session_list, mutant_session_list, condition_1, condition_2, start_window, stop_window, selected_region=24)


