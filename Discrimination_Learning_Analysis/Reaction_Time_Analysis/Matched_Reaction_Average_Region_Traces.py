import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from scipy import stats, signal


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



def get_trial_tensor(delta_f_matrix, onset_list, start_window, stop_window):

    number_of_timepoints = np.shape(delta_f_matrix)[0]

    trial_tensor = []
    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window

        if trial_start >= 0 and trial_stop < number_of_timepoints - 1:
            trial_data = delta_f_matrix[trial_start:trial_stop]

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


def get_session_average_traces(base_directory, activity_matrix, condition_1, condition_2, start_window, stop_window):

    # Get Trial Tensors
    condition_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_1))
    condition_2_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_2))
    condition_1_onsets = remove_early_onsets(condition_1_onsets)
    condition_2_onsets = remove_early_onsets(condition_2_onsets)

    if len(condition_1_onsets) >= 5 and len(condition_2_onsets) >= 5:

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

        return [condition_1_mean, condition_1_sd, condition_2_mean, condition_2_sd]

    else:
        return None

# AI Recorder Processing Functions

def create_stimuli_dictionary():

    channel_index_dictionary = {
        "Photodiode"        :0,
        "Reward"            :1,
        "Lick"              :2,
        "Visual 1"          :3,
        "Visual 2"          :4,
        "Odour 1"           :5,
        "Odour 2"           :6,
        "Irrelevance"       :7,
        "Running"           :8,
        "Trial End"         :9,
        "Camera Trigger"    :10,
        "Camera Frames"     :11,
        "LED 1"             :12,
        "LED 2"             :13,
        "Mousecam"          :14,
        "Optogenetics"      :15,
        }

    return channel_index_dictionary


def get_average_behavioural_traces(base_directory, condition_1, condition_2, start_window, stop_window):

    # Load Downsampled AI
    ai_data = np.load(os.path.join(base_directory, "Downsampled_AI_Matrix_Framewise.npy"))
    stimuli_dictionary = create_stimuli_dictionary()
    running_trace = ai_data[stimuli_dictionary["Running"]]
    lick_trace = ai_data[stimuli_dictionary["Lick"]]

    # Normalise Lick Trace
    lick_trace = np.subtract(lick_trace, np.min(lick_trace))
    lick_trace = np.divide(lick_trace, np.max(lick_trace))
    lick_trace = np.nan_to_num(lick_trace)

    # Get Trial Tensors
    condition_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_1))
    condition_2_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_2))
    condition_1_onsets = remove_early_onsets(condition_1_onsets)
    condition_2_onsets = remove_early_onsets(condition_2_onsets)

    # Get Trial Tensors
    condition_1_running_data = get_trial_tensor(running_trace, condition_1_onsets, start_window, stop_window)
    condition_2_running_data = get_trial_tensor(running_trace, condition_2_onsets, start_window, stop_window)
    condition_1_lick_data = get_trial_tensor(lick_trace, condition_1_onsets, start_window, stop_window)
    condition_2_lick_data = get_trial_tensor(lick_trace, condition_2_onsets, start_window, stop_window)

    # Get Means and SDs
    condition_1_running_mean = np.mean(condition_1_running_data, axis=0)
    condition_2_running_mean = np.mean(condition_2_running_data, axis=0)
    condition_1_lick_mean = np.mean(condition_1_lick_data, axis=0)
    condition_2_lick_mean = np.mean(condition_2_lick_data, axis=0)
    
    return condition_1_running_mean, condition_2_running_mean, condition_1_lick_mean, condition_2_lick_mean



def get_average_sems_and_limits(activity_tensor):
    
    activity_tensor_average = np.mean(activity_tensor, axis=0)
    activity_tensor_sems = stats.sem(activity_tensor, axis=0)
    activity_tensor_upper_limit = np.add(activity_tensor_average, activity_tensor_sems)
    activity_tensor_lower_limit = np.subtract(activity_tensor_average, activity_tensor_sems)
    
    return activity_tensor_average, activity_tensor_sems, activity_tensor_lower_limit, activity_tensor_upper_limit
    


def get_group_averages(control_session_list, mutant_session_list, condition_1, condition_2, start_window, stop_window, selected_region):


    # Get Session Average Traces
    condition_1_control_averages = []
    condition_2_control_averages = []
    condition_1_mutant_averages = []
    condition_2_mutant_averages = []
    
    condition_1_control_running_averages = []   
    condition_2_control_running_averages = []
    condition_1_mutant_running_averages = []
    condition_2_mutant_running_averages = []
    
    condition_1_control_lick_averages = []
    condition_2_control_lick_averages = []
    condition_1_mutant_lick_averages = []
    condition_2_mutant_lick_averages = []


    for base_directory in control_session_list:

        # Load Activity Matrix
        activity_matrix = np.load(os.path.join(base_directory, "Cluster_Activity_Matrix.npy"))

        # Preprocess Activity Matrix
        activity_matrix = preprocess_activity_matrix(activity_matrix)

        # Get Average Traces
        result = get_session_average_traces(base_directory, activity_matrix, condition_1, condition_2, start_window, stop_window)

        if result != None:
            
            condition_1_mean = result[0]
            condition_2_mean = result[2]
            condition_1_control_averages.append(condition_1_mean)
            condition_2_control_averages.append(condition_2_mean)

            condition_1_running_mean, condition_2_running_mean, condition_1_lick_mean, condition_2_lick_mean = get_average_behavioural_traces(base_directory, condition_1, condition_2, start_window, stop_window)
            condition_1_control_running_averages.append(condition_1_running_mean)
            condition_2_control_running_averages.append(condition_2_running_mean)
            condition_1_control_lick_averages.append(condition_1_lick_mean)
            condition_2_control_lick_averages.append(condition_2_lick_mean)


    for base_directory in mutant_session_list:

        # Load Activity Matrix
        activity_matrix = np.load(os.path.join(base_directory, "Cluster_Activity_Matrix.npy"))

        # Preprocess Activity Matrix
        activity_matrix = preprocess_activity_matrix(activity_matrix)

        # Get Average Traces
        result = get_session_average_traces(base_directory, activity_matrix, condition_1, condition_2, start_window, stop_window)

        if result != None:
            condition_1_mean = result[0]
            condition_2_mean = result[2]
            condition_1_mutant_averages.append(condition_1_mean)
            condition_2_mutant_averages.append(condition_2_mean)

            condition_1_running_mean, condition_2_running_mean, condition_1_lick_mean, condition_2_lick_mean = get_average_behavioural_traces(base_directory, condition_1, condition_2, start_window, stop_window)
            condition_1_mutant_running_averages.append(condition_1_running_mean)
            condition_2_mutant_running_averages.append(condition_2_running_mean)
            condition_1_mutant_lick_averages.append(condition_1_lick_mean)
            condition_2_mutant_lick_averages.append(condition_2_lick_mean)


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
    control_condition_1_group_average, control_condition_1_sems, control_condition_1_lower_limit, control_condition_1_upper_limit = get_average_sems_and_limits(condition_1_control_averages)
    control_condition_2_group_average, control_condition_2_sems, control_condition_2_lower_limit, control_condition_2_upper_limit = get_average_sems_and_limits(condition_2_control_averages)
    mutant_condition_1_group_average, mutant_condition_1_sems, mutant_condition_1_lower_limit, mutant_condition_1_upper_limit = get_average_sems_and_limits(condition_1_mutant_averages)
    mutant_condition_2_group_average, mutant_condition_2_sems, mutant_condition_2_lower_limit, mutant_condition_2_upper_limit = get_average_sems_and_limits(condition_2_mutant_averages)
    
    # Get Running Means
    cond_1_control_running_mean, cond_1_control_running_sems, cond_1_control_running_lower_limit, cond_1_control_running_upper_limit = get_average_sems_and_limits(condition_1_control_running_averages)
    cond_2_control_running_mean, cond_2_control_running_sems, cond_2_control_running_lower_limit, cond_2_control_running_upper_limit = get_average_sems_and_limits(condition_2_control_running_averages)
    cond_1_mutant_running_mean, cond_1_mutant_running_sems, cond_1_mutant_running_lower_limit, cond_1_mutant_running_upper_limit = get_average_sems_and_limits(condition_1_mutant_running_averages)
    cond_2_mutant_running_mean, cond_2_mutant_running_sems, cond_2_mutant_running_lower_limit, cond_2_mutant_running_upper_limit = get_average_sems_and_limits(condition_2_mutant_running_averages)

    # Get Lick Means
    cond_1_control_lick_mean, cond_1_control_lick_sems, cond_1_control_lick_lower_limit, cond_1_control_lick_upper_limit = get_average_sems_and_limits(condition_1_control_lick_averages)
    cond_2_control_lick_mean, cond_2_control_lick_sems, cond_2_control_lick_lower_limit, cond_2_control_lick_upper_limit = get_average_sems_and_limits(condition_2_control_lick_averages)
    cond_1_mutant_lick_mean, cond_1_mutant_lick_sems, cond_1_mutant_lick_lower_limit, cond_1_mutant_lick_upper_limit = get_average_sems_and_limits(condition_1_mutant_lick_averages)
    cond_2_mutant_lick_mean, cond_2_mutant_lick_sems, cond_2_mutant_lick_lower_limit, cond_2_mutant_lick_upper_limit = get_average_sems_and_limits(condition_2_mutant_lick_averages)

    # Plot These
    figure_1 = plt.figure()
    gridspec_1 = GridSpec(nrows=4, ncols=2, figure=figure_1)

    condition_1_all_mice_axis = figure_1.add_subplot(gridspec_1[0, 0])
    condition_2_all_mice_axis = figure_1.add_subplot(gridspec_1[0, 1])
    condition_1_group_average_axis = figure_1.add_subplot(gridspec_1[1, 0])
    condition_2_group_average_axis = figure_1.add_subplot(gridspec_1[1, 1])
    condition_1_group_behaviour_axis = figure_1.add_subplot(gridspec_1[2, 0])
    condition_2_group_behaviour_axis = figure_1.add_subplot(gridspec_1[2, 1])
    condition_1_group_lick_axis = figure_1.add_subplot(gridspec_1[3, 0])
    condition_2_group_lick_axis = figure_1.add_subplot(gridspec_1[3, 1])

    # Plot Group Averages
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, 36)
    condition_1_group_average_axis.plot(x_values, control_condition_1_group_average, c='b')
    condition_1_group_average_axis.plot(x_values, mutant_condition_1_group_average, c='g')
    print("lower limit", np.shape(control_condition_1_lower_limit))
    condition_1_group_average_axis.fill_between(x=x_values, y1=control_condition_1_lower_limit, y2=control_condition_1_upper_limit, color='b', alpha=0.2)
    condition_1_group_average_axis.fill_between(x=x_values, y1=mutant_condition_1_lower_limit, y2=mutant_condition_1_upper_limit, color='g', alpha=0.2)

    condition_2_group_average_axis.plot(x_values, control_condition_2_group_average, c='b')
    condition_2_group_average_axis.plot(x_values, mutant_condition_2_group_average, c='g')
    condition_2_group_average_axis.fill_between(x=x_values, y1=control_condition_2_lower_limit, y2=control_condition_2_upper_limit, color='b', alpha=0.2)
    condition_2_group_average_axis.fill_between(x=x_values, y1=mutant_condition_2_lower_limit, y2=mutant_condition_2_upper_limit, color='g', alpha=0.2)

    # Plot Running Averages
    condition_1_group_behaviour_axis.plot(x_values, cond_1_control_running_mean, c='b')
    condition_2_group_behaviour_axis.plot(x_values, cond_2_control_running_mean, c='b')
    condition_1_group_behaviour_axis.plot(x_values, cond_1_mutant_running_mean, c='g')
    condition_2_group_behaviour_axis.plot(x_values, cond_2_mutant_running_mean, c='g')

    condition_1_group_behaviour_axis.fill_between(x=x_values, y1=cond_1_control_running_lower_limit, y2=cond_1_control_running_upper_limit, color='b', alpha=0.2)
    condition_1_group_behaviour_axis.fill_between(x=x_values, y1=cond_1_mutant_running_lower_limit, y2=cond_1_mutant_running_upper_limit, color='g', alpha=0.2)
    condition_2_group_behaviour_axis.fill_between(x=x_values, y1=cond_2_control_running_lower_limit, y2=cond_2_control_running_upper_limit, color='b', alpha=0.2)
    condition_2_group_behaviour_axis.fill_between(x=x_values, y1=cond_2_mutant_running_lower_limit, y2=cond_2_mutant_running_upper_limit, color='g', alpha=0.2)

    # Plot Lick Averages
    condition_1_group_lick_axis.plot(x_values, cond_1_control_lick_mean, c='b')
    condition_2_group_lick_axis.plot(x_values, cond_2_control_lick_mean, c='b')
    condition_1_group_lick_axis.plot(x_values, cond_1_mutant_lick_mean, c='g')
    condition_2_group_lick_axis.plot(x_values, cond_2_mutant_lick_mean, c='g')

    condition_1_group_lick_axis.fill_between(x=x_values, y1=cond_1_control_lick_lower_limit, y2=cond_1_control_lick_upper_limit, color='b', alpha=0.2)
    condition_1_group_lick_axis.fill_between(x=x_values, y1=cond_1_mutant_lick_lower_limit, y2=cond_1_mutant_lick_upper_limit, color='g', alpha=0.2)
    condition_2_group_lick_axis.fill_between(x=x_values, y1=cond_2_control_lick_lower_limit, y2=cond_2_control_lick_upper_limit, color='b', alpha=0.2)
    condition_2_group_lick_axis.fill_between(x=x_values, y1=cond_2_mutant_lick_lower_limit, y2=cond_2_mutant_lick_upper_limit, color='g', alpha=0.2)

    # Plot Individual Mice
    alpha_value = 0.3
    for mouse in condition_1_control_averages:
        condition_1_all_mice_axis.plot(x_values, mouse, c='b', alpha=alpha_value)

    for mouse in condition_1_mutant_averages:
        condition_1_all_mice_axis.plot(x_values, mouse, c='g', alpha=alpha_value)

    for mouse in condition_2_control_averages:
        condition_2_all_mice_axis.plot(x_values, mouse, c='b', alpha=alpha_value)

    for mouse in condition_2_mutant_averages:
        condition_2_all_mice_axis.plot(x_values, mouse, c='g', alpha=alpha_value)

    plt.show()




control_session_list = [

    # Controls 46 sessions

    # 78.1A - 6
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_16_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_17_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_19_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_21_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_24_Discrimination_Imaging",

    # 78.1D - 8
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_14_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_15_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_16_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_17_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_19_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_21_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_23_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_25_Discrimination_Imaging",

    # 4.1B - 7
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging",

    # 22.1A - 7
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_09_25_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_09_29_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_01_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_03_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_05_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_07_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging",

    # 14.1A - 6
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_04_29_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_01_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_03_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_05_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_07_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_09_Discrimination_Imaging",

    # 7.1B - 12
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_01_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_03_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_05_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_07_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_09_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_11_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_13_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_15_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_17_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_19_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_22_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_24_Discrimination_Imaging",

]

mutant_session_list = [
    # Mutants

    # 4.1A - 15
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_08_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_10_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_12_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_14_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_16_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_18_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_23_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_25_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_27_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_03_01_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_03_03_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging",

    # 20.1B - 11
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_09_28_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_09_30_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_09_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_11_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_13_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_15_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_17_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging",

    # 24.1C - 10
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_20_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_22_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_24_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_26_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_28_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_30_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_10_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_10_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_10_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_10_08_Discrimination_Imaging",

    # NXAK16.1B - 16
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_04_30_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_08_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_10_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_12_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_14_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_16_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_18_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_20_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_22_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_24_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_26_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_06_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_06_15_Discrimination_Imaging",

    # 10.1A - 8
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_04_30_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_08_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_10_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_12_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_14_Discrimination_Imaging"


    # 71.2A - 16

]

pixel_assignments = np.load("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Final_Consensus_Clusters.npy")
plt.imshow(pixel_assignments)
plt.show()

# Decoding Parameters
start_window = -42
stop_window = 42
condition_1 = "Stimuli_Onsets_Matched_Reaction_Times_1000_1100.npy"
condition_2 = "Stimuli_Onsets_Matched_Reaction_Times_1100_1200.npy"
#condition_1 = "fast_reaction_lick_onsets.npy"
#condition_2 = "slow_reaction_lick_onsets.npy"

# 0900 - 1000: Controls  285 Mutants 540
# 1000 - 1100: Controls 371 Mutants 522
# 1100 - 1200: Controls 394 Mutants 412
# 1200 - 1300: Controls 310 Mutants 390
# 1300 - 1400: Controls 273 Mutants 257

condition_list = []

get_group_averages(control_session_list, mutant_session_list, condition_1, condition_2, start_window, stop_window, selected_region=54)


