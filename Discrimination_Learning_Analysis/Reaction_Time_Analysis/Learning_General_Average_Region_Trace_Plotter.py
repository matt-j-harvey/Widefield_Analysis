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
            trial_tensor.append(trial_data)

    trial_tensor = np.array(trial_tensor)

    return trial_tensor

def remove_early_onsets(onset_list):

    curated_onsets = []

    for onset in onset_list:
        if onset > 3000:
            curated_onsets.append(onset)

    return curated_onsets


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


def normalise_lick_trace(lick_trace, base_directory):

    lick_threshold = np.load(os.path.join(base_directory, "Lick_Threshold.npy"))
    lick_trace = np.divide(lick_trace, lick_threshold)

    # Normalise Lick Trace
    """
    lick_basline = stats.mode(lick_trace)[0]
    lick_trace = np.clip(lick_trace, a_min=lick_basline, a_max=None)
    lick_trace = np.subtract(lick_trace, np.min(lick_trace))
    lick_trace = np.divide(lick_trace, np.max(lick_trace))
    lick_trace = np.nan_to_num(lick_trace)

    """
    return lick_trace



def get_average_sems_and_limits(activity_tensor):
    activity_tensor_average = np.mean(activity_tensor, axis=0)
    activity_tensor_sems = stats.sem(activity_tensor, axis=0)
    activity_tensor_upper_limit = np.add(activity_tensor_average, activity_tensor_sems)
    activity_tensor_lower_limit = np.subtract(activity_tensor_average, activity_tensor_sems)

    return activity_tensor_average, activity_tensor_sems, activity_tensor_lower_limit, activity_tensor_upper_limit


def get_condition_average_traces(session_list, condition, min_trials):
    
    group_neural_traces = []
    group_running_traces = []
    group_lick_traces = []
    
    for base_directory in session_list:

        # Load Onsets
        condition_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition))
        condition_onsets = remove_early_onsets(condition_onsets)
        
        if len(condition_onsets) >= min_trials:
    
            # Load Activity Matrix
            activity_matrix = np.load(os.path.join(base_directory, "Cluster_Activity_Matrix.npy"))
    
            # Preprocess Activity Matrix
            activity_matrix = preprocess_activity_matrix(activity_matrix)
            
            # Get Activity Trial Tensors
            activity_tensor = get_trial_tensor(activity_matrix, condition_onsets, start_window, stop_window)
            neural_mean_trace = np.mean(activity_tensor, axis=0)
 
            # Load Downsampled AI
            ai_data = np.load(os.path.join(base_directory, "Downsampled_AI_Matrix_Framewise.npy"))
            stimuli_dictionary = create_stimuli_dictionary()
            running_trace = ai_data[stimuli_dictionary["Running"]]
            lick_trace = ai_data[stimuli_dictionary["Lick"]]
            lick_trace = normalise_lick_trace(lick_trace, base_directory)
            
            # Get Behaviour Trial Tensors
            lick_tensor = get_trial_tensor(lick_trace, condition_onsets, start_window, stop_window)
            running_tensor = get_trial_tensor(running_trace, condition_onsets, start_window, stop_window)
            lick_mean = np.mean(lick_tensor, axis=0)
            running_mean = np.mean(running_tensor, axis=0)
            
            # Add To Lists
            group_neural_traces.append(neural_mean_trace)
            group_running_traces.append(running_mean)
            group_lick_traces.append(lick_mean)

    group_neural_traces = np.array(group_neural_traces)
    group_running_traces = np.array(group_running_traces)
    group_lick_traces = np.array(group_lick_traces)

    return group_neural_traces, group_running_traces, group_lick_traces



def get_group_averages(control_session_meta_list, mutant_session_meta_list, condition, start_window, stop_window, selected_region, min_trials=5):

    for group in control_session_meta_list:
        print("Control session list", group)

    # Get Number Of Conditions
    number_of_session_types = len(control_session_meta_list)

    # Create Figure
    figure_1 = plt.figure()
    gridspec_1 = GridSpec(nrows=4, ncols=number_of_session_types, figure=figure_1)

    # Get X Values
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, 36)

    for session_type_index in range(number_of_session_types):

        control_neural_traces, control_running_traces, control_lick_traces = get_condition_average_traces(control_session_meta_list[session_type_index], condition, min_trials)
        mutant_neural_traces, mutant_running_traces, mutant_lick_traces = get_condition_average_traces(mutant_session_meta_list[session_type_index], condition, min_trials)

        # Get Selected Region Trace
        control_neural_traces = control_neural_traces[:,:, selected_region]
        mutant_neural_traces = mutant_neural_traces[:,:, selected_region]

        # Get Neural Means
        control_neural_mean, control_neural_sems, control_neural_lower_limit, control_neural_upper_limit = get_average_sems_and_limits(control_neural_traces)
        mutant_neural_mean, mutant_neural_sems, mutant_neural_lower_limit, mutant_neural_upper_limit = get_average_sems_and_limits(mutant_neural_traces)

        # Get Running Means
        control_running_mean, control_running_sems, control_running_lower_limit, control_running_upper_limit = get_average_sems_and_limits(control_running_traces)
        mutant_running_mean, mutant_running_sems, mutant_running_lower_limit, mutant_running_upper_limit = get_average_sems_and_limits(mutant_running_traces)
     
        # Get Lick Means
        control_lick_mean, control_lick_sems, control_lick_lower_limit, control_lick_upper_limit = get_average_sems_and_limits(control_lick_traces)
        mutant_lick_mean, mutant_lick_sems, mutant_lick_lower_limit, mutant_lick_upper_limit = get_average_sems_and_limits(mutant_lick_traces)

        # Create Axes
        all_sessions_neural_axis = figure_1.add_subplot(gridspec_1[0, session_type_index])
        neural_axis = figure_1.add_subplot(gridspec_1[1, session_type_index])
        running_axis = figure_1.add_subplot(gridspec_1[2, session_type_index])
        lick_axis = figure_1.add_subplot(gridspec_1[3, session_type_index])

        # Plot Group Averages
        neural_axis.plot(x_values, control_neural_mean, c='b')
        neural_axis.plot(x_values, mutant_neural_mean, c='g')
        neural_axis.fill_between(x=x_values, y1=control_neural_lower_limit, y2=control_neural_upper_limit, color='b', alpha=0.2)
        neural_axis.fill_between(x=x_values, y1=mutant_neural_lower_limit, y2=mutant_neural_upper_limit, color='g', alpha=0.2)

        # Plot Running Averages
        running_axis.plot(x_values, control_running_mean, c='b')
        running_axis.plot(x_values, mutant_running_mean, c='g')
        running_axis.fill_between(x=x_values, y1=control_running_lower_limit, y2=control_running_upper_limit, color='b', alpha=0.2)
        running_axis.fill_between(x=x_values, y1=mutant_running_lower_limit, y2=mutant_running_upper_limit, color='g', alpha=0.2)

        # Plot Lick Averages
        lick_axis.plot(x_values, control_lick_mean, c='b')
        lick_axis.plot(x_values, mutant_lick_mean, c='g')
        lick_axis.fill_between(x=x_values, y1=control_lick_lower_limit, y2=control_lick_upper_limit, color='b', alpha=0.2)
        lick_axis.fill_between(x=x_values, y1=mutant_lick_lower_limit, y2=mutant_lick_upper_limit, color='g', alpha=0.2)

        # Plot Individual Mice
        alpha_value = 0.3
        for mouse in control_neural_traces:
            all_sessions_neural_axis.plot(x_values, mouse, c='b', alpha=alpha_value)

        for mouse in mutant_neural_traces:
            all_sessions_neural_axis.plot(x_values, mouse, c='g', alpha=alpha_value)


    plt.show()




def split_sessions_By_d_prime(session_list, intermediate_threshold, post_threshold):

    pre_learning_sessions = []
    intermediate_learning_sessions = []
    post_learning_sessions = []

    # Iterate Throug Sessions
    for session in session_list:

        # Load D Prime
        behavioural_dictionary = np.load(os.path.join(session, "Behavioural_Measures", "Performance_Dictionary.npy"), allow_pickle=True)[()]
        d_prime = behavioural_dictionary["visual_d_prime"]

        if d_prime >= post_threshold:
            post_learning_sessions.append(session)

        if d_prime < post_threshold and d_prime >= intermediate_threshold:
            intermediate_learning_sessions.append(session)

        if d_prime < intermediate_threshold:
            pre_learning_sessions.append(session)

    return pre_learning_sessions, intermediate_learning_sessions, post_learning_sessions





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



control_session_list_post = [
    # Controls 46 sessions

    # 78.1A - 6
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_21_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_24_Discrimination_Imaging",

    # 78.1D - 8
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_23_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_25_Discrimination_Imaging",

    # 4.1B - 7
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging",

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

mutant_session_list_post = [
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
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_14_Discrimination_Imaging"

]





pixel_assignments = np.load("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Final_Consensus_Clusters.npy")
plt.imshow(pixel_assignments)
plt.show()


# Decoding Parameters
start_window = -42
stop_window = 84

# Split Sessions By D Prime
intermeidate_threshold = 1
post_threshold = 2
control_pre_learning_sessions, control_intermediate_learning_sessions, control_post_learning_sessions = split_sessions_By_d_prime(control_session_list, intermeidate_threshold, post_threshold)
mutant_pre_learning_sessions, mutant_intermediate_learning_sessions, mutant_post_learning_sessions = split_sessions_By_d_prime(mutant_session_list, intermeidate_threshold, post_threshold)


condition = "Stimuli_Onsets_Matched_Reaction_Times_1100_1200.npy"
control_session_meta_list = [control_pre_learning_sessions, control_intermediate_learning_sessions, control_post_learning_sessions]
mutant_session_meta_list = [mutant_pre_learning_sessions, mutant_intermediate_learning_sessions, mutant_post_learning_sessions]

get_group_averages(control_session_meta_list, mutant_session_meta_list, condition, start_window, stop_window, selected_region=17, min_trials=5)