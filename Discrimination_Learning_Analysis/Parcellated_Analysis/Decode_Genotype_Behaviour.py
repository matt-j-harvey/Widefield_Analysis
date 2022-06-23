import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
from matplotlib import cm
from matplotlib.gridspec import GridSpec


def create_stimuli_dictionary():

    channel_index_dictionary = {
        "Photodiode": 0,
        "Reward": 1,
        "Lick": 2,
        "Visual 1": 3,
        "Visual 2": 4,
        "Odour 1": 5,
        "Odour 2": 6,
        "Irrelevance": 7,
        "Running": 8,
        "Trial End": 9,
        "Camera Trigger": 10,
        "Camera Frames": 11,
        "LED 1": 12,
        "LED 2": 13,
        "Mousecam": 14,
        "Optogenetics": 15,

    }

    return channel_index_dictionary



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



def balance_classes(condition_1_onsets, condition_2_onsets):

    condition_1_onsets = list(condition_1_onsets)
    condition_2_onsets = list(condition_2_onsets)

    number_of_condition_1_trials = len(condition_1_onsets)
    number_of_condition_2_trials = len(condition_2_onsets)
    print("Condition 1 trials pre balance: ", number_of_condition_1_trials)
    print("Condition 2 trials pre balance: ", number_of_condition_2_trials)

    # If There Are Balanced, Great! Dont change a thing
    if number_of_condition_1_trials == number_of_condition_2_trials:
        return condition_1_onsets, condition_2_onsets

    # Else Remove Random Samples From The Larger Class
    else:
        smallest_class = np.min([number_of_condition_1_trials, number_of_condition_2_trials])
        largest_class = np.max([number_of_condition_1_trials, number_of_condition_2_trials])
        trials_to_remove = largest_class - smallest_class

        if number_of_condition_1_trials > number_of_condition_2_trials:
            for x in range(trials_to_remove):
                random_index = int(np.random.uniform(low=0, high=len(condition_1_onsets)))
                del condition_1_onsets[random_index]
            return condition_1_onsets, condition_2_onsets


        else:
            for x in range(trials_to_remove):
                random_index = int(np.random.uniform(low=0, high=len(condition_2_onsets)))
                del condition_2_onsets[random_index]
            return condition_1_onsets, condition_2_onsets




def perform_k_fold_cross_validation(data, labels, number_of_folds=5):

    # Remove Nans
    data = np.nan_to_num(data)

    score_list = []
    weight_list = []

    # Get Indicies To Split Data Into N Train Test Splits
    k_fold_object = StratifiedKFold(n_splits=number_of_folds, shuffle=True) #random_state=42

    # Iterate Through Each Split
    for train_index, test_index in k_fold_object.split(data, y=labels):

        # Split Data Into Train and Test Sets
        data_train, data_test = data[train_index], data[test_index]

        labels_train, labels_test = labels[train_index], labels[test_index]
        #print("Labels Train Mean", np.mean(labels_train))
        #print("Labels Test Mean", np.mean(labels_test))

        # Train Model
        model = LogisticRegression(max_iter=500)
        #model = LinearDiscriminantAnalysis()
        model.fit(data_train, labels_train)

        # Test Model
        model_score = model.score(data_test, labels_test)

        # Add Score To Score List
        score_list.append(model_score)

        # Get Model Weights
        model_weights = model.coef_
        weight_list.append(model_weights)

    # Return Mean Score and Mean Model Weights

    mean_score = np.mean(score_list)
    score_sd = np.std(score_list)

    weight_list = np.array(weight_list)
    mean_weights = np.mean(weight_list, axis=0)
    return mean_score, mean_weights, score_sd


def create_ai_tensor(ai_data, onset_list, start_window, stop_window):

    number_of_frames = np.shape(ai_data)[1]
    ai_tensor = []
    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window

        if trial_start >= 0 and trial_stop < number_of_frames:
            trial_data = ai_data[:, trial_start:trial_stop]
            ai_tensor.append(trial_data)

    ai_tensor = np.array(ai_tensor)
    return ai_tensor


def view_ai_average_traces(base_directory, condition_1_onsets, condition_2_onsets, start_window, stop_window):

    # Load AI Recroder Data
    ai_data = np.load(os.path.join(base_directory, "Downsample_AI_Data.npy"))

    # Get AI Tensors
    condition_1_tensor = create_ai_tensor(ai_data, condition_1_onsets, start_window, stop_window)
    condition_2_tensor = create_ai_tensor(ai_data, condition_2_onsets, start_window, stop_window)
    print(np.shape(condition_1_tensor))
    print("Condition 2 tensor", np.shape(condition_2_tensor))
    # Get Averages
    condition_1_average = np.mean(condition_1_tensor, axis=0)
    condition_2_average = np.mean(condition_2_tensor, axis=0)

    # Plot These
    rows = 1
    columns = 2
    figure_1 = plt.figure()
    condition_1_axis = figure_1.add_subplot(rows, columns, 1)
    condition_2_axis = figure_1.add_subplot(rows, columns, 2)

    trace_list = ["Lick", "Visual 1", "Visual 2", "Running"]
    stimuli_dictionary = create_stimuli_dictionary()

    for trace in trace_list:
        condition_1_axis.plot(condition_1_average[stimuli_dictionary[trace]])
        condition_2_axis.plot(condition_2_average[stimuli_dictionary[trace]])

    plt.show()

def normalise_activity_matrix(activity_matrix, early_cutoff_window=1500):

    # Subtract Min
    min_vector = np.min(activity_matrix[early_cutoff_window:], axis=0)
    activity_matrix = np.subtract(activity_matrix, min_vector)

    # Divide By Max
    max_vector = np.max(activity_matrix[early_cutoff_window:], axis=0)
    activity_matrix = np.divide(activity_matrix, max_vector)

    activity_matrix = np.nan_to_num(activity_matrix)
    return activity_matrix



def remove_early_onsets(onsets, window=1500):

    thresholded_onsets = []

    for onset in onsets:
        if onset > window:
            thresholded_onsets.append(onset)

    return thresholded_onsets



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


def load_session_data(base_directory, condition, start_window, stop_window, early_cutoff_window=1500):

    # Load Neural Data
    activity_matrix = np.load(os.path.join(base_directory, "Cluster_Activity_Matrix.npy"))

    # Normalise Activity Matrix
    activity_matrix = normalise_activity_matrix(activity_matrix, early_cutoff_window)

    # Remove Background Activity
    activity_matrix = activity_matrix[:, 1:]

    # Load Onsets
    onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition))

    # Remove Early Onsets
    onsets = remove_early_onsets(onsets, window=early_cutoff_window)

    # Load Neural Tensors
    neural_tensor = get_trial_tensor(activity_matrix, onsets, start_window, stop_window)

    return neural_tensor


def view_cortical_vector(cluster_assignments, activity_vector):

    unique_clusters = list(np.unique(cluster_assignments))
    unique_clusters.remove(0)
    number_of_clusters = len(unique_clusters)

    cortical_image = np.zeros(np.shape(cluster_assignments))

    for cluster_index in range(number_of_clusters):
        cluster = unique_clusters[cluster_index]
        if cluster != 0:
            cluster_mask = np.where(cluster_assignments == cluster, 1, 0)
            cluster_pixels = np.nonzero(cluster_mask)
            print("Activity value", activity_vector[cluster_index])
            cortical_image[cluster_pixels] = activity_vector[cluster_index]

    return cortical_image


def view_weights(weights_list, timepoint_values):

    # Load Cluster Assignments
    cluster_assignments = np.load(r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Final_Consensus_Clusters.npy")

    # Iterate Through Timepoints
    number_of_timepoints = len(timepoint_values)
    plt.ion()
    for timepoint_index in range(number_of_timepoints):
        timepoint_weights = weights_list[timepoint_index]
        timepoint_time = timepoint_values[timepoint_index]
        print("Timepoint Weights", timepoint_weights)

        cortical_image = view_cortical_vector(cluster_assignments, timepoint_weights)
        plt.title(str(timepoint_time))
        plt.imshow(cortical_image, vmax=5, vmin=-5, cmap='bwr')
        plt.draw()
        plt.pause(1)
        plt.clf()




def run_genotype_decoding_analysis(control_session_list, mutant_session_list, condition, start_window, stop_window):


    number_of_control_sessions = len(control_session_list)
    number_of_mutant_sessions = len(mutant_session_list)

    combined_data_list = []
    combined_label_list = []

    # Load Control Data
    for session_index in range(number_of_control_sessions):
        base_directory = control_session_list[session_index]
        neural_tensor = load_session_data(base_directory, condition, start_window, stop_window)
        print(base_directory, "Neural tensor shape", np.shape(neural_tensor))
        labels = np.zeros(np.shape(neural_tensor)[0])
        combined_data_list.append(neural_tensor)
        combined_label_list.append(labels)

    # Load Mutant Data
    for session_index in range(number_of_mutant_sessions):
        base_directory = mutant_session_list[session_index]
        neural_tensor = load_session_data(base_directory, condition, start_window, stop_window)
        print(base_directory, "Neural tensor shape", np.shape(neural_tensor))
        labels = np.ones(np.shape(neural_tensor)[0])
        combined_data_list.append(neural_tensor)
        combined_label_list.append(labels)

    combined_data_list = np.vstack(combined_data_list)
    combined_label_list = np.concatenate(combined_label_list)

    print("Combined Data List Shape", np.shape(combined_data_list))
    print("Combined Label List Shape", np.shape(combined_label_list))


    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)

    # Iterate Through Each Session
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, 36)

    # Decode Each Timepoint
    temporal_score_list = []
    temporal_sd_list = []
    mean_weights_list = []
    number_of_timepoints = np.shape(combined_data_list)[1]
    for timepoint_index in range(number_of_timepoints):
        mean_score, mean_weights, score_sd = perform_k_fold_cross_validation(combined_data_list[:, timepoint_index], combined_label_list)

        print("Timepoint: ", timepoint_index, "Mean Score: ", mean_score)

        temporal_score_list.append(mean_score)
        mean_weights_list.append(mean_weights)
        temporal_sd_list.append(score_sd)

    score_upper_bound = np.add(temporal_score_list, temporal_sd_list)
    score_lower_bound = np.subtract(temporal_score_list, temporal_sd_list)

    axis_1.plot(x_values, temporal_score_list)
    axis_1.fill_between(x=x_values, y1=score_lower_bound, y2=score_upper_bound, alpha=0.2)
    axis_1.axvline(0, color='k', linestyle='--')
    plt.show()

    mean_weights_list = np.array(mean_weights_list)
    mean_weights_list = mean_weights_list[:, 0]
    print("MEan weights list", np.shape(mean_weights_list))
    view_weights(mean_weights_list, x_values)




wildtye_session_list = [
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    #"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging",

    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_16_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_17_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_19_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_21_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_24_Discrimination_Imaging",

    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_09_25_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_09_29_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_10_01_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_10_03_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_10_05_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_10_07_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging",

]


mutant_session_list = [
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_08_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_10_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_12_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_14_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_16_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_18_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_23_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_25_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_02_27_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_03_01_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_03_03_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging",

    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_09_28_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_09_30_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_09_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_11_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_13_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_15_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_17_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging",
]

# Decoding Parameters
start_window = -28
stop_window = 57
condition_name = "visual_1_all_onsets.npy"
#condition_2 = "visual_2_all_onsets.npy"

# Run Decoding
mouse_score_list = run_genotype_decoding_analysis(wildtye_session_list, mutant_session_list, condition_name, start_window, stop_window)
