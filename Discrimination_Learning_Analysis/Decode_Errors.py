import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
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
        #print("Data Train", np.shape(data_train))
        #print("Data Test", np.shape(data_test))

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

def normalise_activity_matrix(activity_matrix):

    # Subtract Min
    min_vector = np.min(activity_matrix, axis=0)
    activity_matrix = np.subtract(activity_matrix, min_vector)

    # Divide By Max
    max_vector = np.max(activity_matrix, axis=0)
    activity_matrix = np.divide(activity_matrix, max_vector)

    activity_matrix = np.nan_to_num(activity_matrix)
    return activity_matrix


def perform_decoding(delta_f_matrix, condition_1_onsets, condition_2_onsets, start_window, stop_window):

    # Balanace Trial Numbers
    condition_1_onsets, condition_2_onsets = balance_classes(condition_1_onsets, condition_2_onsets)

    # Get Trial Tensors
    condition_1_data = get_trial_tensor(delta_f_matrix, condition_1_onsets, start_window, stop_window)
    condition_2_data = get_trial_tensor(delta_f_matrix, condition_2_onsets, start_window, stop_window)
    combined_data = np.concatenate([condition_1_data, condition_2_data], axis=0)

    # Create Labels
    condition_1_labels = np.ones(np.shape(condition_1_data)[0])
    condition_2_labels = np.zeros(np.shape(condition_2_data)[0])
    data_labels = np.concatenate([condition_1_labels, condition_2_labels], axis=0)

    # Perform Decoding With K Fold Cross Validation
    temporal_score_list = []
    temporal_sd_list = []
    mean_weights_list = []
    number_of_timepoints = np.shape(combined_data)[1]

    for timepoint_index in range(number_of_timepoints):
        timepoint_data = combined_data[:, timepoint_index]
        mean_score, mean_weights, score_sd = perform_k_fold_cross_validation(timepoint_data, data_labels)
        temporal_score_list.append(mean_score)
        mean_weights_list.append(mean_weights)
        temporal_sd_list.append(score_sd)

    mean_weights_list = np.array(mean_weights_list)
    mean_weights_list = mean_weights_list[:, 0, :]
    mean_weights_list = np.reshape(mean_weights_list, (np.shape(mean_weights_list)[0], np.shape(mean_weights_list)[1]))

    return temporal_score_list, temporal_sd_list, mean_weights_list






def run_decoding_analysis(session_list, condition_1, condition_2, start_window, stop_window):

    # Predict D Prime Of session
    decoding_scores_list = []
    colourmap = cm.get_cmap('plasma')

    decoding_figure = plt.figure()
    number_of_sessions = len(session_list)
    grid_spec = GridSpec(nrows=number_of_sessions, ncols=3)

    decoding_performance_axis = decoding_figure.add_subplot(grid_spec[:, 0:2])

    # Iterate Through Each Session
    x_values = list(range(start_window, stop_window))

    for session_index in range(number_of_sessions):
        base_directory = session_list[session_index]

        # Load Neural Data
        activity_matrix = np.load(os.path.join(base_directory, "Cluster_Activity_Matrix.npy"))

        # Normalise Activity Matrix
        activity_matrix = normalise_activity_matrix(activity_matrix)

        # Remove Background Activity
        activity_matrix = activity_matrix[:, 1:]

        # Load Onsets
        vis_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_1))
        vis_2_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_2))

        # View Downsampled AI
        #view_ai_average_traces(base_directory, vis_1_onsets, vis_2_onsets, start_window, stop_window)

        # Decode Visual Stimuli
        decoding_scores, decoding_sds, mean_weights = perform_decoding(activity_matrix, vis_1_onsets, vis_2_onsets, start_window, stop_window)
        decoding_scores_list.append(decoding_scores)

        # Plot These
        plot_colour = colourmap(float(session_index)/number_of_sessions)
        decoding_performance_axis.plot(x_values, decoding_scores, c=plot_colour)

        # Shade STDs
        """
        x_values = list(range(len(decoding_scores)))
        upperbound = np.add(decoding_scores, decoding_sds)
        lower_bound = np.subtract(decoding_scores, decoding_sds)
        plt.fill_between(x=x_values, y1=lower_bound, y2=upperbound, color=plot_colour, alpha=0.5)
        """

        mean_weights_magnitude = np.max(np.abs(mean_weights))
        mean_weights_axis = decoding_figure.add_subplot(grid_spec[session_index, 2])
        mean_weights_axis.imshow(np.transpose(mean_weights), cmap='bwr',vmin=-mean_weights_magnitude, vmax=mean_weights_magnitude)


    decoding_performance_axis.axvline(0, c='k', linestyle='--')
    decoding_performance_axis.set_ylim([0.2, 1])
    plt.show()



session_list = [

    #"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging",

    #"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_22_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_24_Discrimination_Imaging",

    #"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_07_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_09_Discrimination_Imaging",

    #"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_21_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_24_Discrimination_Imaging",

    #"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_25_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_23_Discrimination_Imaging",

    #"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_07_Discrimination_Imaging"
    #"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging",

]


session_list = [
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_17_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging",

    "/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_03_03_Discrimination_Imaging",


]

# Decoding Parameters
start_window = -50
stop_window = 100
condition_1 = "visual_2_correct_onsets.npy"
condition_2 = "visual_2_incorrect_onsets.npy"

# Run Decoding
mouse_score_list = run_decoding_analysis(session_list, condition_1, condition_2, start_window, stop_window)
