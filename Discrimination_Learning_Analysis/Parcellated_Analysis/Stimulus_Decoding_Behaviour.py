import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
import tables
from matplotlib import cm


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



def get_ai_filename(base_directory):

    ai_filename = None

    # Get List of all files
    file_list = os.listdir(base_directory)

    # Get .h5 files
    h5_file_list = []
    for file in file_list:
        if file[-3:] == ".h5":
            h5_file_list.append(file)

    # File the H5 file which is two dates seperated by a dash
    for h5_file in h5_file_list:

        original_filename = h5_file

        # Remove Ending
        h5_file = h5_file[0:-3]

        # Split By Dashes
        h5_file = h5_file.split("-")

        if len(h5_file) == 2 and h5_file[0].isnumeric() and h5_file[1].isnumeric():
            return original_filename



def load_ai_recorder_file(ai_recorder_file_location):

    table = tables.open_file(ai_recorder_file_location, mode='r')
    data = table.root.Data
    number_of_seconds = np.shape(data)[0]
    number_of_channels = np.shape(data)[1]
    sampling_rate = np.shape(data)[2]
    data_matrix = np.zeros((number_of_channels, number_of_seconds * sampling_rate))

    for second in range(number_of_seconds):
        data_window = data[second]
        start_point = second * sampling_rate
        for channel in range(number_of_channels):
            data_matrix[channel, start_point:start_point + sampling_rate] = data_window[channel]

    data_matrix = np.clip(data_matrix, a_min=0, a_max=None)

    return data_matrix


def invert_dictionary(dictionary):
    inv_map = {v: k for k, v in dictionary.items()}
    return inv_map







def create_ai_tensor_raw(ai_data, onset_list, start_window, stop_window, frame_times, number_of_frames):

    ai_tensor = []
    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window

        if trial_start >= 0 and trial_stop < number_of_frames:
            trial_start_time = frame_times[trial_start]
            trial_stop_time = frame_times[trial_stop]

            trial_data = ai_data[:, trial_start_time:trial_stop_time]
            ai_tensor.append(trial_data)

    # Get Smallest Trace
    trial_lengths = []
    for trial in ai_tensor:
        trial_length = np.shape(trial)[1]
        trial_lengths.append(trial_length)
    smallest_length = np.min(trial_lengths)

    # Get Cut Tensor
    cut_ai_tensor = []
    for trial in ai_tensor:
        cut_ai_tensor.append(trial[:, 0:smallest_length-1])

    cut_ai_tensor = np.array(cut_ai_tensor)
    return cut_ai_tensor




def get_ai_tensors(base_directory, condition_1_onsets, condition_2_onsets, start_window, stop_window):

    # Load AI Recroder Data
    ai_filename = get_ai_filename(base_directory)
    ai_data = load_ai_recorder_file(os.path.join(base_directory, ai_filename))

    # Load Frame Times
    frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    frame_times = invert_dictionary(frame_times)
    number_of_frames = len(frame_times.keys())

    # Get AI Tensors
    condition_1_tensor = create_ai_tensor_raw(ai_data, condition_1_onsets, start_window, stop_window, frame_times, number_of_frames)
    condition_2_tensor = create_ai_tensor_raw(ai_data, condition_2_onsets, start_window, stop_window, frame_times, number_of_frames)

    # Stimuli Dictionary
    stimuli_dictionary = create_stimuli_dictionary()

    # Get Selected Stimuli Tensor
    selected_stimuli = ["Running", "Lick"]
    condition_1_selected_tensor = []
    condition_2_selected_tensor = []

    for stimulus in selected_stimuli:
        condition_1_trace = condition_1_tensor[:, stimuli_dictionary[stimulus]]
        condition_1_selected_tensor.append(condition_1_trace)

        condition_2_trace = condition_2_tensor[:, stimuli_dictionary[stimulus]]
        condition_2_selected_tensor.append(condition_2_trace)

    condition_1_selected_tensor = np.array(condition_1_selected_tensor)
    condition_2_selected_tensor = np.array(condition_2_selected_tensor)

    condition_1_selected_tensor = np.moveaxis(condition_1_selected_tensor, [0, 1, 2], [2, 0, 1])
    condition_2_selected_tensor = np.moveaxis(condition_2_selected_tensor, [0, 1, 2], [2, 0, 1])

    return condition_1_selected_tensor, condition_2_selected_tensor


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




def perform_k_fold_cross_validation(data, labels, number_of_folds=10):

    score_list = []
    weight_list = []

    # Get Indicies To Split Data Into N Train Test Splits
    k_fold_object = StratifiedKFold(n_splits=number_of_folds, shuffle=True) #random_state=42

    # Iterate Through Each Split
    for train_index, test_index in k_fold_object.split(data, y=labels):

        # Split Data Into Train and Test Sets
        data_train, data_test = data[train_index], data[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

        # Train Model
        model = LogisticRegression(max_iter=500)
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

    weight_list = np.array(weight_list)
    mean_weights = np.mean(weight_list, axis=0)
    return mean_score, mean_weights


def perform_decoding(base_directory, condition_1_onsets, condition_2_onsets, start_window, stop_window):

    # Balanace Trial Numbers
    condition_1_onsets, condition_2_onsets = balance_classes(condition_1_onsets, condition_2_onsets)
    print("Number Of Condition 1 Onsets Post Balance", len(condition_1_onsets))
    print("Number of condition 2 onsets post balance", len(condition_2_onsets))

    # Get AI Tensors
    condition_1_data, condition_2_data = get_ai_tensors(base_directory, condition_1_onsets, condition_2_onsets, start_window, stop_window)
    print("Condition 1 data",  np.shape(condition_1_data))
    print("Condition 2 data", np.shape(condition_2_data))

    combined_data = np.concatenate([condition_1_data, condition_2_data], axis=0)
    print("Combined Data Shape", np.shape(combined_data))

    # Create Labels
    condition_1_labels = np.ones(np.shape(condition_1_data)[0])
    condition_2_labels = np.zeros(np.shape(condition_2_data)[0])
    data_labels = np.concatenate([condition_1_labels, condition_2_labels], axis=0)
    print("Data Labels Shape", np.shape(data_labels))

    # Perform Decoding With K Fold Cross Validation
    temporal_score_list = []
    number_of_timepoints = np.shape(combined_data)[1]
    print("Number Of Timepoints", number_of_timepoints)

    for timepoint_index in range(number_of_timepoints):
        timepoint_data = combined_data[:, timepoint_index]

        mean_score, mean_weights = perform_k_fold_cross_validation(timepoint_data, data_labels)
        temporal_score_list.append(mean_score)

    return temporal_score_list


def decode_based_on_behaviour(session_list, condition_1, condition_2, start_window, stop_window):

    number_of_sessions = len(session_list)
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)

    #rows = 1
    #columns = number_of_sessions

    colourmap = cm.get_cmap('plasma')

    for session_index in range(number_of_sessions):
        base_directory = session_list[session_index]

        # Load Onsets
        vis_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_1))
        vis_2_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_2))

        # Perform Decoding
        decoding_performance = perform_decoding(base_directory, vis_1_onsets, vis_2_onsets, start_window, stop_window)

        session_colour = colourmap(float(session_index) / number_of_sessions)
        x_values = np.linspace(start=start_window, stop=stop_window, num=len(decoding_performance))

        axis_1.plot(x_values, decoding_performance, c=session_colour)
        axis_1.set_ylim([0.4, 1])




    plt.show()





session_list = [
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging"]

# Decoding Parameters
start_window = -28
stop_window = 57
condition_1 = "visual_1_all_onsets.npy"
condition_2 = "visual_2_all_onsets.npy"

decode_based_on_behaviour(session_list, condition_1, condition_2, start_window, stop_window)
