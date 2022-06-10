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


def create_ai_tensor_raw(ai_data, onset_list, start_window, stop_window, frame_times, number_of_frames):

    ai_tensor = []
    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window

        if trial_start >= 0 and trial_stop < number_of_frames:
            trial_start_time = frame_times[trial_start]
            trial_stop_time = frame_times[trial_stop]

            trial_data = ai_data[:, trial_start_time:trial_stop_time]
            print("Trial Data Shape", np.shape(trial_data))
            ai_tensor.append(trial_data)

    # Get Smallest Trace
    trial_lengths = []
    for trial in ai_tensor:
        trial_length = np.shape(trial)[1]
        trial_lengths.append(trial_length)
    smallest_length = np.min(trial_lengths)

    print("Smallest Length", smallest_length)

    # Get Cut Tensor
    cut_ai_tensor = []
    for trial in ai_tensor:
        cut_ai_tensor.append(trial[:, 0:smallest_length-1])

    cut_ai_tensor = np.array(cut_ai_tensor)
    return cut_ai_tensor


def view_ai_average_traces_raw(base_directory, condition_1_onsets, condition_2_onsets, start_window, stop_window, condition_1, condition_2):

    # Load AI Recroder Data
    ai_filename = get_ai_filename(base_directory)
    ai_data = load_ai_recorder_file(os.path.join(base_directory, ai_filename))

    # Load Frame Times
    frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    frame_times = invert_dictionary(frame_times)
    number_of_frames = len(frame_times.keys())
    print("Number of frames", number_of_frames)

    # Get AI Tensors
    condition_1_tensor = create_ai_tensor_raw(ai_data, condition_1_onsets, start_window, stop_window, frame_times, number_of_frames)
    condition_2_tensor = create_ai_tensor_raw(ai_data, condition_2_onsets, start_window, stop_window, frame_times, number_of_frames)
    print("Condition 1 tensor", np.shape(condition_1_tensor))

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

    condition_1_axis.set_title(condition_1)
    condition_2_axis.set_title(condition_2)

    # Create Save Directory
    save_directory = os.path.join(base_directory, "AI_Sanity_Checks")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    figure_name = condition_1.replace(".npy","") + "_" + condition_2.replace(".npy","") + ".png"
    plt.savefig(os.path.join(save_directory, figure_name))
    plt.close()


def view_ai_average_traces(base_directory, condition_1_onsets, condition_2_onsets, start_window, stop_window):

    # Load AI Recroder Data
    ai_data = np.load(os.path.join(base_directory, "Downsampled_AI_Matrix.npy"))

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



def visualise_full_ai_data(session_list, condition_1, condition_2, start_window, stop_window):

    number_of_sessions = len(session_list)
    for session_index in range(number_of_sessions):
        base_directory = session_list[session_index]

        # Load Neural Data
        activity_matrix = np.load(os.path.join(base_directory, "Cluster_Activity_Matrix.npy"))

        # Load Onsets
        vis_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_1))
        vis_2_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_2))

        # View Raw AI
        view_ai_average_traces_raw(base_directory, vis_1_onsets, vis_2_onsets, start_window, stop_window, condition_1, condition_2)


def visualise_downsampled_ai_data(session_list, condition_1, condition_2, start_window, stop_window):

    # Iterate Through Each Session
    number_of_sessions = len(session_list)
    for session_index in range(number_of_sessions):
        base_directory = session_list[session_index]

        # Load Neural Data
        activity_matrix = np.load(os.path.join(base_directory, "Cluster_Activity_Matrix.npy"))

        # Load Onsets
        vis_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_1))
        vis_2_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_2))

        # View Downsampled AI
        view_ai_average_traces(base_directory, vis_1_onsets, vis_2_onsets, start_window, stop_window)




session_list = [
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging",
]


# Decoding Parameters
start_window = -20
stop_window = 40
condition_1 = "visual_1_all_onsets.npy"
condition_2 = "visual_2_all_onsets.npy"

# Run Decoding
visualise_downsampled_ai_data(session_list, condition_1, condition_2, start_window, stop_window)
#visualise_full_ai_data(session_list, condition_1, condition_2, start_window, stop_window)
