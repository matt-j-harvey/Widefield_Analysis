import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
import tables
import random
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist, cdist


def sort_matrix(matrix):

    # Cluster Matrix
    Z = ward(pdist(matrix))

    # Get Dendogram Leaf Order
    new_order = leaves_list(Z)

    # Sorted Matrix
    sorted_matrix = matrix[:, new_order][new_order]

    return sorted_matrix


def ResampleLinear1D(original, targetLen):
    original = np.array(original, dtype=float)
    index_arr = np.linspace(0, len(original) - 1, num=targetLen, dtype=float)
    index_floor = np.array(index_arr, dtype=int)  # Round down
    index_ceil = index_floor + 1
    index_rem = index_arr - index_floor  # Remain
    val1 = original[index_floor]
    val2 = original[index_ceil % len(original)]
    interp = val1 * (1.0 - index_rem) + val2 * index_rem
    assert (len(interp) == targetLen)
    return interp


def invert_dictionary(dictionary):
    inv_map = {v: k for k, v in dictionary.items()}
    return inv_map


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



def normalise_activity_matrix(activity_matrix):

    # Subtract Min
    min_vector = np.min(activity_matrix, axis=0)
    activity_matrix = np.subtract(activity_matrix, min_vector)

    # Divide By Max
    max_vector = np.max(activity_matrix, axis=0)
    activity_matrix = np.divide(activity_matrix, max_vector)

    return activity_matrix


def forceAspect(ax, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)


def view_raster(activity_matrix):

    # activity_matrix = activity_matrix[0:10000]
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    axis_1.imshow(np.transpose(activity_matrix), cmap='jet')
    forceAspect(axis_1, aspect=4)
    plt.show()


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


def downsample_ai_traces(base_directory, delta_f_matrix):

    # Load Frame Times
    frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    frame_times = invert_dictionary(frame_times)

    # Load AI Recorder File
    ai_filename = get_ai_filename(base_directory)
    ai_data = load_ai_recorder_file(os.path.join(base_directory, ai_filename))

    # Extract Relevant Traces
    stimuli_dictionary = create_stimuli_dictionary()
    running_trace = ai_data[stimuli_dictionary["Running"]]
    lick_trace = ai_data[stimuli_dictionary["Lick"]]

    # Get Data Structure
    number_of_timepoints = np.shape(delta_f_matrix)[0]
    imaging_start = frame_times[0]
    imaging_stop = frame_times[number_of_timepoints - 1]

    # Get Traces Only While Imaging
    imaging_running_trace = running_trace[imaging_start:imaging_stop]
    imaging_lick_trace = lick_trace[imaging_start:imaging_stop]

    # Downsample Traces
    downsampled_running_trace = ResampleLinear1D(imaging_running_trace, number_of_timepoints)
    downsampled_lick_trace = ResampleLinear1D(imaging_lick_trace, number_of_timepoints)

    # Normalise
    downsampled_lick_trace = np.subtract(downsampled_lick_trace, np.min(downsampled_lick_trace))
    downsampled_lick_trace = np.divide(downsampled_lick_trace, np.max(downsampled_lick_trace))
    downsampled_running_trace = np.subtract(downsampled_running_trace, np.min(downsampled_running_trace))
    downsampled_running_trace = np.divide(downsampled_running_trace, np.max(downsampled_running_trace))

    return downsampled_running_trace, downsampled_lick_trace


class custom_rnn(torch.nn.Module):

    def __init__(self, n_inputs, n_neurons, device):
        super(custom_rnn, self).__init__()

        # Initialise Weights
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.input_weights = torch.nn.Linear(n_inputs, n_neurons, bias=True, device=device).float()
        self.recurrent_weights = torch.nn.Linear(n_neurons, n_neurons, bias=True, device=device).float()

        # Initialise Hidden State
        self.hidden_state = torch.zeros(1, n_neurons, dtype=torch.float, device=device)


    def forward(self, external_input):

        # Get External Input
        input_contribution = self.input_weights(external_input)

        # Get Recurrent Input
        recurrent_contribution = self.recurrent_weights(self.hidden_state)

        # Sum These and Biases
        new_activity = input_contribution + recurrent_contribution

        # Put Through Activation Function
        new_activity = torch.sigmoid(new_activity)

        self.hidden_state = new_activity

        return new_activity


def get_activity_tensor(activity_matrix, onset, start_window, stop_window):
    trial_start = onset + start_window

    trial_stop = onset + stop_window

    initial_state_timepoint = trial_start - 1

    trial_activity = activity_matrix[trial_start:trial_stop]

    initial_state = activity_matrix[initial_state_timepoint]

    return initial_state, trial_activity


def create_input_tensor(onset, running_trace, lick_trace, start_window, stop_window, stimuli_index):
    trial_length = stop_window - start_window

    stimuli_1_input = np.eye(trial_length)

    stimuli_2_input = np.zeros((trial_length, trial_length))

    if stimuli_index == 1:

        stimuli_regressor = np.vstack([stimuli_1_input, stimuli_2_input])

    elif stimuli_index == 2:

        stimuli_regressor = np.vstack([stimuli_2_input, stimuli_1_input])

    trial_start = onset + start_window

    trial_stop = onset + stop_window

    running_regressor = running_trace[trial_start:trial_stop]

    lick_regressor = lick_trace[trial_start:trial_stop]

    input_tensor = np.vstack([stimuli_regressor, running_regressor, lick_regressor])

    return input_tensor


def create_training_data(activity_matrix, running_trace, lick_trace, onsets, stimuli_index, start_window, stop_window):
    initial_states = []

    inputs = []

    actual_activity = []

    for onset in onsets:
        trial_initial_state, trial_activity_tensor = get_activity_tensor(activity_matrix, onset, start_window,
                                                                         stop_window)

        trial_inputs = create_input_tensor(onset, running_trace, lick_trace, start_window, stop_window, stimuli_index)

        initial_states.append(trial_initial_state)

        inputs.append(trial_inputs)

        actual_activity.append(trial_activity_tensor)

    initial_states = np.array(initial_states)

    inputs = np.array(inputs)

    actual_activity = np.array(actual_activity)

    return initial_states, inputs, actual_activity


def split_condition_data(initial_states, inputs, actual_activity):
    number_of_trials = len(initial_states)

    trial_indexes = list(range(number_of_trials))

    random.shuffle(trial_indexes)

    validation_fraction = 0.2

    number_of_validation_trials = int(number_of_trials * validation_fraction)

    test_indexes = trial_indexes[0:number_of_validation_trials]

    train_indexes = trial_indexes[number_of_validation_trials:]

    train_initial_states = []

    train_inputs = []

    train_actual_activity = []

    test_initial_states = []

    test_inputs = []

    test_actual_activity = []

    for index in train_indexes:
        train_initial_states.append(initial_states[index])

        train_inputs.append(inputs[index])

        train_actual_activity.append(actual_activity[index])

    for index in test_indexes:
        test_initial_states.append(initial_states[index])

        test_inputs.append(inputs[index])

        test_actual_activity.append(actual_activity[index])

    return [train_initial_states, train_inputs, train_actual_activity, test_initial_states, test_inputs,
            test_actual_activity]


def jointly_shuffle_lists(list_1, list_2, list_3):
    shuffle_list_1 = []

    shuffle_list_2 = []

    shuffle_list_3 = []

    number_of_items = len(list_1)

    index_list = list(range(0, number_of_items))

    random.shuffle(index_list)

    for index in index_list:
        shuffle_list_1.append(list_1[index])

        shuffle_list_2.append(list_2[index])

        shuffle_list_3.append(list_3[index])

    return shuffle_list_1, shuffle_list_2, shuffle_list_3


def train_network(rnn, input_tensor_list, hidden_state_list, activity_tensor_list, crtierion, optimiser,
                  number_of_trials, number_of_timepoints, number_of_regions):
    for trial in range(number_of_trials):

        # Set Hidden State

        rnn.hidden_state = hidden_state_list[trial]

        # Iterate Through Each Timestep

        output_tensor = torch.zeros(size=(trial_length, number_of_regions), dtype=torch.float32, device=device)

        for timepoint in range(number_of_timepoints):
            output_tensor[timepoint] = rnn(input_tensor_list[trial, :, timepoint])

        # Get Loss
        loss = crtierion(output_tensor, activity_tensor_list[trial])
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    return loss


def get_validation_loss(rnn, input_tensor_list, hidden_state_list, activity_tensor_list, number_of_trials, number_of_timepoints, number_of_regions, crtierion, device):

    loss_list = []

    for trial in range(number_of_trials):

        # Set Hidden State
        rnn.hidden_state = hidden_state_list[trial]

        # Iterate Through Each Timestep
        output_tensor = torch.zeros(size=(trial_length, number_of_regions), dtype=torch.float32, device=device)

        for timepoint in range(number_of_timepoints):
            output_tensor[timepoint] = rnn(input_tensor_list[trial, :, timepoint])

        # Get Loss
        loss = crtierion(output_tensor, activity_tensor_list[trial])
        loss_list.append(loss.cpu().detach().numpy())

    mean_loss = np.mean(loss_list)

    return mean_loss


def split_data(condition_1_data, condition_2_data, device):

    # Parse Inputs
    condition_1_initial_states = condition_1_data[0]
    condition_1_inputs = condition_1_data[1]
    condition_1_actual_activity = condition_1_data[2]

    condition_2_initial_states = condition_2_data[0]
    condition_2_inputs = condition_2_data[1]
    condition_2_actual_activity = condition_2_data[2]

    # Split Conditions Individually
    [c1_train_initial_states, c1_train_inputs, c1_train_actual_activity,
     c1_test_initial_states, c1_test_inputs, c1_test_actual_activity] = split_condition_data(condition_1_initial_states, condition_1_inputs, condition_1_actual_activity)

    [c2_train_initial_states, c2_train_inputs, c2_train_actual_activity,
     c2_test_initial_states, c2_test_inputs, c2_test_actual_activity] = split_condition_data(condition_2_initial_states, condition_2_inputs, condition_2_actual_activity)

    # Combine These
    train_combined_initial_states = np.vstack([c1_train_initial_states, c2_train_initial_states])
    train_combined_inputs = np.vstack([c1_train_inputs, c2_train_inputs])
    train_combined_activity = np.vstack([c1_train_actual_activity, c2_train_actual_activity])
    test_combined_initial_states = np.vstack([c1_test_initial_states, c2_test_initial_states])
    test_combined_inputs = np.vstack([c1_test_inputs, c2_test_inputs])
    test_combined_activity = np.vstack([c1_test_actual_activity, c2_test_actual_activity])

    # Get Nubmers
    number_of_training_samples = np.shape(train_combined_inputs)[0]
    number_of_test_samples = np.shape(test_combined_inputs)[0]

    # Shuffle These
    train_combined_initial_states, train_combined_inputs, train_combined_activity = jointly_shuffle_lists(train_combined_initial_states, train_combined_inputs, train_combined_activity)
    test_combined_initial_states, test_combined_inputs, test_combined_activity = jointly_shuffle_lists(test_combined_initial_states, test_combined_inputs, test_combined_activity)

    # Convert To Tensors
    train_combined_initial_states = torch.tensor(train_combined_initial_states, dtype=torch.float32, device=device)
    train_combined_inputs = torch.tensor(train_combined_inputs, dtype=torch.float32, device=device)
    train_combined_activity = torch.tensor(train_combined_activity, dtype=torch.float32, device=device)
    test_combined_initial_states = torch.tensor(test_combined_initial_states, dtype=torch.float32, device=device)
    test_combined_inputs = torch.tensor(test_combined_inputs, dtype=torch.float32, device=device)
    test_combined_activity = torch.tensor(test_combined_activity, dtype=torch.float32, device=device)

    return [train_combined_initial_states, train_combined_inputs, train_combined_activity,
            test_combined_initial_states, test_combined_inputs, test_combined_activity,
            number_of_training_samples, number_of_test_samples]

# Cuda Stuff
"""
print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device:{torch.cuda.current_device()}")
print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""

device = "cpu"


# Settings
condition_1 = "visual_1_all_onsets.npy"
condition_2 = "visual_2_all_onsets.npy"

start_window = -14
stop_window = 40
trial_length = stop_window - start_window
criterion = torch.nn.MSELoss()
learning_rate = 0.0001
n_iters = 100
print_steps = 100
previous_loss_window_size = 1000

# Load Neural Data
base_directory = r"C:\Users\matth\Documents\Functional-Connectivity_V2\Parcellated_Delta_F\NXAK7.1B\2021_02_22_Discrimination_Imaging"
activity_matrix = np.load(os.path.join(base_directory, "Cluster_Activity_Matrix.npy"))
print("Delta F Matrix Shape", np.shape(activity_matrix))

# Normalise Activity Matrix
activity_matrix = normalise_activity_matrix(activity_matrix)
#activity_matrix = np.subtract(activity_matrix, 0.5)
#activity_matrix = np.multiply(activity_matrix, 2)

# Remove Background Activity
activity_matrix = activity_matrix[:, 1:]

# Create Output Directory
output_directory = os.path.join(base_directory, "rnn_modelling")
if not os.path.exists(output_directory):
    os.mkdir(output_directory)

# Load Onsets
vis_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_1))
vis_2_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_2))

# Get Downsampled Running and Lick Traces
running_trace, lick_trace = downsample_ai_traces(base_directory, activity_matrix)

# Create RNN Model
number_of_regions = np.shape(activity_matrix)[1]
number_of_inputs = (2 * trial_length) + 2
rnn = custom_rnn(n_inputs=number_of_inputs, n_neurons=number_of_regions, device=device)
optimiser = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=1e-5)

# Create Training Data
condition_1_data = create_training_data(activity_matrix, running_trace, lick_trace, vis_1_onsets, 1, start_window, stop_window)
condition_2_data = create_training_data(activity_matrix, running_trace, lick_trace, vis_2_onsets, 2, start_window, stop_window)

# Split Into Training and Validation Sets
[train_combined_initial_states, train_combined_inputs, train_combined_activity,
 test_combined_initial_states, test_combined_inputs, test_combined_activity,
 number_of_training_samples, number_of_test_samples] = split_data(condition_1_data, condition_2_data, device)

# Training Settings
current_loss = 0
all_training_losses = []
all_validation_losses = []

converged = False
i = 0
while converged == False:

    # Train Network
    loss = train_network(rnn, train_combined_inputs, train_combined_initial_states, train_combined_activity, criterion, optimiser, number_of_training_samples, trial_length, number_of_regions)
    current_loss += loss

    if (i) % print_steps == 0:

        # Get Training Loss
        training_loss = current_loss / print_steps
        all_training_losses.append(training_loss)
        current_loss = 0

        # Get Validation Loss
        validation_loss = get_validation_loss(rnn, test_combined_inputs, test_combined_initial_states, test_combined_activity, number_of_test_samples, trial_length, number_of_regions, criterion, device)
        all_validation_losses.append(validation_loss)

        print("Iteration: ", i, "Training loss:", training_loss.cpu().detach().numpy(), "Validation loss:", validation_loss)

        # Check If Converged
        if len(all_validation_losses) > previous_loss_window_size:
            previous_n_losses = all_validation_losses[-previous_loss_window_size:]

            if validation_loss > np.max(previous_n_losses):
                converged = True
                print("Model Converged :)")

        torch.save(rnn, os.path.join(output_directory, "RNN_Iteration_" + str(i).zfill(6) + ".pt"))

    i += 1


    """ 



        plt.clf() 

        plt.plot(all_training_losses) 

        plt.plot(all_validation_losses) 

        plt.draw() 

        plt.pause(0.1) 

    """

weight_matrix = np.array(rnn.recurrent_weights.weight.detach().numpy())
np.save(os.path.join(output_directory, "RNN_Weight_Matrix.npy"), weight_matrix)
plt.imshow(weight_matrix)

plt.show()
sorted_matrix = sort_matrix(weight_matrix)
plt.imshow(sorted_matrix)

plt.show()

# Predict Data
for trial in range(number_of_training_samples):

    # Set Hidden State
    rnn.hidden_state = train_combined_initial_states[trial]

    # Iterate Through Each Timestep
    output_tensor = torch.zeros(size=(trial_length, number_of_regions), dtype=torch.float32)

    for timepoint in range(trial_length):
        output_tensor[timepoint] = rnn(train_combined_inputs[trial, :, timepoint])

        # Convert Prediction To Numpy Tensor

    predicted_data = output_tensor.detach().numpy()
    real_data = train_combined_activity[trial]
    figure_1 = plt.figure()
    real_axis = figure_1.add_subplot(1, 2, 1)
    predicted_axis = figure_1.add_subplot(1, 2, 2)
    real_axis.imshow(np.transpose(real_data), cmap='jet')
    predicted_axis.imshow(np.transpose(predicted_data), cmap='jet')
    plt.show()