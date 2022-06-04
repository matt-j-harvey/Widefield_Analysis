import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
import tables
import random
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist, cdist

import Custom_RNN_Model
import Load_RNN_Data

# Cuda Stuff

print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device:{torch.cuda.current_device()}")
print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



base_directory = r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_24_Discrimination_Imaging"

# Settings
condition_1 = "visual_1_all_onsets.npy"
condition_2 = "visual_2_all_onsets.npy"
start_window = -14
stop_window = 40

# Load Data
[train_combined_initial_states,
 train_combined_inputs,
 train_combined_activity,

 test_combined_initial_states,
 test_combined_inputs,
 test_combined_activity,

 number_of_training_samples,
 number_of_test_samples,
 number_of_regions,
 number_of_inputs] = Load_RNN_Data.load_rnn_data(base_directory, condition_1, condition_2, start_window, stop_window)

# Create RNN Model
rnn_model = Custom_RNN_Model.custom_rnn(number_of_inputs, number_of_regions, device)

# Move Everything To GPU
train_combined_initial_states = train_combined_initial_states.to(device)
train_combined_inputs = train_combined_inputs.to(device)
train_combined_activity = train_combined_activity.to(device)
rnn_model.to(device)

# Training Parameters
trial_length = stop_window - start_window
criterion = torch.nn.MSELoss()
learning_rate = 0.0001
n_iters = 100
print_steps = 100
previous_loss_window_size = 1000
optimiser = torch.optim.Adam(rnn_model.parameters(), lr=learning_rate)


converged = False
i = 0

while converged == False:
    print(i)

    for trial in range(number_of_training_samples):

        # Set Hidden State
        rnn_model.hidden_state = train_combined_initial_states[trial]

        # Iterate Through Each Timepoint
        output_tensor = torch.zeros(size=(trial_length, number_of_regions), dtype=torch.float32, device=device)

        for timepoint in range(trial_length):
             output_tensor[timepoint] = rnn_model(train_combined_inputs[trial, :, timepoint])

        optimiser.zero_grad()
        loss = criterion(output_tensor, train_combined_activity[trial])
        loss.backward()
        optimiser.step()

    i += 1