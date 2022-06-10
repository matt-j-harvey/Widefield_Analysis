import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
import tables
import random
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist, cdist

from sklearn.linear_model import LinearRegression, Ridge

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


def forceAspect(ax, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)



def get_maximum_stimuli_length(trace, threshold):

    stimuli_lengths = []
    number_of_timepoints = len(trace)

    # Get Initial State
    if trace[0] >= threshold:
        current_state = 1
        stimuli_length = 1
    else:
        current_state = 0
        stimuli_length = 0


    for timepoint in range(1, number_of_timepoints):

        # If We are Presenting a Stimulus
        if trace[timepoint] >= threshold:

            # If We are Already In A Stimulus - Simple Increase Duration Of Current Stimuli
            if current_state == 1:
                stimuli_length += 1

            # If We Are Not Currently In A Stimulus - Start A New One!
            elif current_state == 0:
                current_state = 1
                stimuli_length = 1

        # If There Is No Stimulus
        else:

            # If We Are Not Currently In A Stimulus, Do Nothing
            if current_state == 0:
                pass

            # Else a Stimulus Has Just Ended, Append It To THe List, Set Out Current State to 0, and Stimulus Length To 0
            elif current_state == 1:
                stimuli_lengths.append(stimuli_length)
                current_state = 0
                stimuli_length = 0

    max_stimuli_length = np.max(stimuli_lengths)
    return max_stimuli_length


def create_stimuli_design_matrix(trace, stimuli_design_matrix, step_threshold=0.5):

    # Get Number Of Timepoints
    number_of_timepoints = len(trace)

    # Get Initial State
    if trace[0] >= step_threshold:
        current_state = 1
        stimuli_length = 1
        stimuli_design_matrix[0, stimuli_length - 1] = 1
    else:
        current_state = 0
        stimuli_length = 0

    # Iterate Through All Other Timepoints
    for timepoint in range(1, number_of_timepoints):

        # If We are Presenting a Stimulus
        if trace[timepoint] >= step_threshold:

            # If We are Already In A Stimulus - Simple Increase Duration Of Current Stimuli
            if current_state == 1:
                stimuli_length += 1

            # If We Are Not Currently In A Stimulus - Start A New One!
            elif current_state == 0:
                current_state = 1
                stimuli_length = 1

            stimuli_design_matrix[timepoint, stimuli_length - 1] = 1

        # If There Is No Stimulus
        else:

            # If We Are Not Currently In A Stimulus, Do Nothing
            if current_state == 0:
                pass

            # Else a Stimulus Has Just Ended, Append It To THe List, Set Out Current State to 0, and Stimulus Length To 0
            elif current_state == 1:
                current_state = 0
                stimuli_length = 0



    """
    trace = np.reshape(trace, (number_of_timepoints, 1))
    combined_design_matrix = np.hstack([trace, stimuli_design_matrix]) 
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.imshow(np.transpose(combined_design_matrix[0:1000]))
    forceAspect(axis_1, 3)
    plt.show()
    """

    return stimuli_design_matrix


def create_regressors(downsampled_ai_data, step_threshold=0.5):

    # Get Vis 1 and Vis 2 Traces
    stimuli_dictionary = create_stimuli_dictionary()
    vis_1_trace = downsampled_ai_data[stimuli_dictionary["Visual 1"]]
    vis_2_trace = downsampled_ai_data[stimuli_dictionary["Visual 2"]]

    # Get Maximum Stimuli Length
    vis_1_max_length = get_maximum_stimuli_length(vis_1_trace, step_threshold)
    vis_2_max_length = get_maximum_stimuli_length(vis_2_trace, step_threshold)

    # Create Design Matricies
    number_of_timepoints = len(vis_1_trace)
    vis_1_design_matrix_empty = np.zeros((number_of_timepoints, vis_1_max_length))
    vis_2_design_matrix_empty = np.zeros((number_of_timepoints, vis_2_max_length))

    vis_1_design_matrix = create_stimuli_design_matrix(vis_1_trace, vis_1_design_matrix_empty, step_threshold)
    vis_2_design_matrix = create_stimuli_design_matrix(vis_2_trace, vis_2_design_matrix_empty, step_threshold)

    return vis_1_design_matrix, vis_2_design_matrix


def view_cluster_vector(vector):

    clusters = np.load("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Final_Consensus_Clusters.npy")




def get_percentage_variance_explained(prediction, data):

    # Get Variance Of Data
    print("Data Shape", np.shape(data))
    data_variance = np.var(data, axis=0)
    print("Data Variance", np.shape(data_variance))

    # Get Error
    error = np.subtract(data, prediction)
    print("Error Shape", np.shape(error))
    error_variance = np.var(error, axis=0)
    print("Error Variance", error_variance)

    fraction_variance_unexplained = error_variance / data_variance
    fraction_of_variance_explained = 1 - fraction_variance_unexplained
    percentage_of_variance_explained = fraction_of_variance_explained * 100

    print(fraction_of_variance_explained, "% of Variance Explained ")



    return percentage_of_variance_explained


def subtract_mean_activity_matrix(activity_matrix):

    mean_vector = np.mean(activity_matrix, axis=0)

    activity_matrix = np.tranpose(activity_matrix)
    activity_matrix = np.subtract(activity_matrix, mean_vector)
    activity_matrix = np.transpose(activity_matrix)

    return activity_matrix


def normalise_activity_matrix(activity_matrix):

    # Subtract Min
    activity_matrix = np.subtract(activity_matrix, np.min(activity_matrix))

    # Divide By Max
    activity_matrix = np.divide(activity_matrix, np.max(activity_matrix))

    return activity_matrix


def get_unexplained_activity(base_directory, visualise=1):

    # Load Neural Data
    activity_matrix = np.load(os.path.join(base_directory, "Cluster_Activity_Matrix.npy"))
    activity_matrix = np.nan_to_num(activity_matrix)
    print("Activity Matrix Shape", np.shape(activity_matrix))

    # Get Behaviour Data
    downsampled_ai_data = np.load(os.path.join(base_directory, "Downsampled_AI_Matrix.npy"))
    print("downsampled_ai_data Shape", np.shape(downsampled_ai_data))

    # Remove First 1500 Frames
    section_to_remove = 1500
    activity_matrix = activity_matrix[section_to_remove:]
    downsampled_ai_data = downsampled_ai_data[:, section_to_remove:]

    # Remove Background Activity
    activity_matrix = activity_matrix[:, 1:]

    # Normalise Activity Matrix
    activity_matrix = normalise_activity_matrix(activity_matrix)

    # Create Design Matricies
    #vis_1_design_matrix, vis_2_design_matrix = create_regressors(downsampled_ai_data)

    # Get Lick and Running Trace
    stimuli_dictionary = create_stimuli_dictionary()
    lick_trace = downsampled_ai_data[stimuli_dictionary["Lick"]]
    running_trace = downsampled_ai_data[stimuli_dictionary["Running"]]

    lick_trace = np.reshape(lick_trace, (len(lick_trace), 1))
    running_trace = np.reshape(running_trace, (len(running_trace), 1))

    # Combine Into Design Matrix
    design_matrix = np.hstack([lick_trace, running_trace])
    print("Design Matrix Shape", np.shape(design_matrix))

    plt.plot(np.mean(activity_matrix, axis=1))
    plt.plot(running_trace)
    plt.show()

    # Get Some Test Data
    test_proportion = 0.2
    number_of_timepoints = np.shape(activity_matrix)[0]
    test_timepoints = int(number_of_timepoints * test_proportion)
    train_timepoints = number_of_timepoints - test_timepoints
    print("Test Timepoints", test_timepoints)
    print("Train Timepoints", train_timepoints)

    train_data = activity_matrix[0:train_timepoints]
    train_predictiors = design_matrix[0:train_timepoints]

    test_data = activity_matrix[train_timepoints:]
    test_predictors = design_matrix[train_timepoints:]

    # Perform Regression
    model = Ridge()
    model.fit(X=train_predictiors, y=train_data)

    # Get Score
    prediction = model.predict(test_predictors)

    get_percentage_variance_explained(prediction, test_data)
    """
    # Predict Data and Get Residual
    predicted_data = model.predict(design_matrix)
    residual_data = np.subtract(activity_matrix, predicted_data)

    if visualise == 1:
        figure_1 = plt.figure()
        real_axis = figure_1.add_subplot(3,1,1)
        predicted_axis = figure_1.add_subplot(3, 1, 2)
        residual_axis = figure_1.add_subplot(3, 1, 3)

        samplesize = 50000
        real_axis.imshow(np.transpose(activity_matrix[0:samplesize]))
        predicted_axis.imshow(np.transpose(predicted_data[0:samplesize]))
        residual_axis.imshow(np.transpose(residual_data[0:samplesize]))

        aspect=4
        forceAspect(real_axis, aspect=aspect)
        forceAspect(predicted_axis, aspect=aspect)
        forceAspect(residual_axis, aspect=aspect)

        plt.show()

    # Save Data
    save_directory = os.path.join(base_directory, "Movement_Correction")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    np.save(os.path.join(save_directory, "Unexplained_Activity.npy"), residual_data)
    """

session_list = [
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    #"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging",
]




number_of_sessions  = len(session_list)
for session_index in range(number_of_sessions):
    print("Session: ", session_index, " of ", number_of_sessions)

    session = session_list[session_index]
    get_unexplained_activity(session, visualise=False)

