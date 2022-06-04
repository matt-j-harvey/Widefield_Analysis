import matplotlib.pyplot as plt
import numpy as np
import os
import math
import cv2
from sklearn.decomposition import TruncatedSVD, PCA, IncrementalPCA

def factor_number(number_to_factor):
    # Returns The Factors Of A Number

    factor_list = []

    for potential_factor in range(1, number_to_factor):
        if number_to_factor % potential_factor == 0:
            factor_pair = [potential_factor, int(number_to_factor/potential_factor)]
            factor_list.append(factor_pair)

    return factor_list


def get_best_grid(number_of_items):
    # Given A Number Whats The Best Way To Make A Sqaure Grid With That Many Elements

    factors = factor_number(number_of_items)
    factor_difference_list = []

    #Get Difference Between All Factors
    for factor_pair in factors:
        factor_difference = abs(factor_pair[0] - factor_pair[1])
        factor_difference_list.append(factor_difference)

    #Select Smallest Factor difference
    smallest_difference = np.min(factor_difference_list)
    best_pair = factor_difference_list.index(smallest_difference)

    return factors[best_pair]




def create_mousecam_tensor(base_directory, video_file, onsets, start_window, stop_window):

    # Get Number Of Trials
    number_of_trials = len(onsets)

    # Get The Selected Mousecam Frames For Each Trial
    selected_frames = []
    for trial_index in range(number_of_trials):
        trial_start = onsets[trial_index] + start_window
        trial_stop = onsets[trial_index] + stop_window
        trial_frames = list(range(trial_start, trial_stop))
        selected_frames.append(trial_frames)
    selected_frames = np.array(selected_frames)

    full_video_filepath = os.path.join(base_directory, video_file)
    mousecam_tensor = load_video_as_numpy_array(full_video_filepath, selected_frames)


    return mousecam_tensor


def perform_svd_on_video(video_array, number_of_components=100):

    print("Performing SVD")

    # Assumes Video Data is a 3D array with the structure: Trials, Timepoints, height, width
    number_of_trials = np.shape(video_array)[0]
    trial_length     = np.shape(video_array)[1]
    video_height     = np.shape(video_array)[2]
    video_width      = np.shape(video_array)[3]
    number_of_frames = number_of_trials * trial_length

    # Flatten Video
    video_array = np.reshape(video_array, (number_of_frames, video_height * video_width))
    print("Flat Video Array Shaoe", np.shape(video_array))

    # Perform PCA
    pca_model = TruncatedSVD(n_components=number_of_components)
    pca_model.fit(video)
    prinicpal_components = pca_model.components_
    transformed_data = pca_model.transform(video)

    # Put Data Back Into Original Shape
    transformed_data = np.reshape(transformed_data, (number_of_trials, trial_length, number_of_components))

    return transformed_data, prinicpal_components


def load_video_as_numpy_array(video_file, selected_mousecam_frames):

    # Open Video File
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get Trial Details
    number_of_trials = len(selected_mousecam_frames)
    trial_length = len(selected_mousecam_frames[0])
    extracted_frames = np.zeros((number_of_trials, trial_length, frameHeight, frameWidth, 3))

    # Extract Selected Frames
    frame_index = 0
    ret = True
    while (frame_index < frameCount and ret):
        ret, frame = cap.read()
        frame_index += 1

        # See If This Is a Frame We Want
        for trial in range(number_of_trials):
            for timepoint in range(trial_length):
                if frame_index == selected_mousecam_frames[trial][timepoint]:
                    extracted_frames[trial][timepoint] = frame

    cap.release()
    extracted_frames = extracted_frames[:,:,:,:,0]

    return extracted_frames


def get_motion_energy(mousecam_tensor, visualise=False):

    # Get Difference Along Time Axis
    mousecam_tensor = np.diff(mousecam_tensor, axis=0)
    mousecam_tensor = np.nan_to_num(mousecam_tensor)
    #print("Difference Tensor Shape", np.shape(mousecam_tensor))
    #print("Difference Tensor Sizee", mousecam_tensor.nbytes)

    # Get Absolute Value of Difference Tensor
    np.abs(mousecam_tensor, out=mousecam_tensor)
    #print("Difference Tensor Sizee", mousecam_tensor.nbytes)

    """
    # Sanity Check
    if visualise == True:
        plt.ion()
        figure_1 = plt.figure()
        for trial in range(20):
            for timepoint in range(49):
                original_frame = mousecam_tensor[trial, timepoint]
                next_frame = mousecam_tensor[trial, timepoint+1]
                motion_energy = difference_tensor[trial, timepoint]

                original_axis = figure_1.add_subplot(1,3,1)
                next_axis = figure_1.add_subplot(1, 3, 2)
                energy_axis = figure_1.add_subplot(1, 3, 3)

                original_axis.set_title("Original Frame")
                next_axis.set_title("Next Frame")
                energy_axis.set_title("Motion Energy")

                original_axis.axis('off')
                next_axis.axis('off')
                energy_axis.axis('off')


                original_axis.imshow(original_frame, cmap='binary_r')
                next_axis.imshow(next_frame, cmap='binary_r')
                energy_axis.imshow(motion_energy, cmap='jet')


                plt.title(str(timepoint))
                plt.draw()
                plt.pause(0.1)
                plt.clf()

        plt.ioff()
    """
    return mousecam_tensor


def load_video_as_numpy_array(video_file, selected_mousecam_frames):

    # Open Video File
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get Trial Details
    number_of_trials = len(selected_mousecam_frames)
    trial_length = len(selected_mousecam_frames[0])
    extracted_frames = np.zeros((number_of_trials, trial_length, frameHeight, frameWidth, 3))

    # Extract Selected Frames
    frame_index = 0
    ret = True
    while (frame_index < frameCount and ret):
        ret, frame = cap.read()
        frame_index += 1

        # See If This Is a Frame We Want
        for trial in range(number_of_trials):
            for timepoint in range(trial_length):
                if frame_index == selected_mousecam_frames[trial][timepoint]:
                    extracted_frames[trial][timepoint] = frame

    cap.release()
    extracted_frames = extracted_frames[:,:,:,:,0]

    return extracted_frames


def get_video_details(video_file):
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return frameCount, frameHeight, frameWidth

def visualise_mousecam_components(components, base_directory, video_file):

    # Load Video Data
    frame_count, video_height, video_width = get_video_details(os.path.join(base_directory, video_file))

    number_of_components = np.shape(components)[0]
    [rows, columns] = get_best_grid(number_of_components)

    axis_list = []
    figure_1 = plt.figure()
    for component_index in range(number_of_components):
        component_data = components[component_index]
        component_data = np.abs(component_data)
        component_data = np.reshape(component_data, (video_height, video_width))

        axis_list.append(figure_1.add_subplot(rows, columns, component_index + 1))
        axis_list[component_index].imshow(component_data, cmap='jet')
        axis_list[component_index].set_title(str(component_index))
        axis_list[component_index].axis('off')

    plt.show()


def load_onsets(base_directory, onsets_file_list):

    # Load Onsets
    onsets = []
    for onsets_file in onsets_file_list:
        print(onsets_file_list)
        print(onsets_file)
        onsets_file_contents = np.load(os.path.join(base_directory, "Stimuli_Onsets", onsets_file + "_onsets.npy"))
        for onset in onsets_file_contents:
            onsets.append(onset)
    print("Number_of_trails: ", len(onsets))

    return onsets



def convert_widefield_onsets_into_mousecam_onsets(base_directory, widefield_onsets):

    # Load Widefield to Mousecam Dictionary
    widefield_to_mousecam_dictionary = np.load(os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy"), allow_pickle=True)[()]

    mousecam_onset_list = []
    for onset in widefield_onsets:
        mousecam_onset = widefield_to_mousecam_dictionary[onset]
        mousecam_onset_list.append(mousecam_onset)

    return mousecam_onset_list


def get_bodycam_tensor(base_directory, video_file, onsets_file_list, start_window, stop_window, number_of_components=20):

    # This code will extract the bodycam video around a number of trials and perform SVD on this data
    # Onsets File - should be a numpy array which contains the bodycam frame index at the start of each trial
    # Start Window - how many bodycam frames to include from before trial start
    # Stop Window - how many bodycam frames to include from after trial start
    # Video File - full file path to video file

    # Load Widefield Frame Indexes of Trial Starts
    onsets = load_onsets(base_directory, onsets_file_list)

    # Convert Widefield Frames Into Mousecam Frames
    onsets = convert_widefield_onsets_into_mousecam_onsets(base_directory, onsets)

    # Load Video Data Into A Tensor Of Shape (N_trials , Trial Length, Image Height, Image Width)
    mousecam_tensor = create_mousecam_tensor(base_directory, video_file, onsets, start_window, stop_window)

    #Get "Motion Energy" - (Absolute Value Of THe Difference Between Subsequent Frames)
    mousecam_tensor = get_motion_energy(mousecam_tensor, visualise=True)

    # Perform SVD on Video To Decompose It Into A number of components and loadings of these components over time
    transformed_data, components = perform_svd_on_video(mousecam_tensor, number_of_components=number_of_components)

    # View These Mousecam Components For A Sanity Check
    visualise_mousecam_components(components, base_directory, video_file)

    return transformed_data, components



def incremental_mousecam_pca(base_directory, video_file, onsets, start_window, stop_window):

    # Create Model
    model = IncrementalPCA(n_components=20)

    # Load Video File
    cap = cv2.VideoCapture(os.path.join(base_directory, video_file))
    print("video file", video_file)

    # Iterate Through Each Trial
    trial_count = 0
    for onset in onsets:

        print("Trial: ", trial_count)

        # Load Video Data For Trial
        trial_data = []
        trial_start = onset + start_window
        trial_stop = onset + stop_window
        for frame_index in range(trial_start, trial_stop):
            cap.set(1, frame_index)
            ret, frame = cap.read()
            trial_data.append(frame[:, :, 0])
        trial_data = np.array(trial_data)
        #print("Trial Data Shape", np.shape(trial_data))

        # Get Motion Energy
        trial_motion_energy = get_motion_energy(trial_data)
        #print("Trial Motion Energy Shape", np.shape(trial_motion_energy))

        # Reshape Motion Energy
        trial_length = np.shape(trial_motion_energy)[0]
        video_height = np.shape(trial_motion_energy)[1]
        video_width = np.shape(trial_motion_energy)[2]
        trial_motion_energy = np.reshape(trial_motion_energy, (trial_length, video_height * video_width))
        #print("Trial Motion Energy Shape", np.shape(trial_motion_energy))

        # Train Model
        model.partial_fit(trial_motion_energy)
        #print("Finished Partial Fit")

        trial_count += 1


    # Iterate Through Each Onset To Transform Data
    transformed_data_tensor = []
    for onset in onsets:

        print("Trial: ", trial_count)

        # Load Video Data For Trial
        trial_data = []
        trial_start = onset + start_window
        trial_stop = onset + stop_window
        for frame_index in range(trial_start, trial_stop):
            cap.set(1, frame_index)
            ret, frame = cap.read()
            trial_data.append(frame[:, :, 0])
        trial_data = np.array(trial_data)
        # print("Trial Data Shape", np.shape(trial_data))

        # Get Motion Energy
        trial_motion_energy = get_motion_energy(trial_data)
        # print("Trial Motion Energy Shape", np.shape(trial_motion_energy))

        # Reshape Motion Energy
        trial_length = np.shape(trial_motion_energy)[0]
        video_height = np.shape(trial_motion_energy)[1]
        video_width = np.shape(trial_motion_energy)[2]
        trial_motion_energy = np.reshape(trial_motion_energy, (trial_length, video_height * video_width))
        # print("Trial Motion Energy Shape", np.shape(trial_motion_energy))
        # Trannsform Data

        transformed_data = model.transform(trial_motion_energy)
        print("Transforrrmed Data Shape", np.shape(transformed_data))
        # Add To Tensor
        transformed_data_tensor.append(transformed_data)


    transformed_data_tensor = np.array(transformed_data_tensor)

    # Get Components
    components = model.components_


    return transformed_data_tensor, components




def load_onsets(base_directory, onsets_group_list):

    onsets_list = []
    for group in onsets_group_list:
        for file in group:
            onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", file))
            onsets_list.append(onsets)

    return onsets_list


def get_mousecam_trial_windows(base_directory, onsets_list, start_window, stop_window):

    # Get Trial Start Widefield Frame Onsets
    trial_start_widefield_onsets = []
    for onset in onsets_list:
        trial_start_widefield_onsets.append(onset + start_window)

    # Get Trial Stop Widefield Frame Onsets
    trial_stop_widefield_onsets = []
    for onset in onsets_list:
        trial_stop_widefield_onsets.append(onset + stop_window)

    # Convert These To Mousecam Frames
    mousecam_onsets_list = convert_widefield_onsets_into_mousecam_onsets(base_directory, onsets_list)
    mousecam_trial_start_onsets = convert_widefield_onsets_into_mousecam_onsets(base_directory, trial_start_widefield_onsets)
    mousecam_trial_stop_onsets = convert_widefield_onsets_into_mousecam_onsets(base_directory, trial_stop_widefield_onsets)

    # Get Differences
    print("Onsets", mousecam_onsets_list)
    print("Trial Start Onsets", mousecam_trial_start_onsets)

    mousecam_start_window_list = np.subtract(mousecam_trial_start_onsets, mousecam_onsets_list)
    mousecam_stop_window_list = np.subtract(mousecam_trial_stop_onsets, mousecam_onsets_list)

    print("Start window list", mousecam_start_window_list)
    print("Stop window list", mousecam_stop_window_list)

    mousecam_start_window = int(np.mean(mousecam_start_window_list))
    mousecam_stop_window = int(np.mean(mousecam_stop_window_list))

    print("Start window", mousecam_start_window)
    print("Stop window", mousecam_stop_window)

    return mousecam_start_window, mousecam_stop_window


def get_bodycam_tensor_multiple_conditions(base_directory, video_file, onsets_group_list, start_window, stop_window, number_of_components=20):

    # This code will extract the bodycam video around a number of trials and perform SVD on this data
    # Onsets File - should be a numpy array which contains the bodycam frame index at the start of each trial
    # Start Window - how many bodycam frames to include from before trial start
    # Stop Window - how many bodycam frames to include from after trial start
    # Video File - full file path to video file

    # Load Widefield Frame Indexes of Trial Starts
    onsets_list = []
    for onset_group in onsets_group_list:
        for onset in onset_group:
            onsets_list.append(onset)

    # Convert Widefield Frames Into Mousecam Frames
    mousecam_onsets = convert_widefield_onsets_into_mousecam_onsets(base_directory, onsets_list)

    # Convert Widefield Start and Stop Windows Into Mousecam Windows
    #mousecam_start_window, mousecam_stop_window = get_mousecam_trial_windows(base_directory, onsets_list, start_window, stop_window)

    # Get Motion Energy PCA
    transformed_data, components = incremental_mousecam_pca(base_directory, video_file, mousecam_onsets, start_window-1, stop_window)

    # View These Mousecam Components For A Sanity Check
    #visualise_mousecam_components(components, base_directory, video_file)


    # Split Combined Tensor

    number_of_condition_1_trials = len(onsets_group_list[0])
    number_of_condition_2_trials = len(onsets_group_list[1])
    condition_1_bodycam_tensor = transformed_data[0:number_of_condition_1_trials]
    condition_2_bodycam_tensor = transformed_data[number_of_condition_1_trials:]

    print("Number of condition 1 trials", number_of_condition_1_trials)
    print("Number of cobdition 2 trials", number_of_condition_2_trials)
    print("Condition 1 bodycam tensor", np.shape(condition_1_bodycam_tensor))
    print("Condition 2 bodycam tensor", np.shape(condition_2_bodycam_tensor))

    return condition_1_bodycam_tensor, condition_2_bodycam_tensor, components

"""
base_directory = "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_02_Transition_Imaging"
video_file = "NXAK4.1B_2021-04-02-16-11-08_cam_1.mp4"
start_window = -10
stop_window = 20
onsets_group_list = [["visual_context_stable_vis_1_onsets.npy"], ["visual_context_stable_vis_2_onsets.npy"]]
onsets_group_list = load_onsets(base_directory, onsets_group_list)
get_bodycam_tensor_multiple_conditions(base_directory, video_file, onsets_group_list, start_window, stop_window, number_of_components=20)
"""