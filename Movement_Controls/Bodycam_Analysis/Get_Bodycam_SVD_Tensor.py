import matplotlib.pyplot as plt
import numpy as np
import os
import math
import cv2
from sklearn.decomposition import TruncatedSVD, PCA

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

    # Assumes Video Data is a 3D array with the structure: Trials, Timepoints, height, width
    number_of_trials = np.shape(video_array)[0]
    trial_length     = np.shape(video_array)[1]
    video_height     = np.shape(video_array)[2]
    video_width      = np.shape(video_array)[3]
    number_of_frames = number_of_trials * trial_length

    # Flatten Video
    video = np.reshape(video_array, (number_of_frames, video_height * video_width))

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


def get_motion_energy(mousecam_tensor, visualise=True):

    # Get Difference Along Time Axis
    difference_tensor = np.diff(mousecam_tensor, axis=1)
    print("Difference Tensor Shape", np.shape(difference_tensor))

    # Get Absolute Value of Difference Tensor
    difference_tensor = np.abs(difference_tensor)

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

    return difference_tensor


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
        onsets_file_contents = np.load(os.path.join(base_directory, "Stimuli_Onsets", onsets_file))
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
    mousecam_tensor = get_motion_energy(mousecam_tensor, visualise=False)

    # Perform SVD on Video To Decompose It Into A number of components and loadings of these components over time
    transformed_data, components = perform_svd_on_video(mousecam_tensor, number_of_components=number_of_components)

    # View These Mousecam Components For A Sanity Check
    visualise_mousecam_components(components, base_directory, video_file)

    return transformed_data, components



base_directory = r"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging"
start_window = -10
stop_window = 40
onset_files = [["visual_context_stable_vis_2_onsets.npy"], ["odour_context_stable_vis_2_onsets.npy"]]
tensor_names = ["Vis_2_Stable_Visual", "Vis_2_Stable_Odour"]
video_file = "NXAK14.1A_2021-06-17-14-30-28_cam_1.mp4"
get_bodycam_tensor(base_directory, video_file, onset_files[0], start_window, stop_window, number_of_components=20)