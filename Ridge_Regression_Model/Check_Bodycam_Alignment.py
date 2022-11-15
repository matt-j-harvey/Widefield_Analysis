import random
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec
import cv2
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import Regression_Utils

def get_video_name(base_directory):
    file_list = os.listdir(base_directory)
    for file in file_list:
        if "_cam_1" in file:
            return file




def get_matched_bodycam_motion(base_directory):

    # Get Video Name
    video_name = get_video_name(base_directory)

    # Load Widefield Frame Dict
    widefield_frame_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy"), allow_pickle=True)[()]

    # Open Video File
    video_file = os.path.join(base_directory, video_name)
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create File To Save This
    mousecam_motion_energy_filepath = os.path.join(base_directory, "Matched_Mousecam_Motion_Energy.h5")
    mousecam_motion_energy_file = tables.open_file(mousecam_motion_energy_filepath, mode="w")
    motion_energy_array = mousecam_motion_energy_file.create_earray(mousecam_motion_energy_file.root, "data", tables.UInt8Atom(), shape=(0, 240 * 320))

    count = 0
    for widefield_frame in tqdm(widefield_frame_dict.keys()):
        mousecam_frame = widefield_frame_dict[widefield_frame]

        if mousecam_frame == 0:
            motion_energy = np.zeros(240 * 320)

        else:

            preeceding_frame = mousecam_frame - 1

            # Read Current Frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, mousecam_frame)
            ret, current_frame_data = cap.read()
            # print("Current Frame Min", np.min(current_frame_data), "Frame Max", np.max(current_frame_data), "Dtype", current_frame_data.dtype)

            # Read Preceeding Frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, preeceding_frame)
            ret, preceeding_frame_data = cap.read()

            # Get Motion Energy
            motion_energy = np.subtract(current_frame_data[:, :, 0], preceeding_frame_data[:, :, 0])
            motion_energy = np.abs(motion_energy)

            # Downsize
            motion_energy = resize(motion_energy, (240, 320))

            # Flatten
            motion_energy = np.reshape(motion_energy, 240 * 320)

        # Write To Tables File
        motion_energy_array.append([motion_energy])
        count += 1

        # Flush eArray every 10 Frames
        if count % 10 == 0:
            motion_energy_array.flush()

    mousecam_motion_energy_file.close()


def get_tensor_frames(onset_list, start_window, stop_window, widefield_frame_dict):

    mousecam_frame_tensor = []

    for onset in onset_list:
        trial_frames = []
        trial_start = onset + start_window
        trial_stop = onset + stop_window

        for widefield_frame in range(trial_start, trial_stop):
            mousecam_frame = widefield_frame_dict[widefield_frame]
            trial_frames.append(mousecam_frame)

        mousecam_frame_tensor.append(trial_frames)

    return mousecam_frame_tensor


def get_mousecam_motion_tensor(base_directory, video_name, frames_tensor):

    # Open Video File
    video_file = os.path.join(base_directory, video_name)
    cap = cv2.VideoCapture(video_file)

    mousecam_tensor = []
    for trial in frames_tensor:
        trial_data = []
        for frame in trial:

            # Read Frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame-1)
            ret, preceeding_frame_data = cap.read()

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, frame_data = cap.read()

            frame_data = np.ndarray.astype(frame_data, float)
            preceeding_frame_data = np.ndarray.astype(preceeding_frame_data, float)

            motion_energy = np.subtract(frame_data[:, :, 0], preceeding_frame_data[:, :, 0])
            motion_energy = np.abs(motion_energy)

            #plt.imshow(motion_energy)
            #plt.show()

            motion_energy = np.reshape(motion_energy, (480 * 640))


            trial_data.append(motion_energy)

        mousecam_tensor.append(trial_data)
    mousecam_tensor = np.array(mousecam_tensor)
    return mousecam_tensor

def get_mousecam_tensor(base_directory, video_name, frames_tensor):

    # Open Video File
    video_file = os.path.join(base_directory, video_name)
    cap = cv2.VideoCapture(video_file)

    mousecam_tensor = []
    for trial in frames_tensor:
        trial_data = []
        for frame in trial:

            # Read Frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, frame_data = cap.read()
            trial_data.append(frame_data[:, :, 0])

        mousecam_tensor.append(trial_data)
    mousecam_tensor = np.array(mousecam_tensor)
    return mousecam_tensor


def view_mousecam_tensors(tensor):

    number_of_trials, trial_length, image_height, image_width = np.shape(tensor)

    figure_1 = plt.figure()
    gridspec_1 = GridSpec(nrows=number_of_trials, ncols=trial_length)

    for trial_index in range(number_of_trials):
        for timepoint_index in range(trial_length):
            axis = figure_1.add_subplot(gridspec_1[trial_index, timepoint_index])
            axis.imshow(tensor[trial_index, timepoint_index], vmin=0, vmax=255)
            axis.axis('off')

    plt.show()


def check_bodycam_alignment(base_directory, vis_1_onset_file, vis_2_onset_file, sample_size=5, start_window=-2, stop_window=5):

    # Load Stimuli Onsets
    vis_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", vis_1_onset_file))
    vis_2_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", vis_2_onset_file))

    # Get Sample
    vis_1_sample = random.sample(list(vis_1_onsets), sample_size)
    vis_2_sample = random.sample(list(vis_2_onsets), sample_size)

    # Load Widefield Frame Dict
    widefield_frame_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy"), allow_pickle=True)[()]

    # Get Mousecam Frame Tensors
    vis_1_frame_tensor = get_tensor_frames(vis_1_sample, start_window, stop_window, widefield_frame_dict)
    vis_2_frame_tensor = get_tensor_frames(vis_2_sample, start_window, stop_window, widefield_frame_dict)

    # Get Video Name
    video_name = get_video_name(base_directory)

    # Get Mousecam Tensors
    vis_1_mousecam_tensor = get_mousecam_tensor(base_directory, video_name, vis_1_frame_tensor)
    vis_2_mousecam_tensor = get_mousecam_tensor(base_directory, video_name, vis_2_frame_tensor)


    view_mousecam_tensors(vis_1_mousecam_tensor)
    view_mousecam_tensors(vis_2_mousecam_tensor)


def display_traces(base_directory, vis_1_onset_file):

    # Load Stimuli Onsets
    vis_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", vis_1_onset_file))
    print("Vis 1 onsets", vis_1_onsets)

    # Load Frame Times Dict
    widefield_frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    widefield_frame_times = Regression_Utils.invert_dictionary(widefield_frame_times)

    # Load AI Reocrder File
    ai_matrix = Regression_Utils.load_ai_recorder_file(base_directory)
    stimuli_dictionary = Regression_Utils.create_stimuli_dictionary()

    # Load Traces
    photodiode_trace = ai_matrix[stimuli_dictionary["Photodiode"]]
    vis_1_trace = ai_matrix[stimuli_dictionary["Visual 1"]]
    mousecam_trace = ai_matrix[stimuli_dictionary["Mousecam"]]
    blue_led = ai_matrix[stimuli_dictionary["LED 1"]]

    print("photodiode length", np.shape(photodiode_trace))
    print("vis 1 length", np.shape(vis_1_trace))

    # Get Trial Sample
    start_window = 200
    stop_window = 200

    # Load Mousecam Frame Time Dict
    widefield_to_mousecam_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy"), allow_pickle=True)[()]
    mousecam_frametime_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Mousecam_Frame_Times.npy"), allow_pickle=True)[()]
    mousecam_frametime_dict = Regression_Utils.invert_dictionary(mousecam_frametime_dict)

    for stimuli_onset in vis_1_onsets:
        trial_onset = widefield_frame_times[stimuli_onset]

        closest_mousecam_frame = widefield_to_mousecam_dict[stimuli_onset]
        closest_mousecam_frame_time = mousecam_frametime_dict[closest_mousecam_frame]
        print("Stimuli onset", stimuli_onset)
        print("Trial onset",trial_onset)

        print("Closest mousecam Frame", closest_mousecam_frame)
        print("Closes Mousecam Frame Time", closest_mousecam_frame_time)

        plt.axvline(start_window, c='k', linestyle='--')
        plt.axvline((closest_mousecam_frame_time - trial_onset) + start_window, c='m', linestyle='--')

        trial_photodiode_trace = photodiode_trace[trial_onset-start_window:trial_onset + stop_window]
        trial_vis_1_trace = vis_1_trace[trial_onset-start_window:trial_onset + stop_window]
        trial_mousecam_trace = mousecam_trace[trial_onset-start_window:trial_onset + stop_window]
        trial_blue_led = blue_led[trial_onset-start_window:trial_onset + stop_window]

        plt.plot(trial_photodiode_trace, c='g')
        plt.plot(trial_vis_1_trace, c='gold')
        plt.plot(trial_blue_led, c='b')
        plt.plot(trial_mousecam_trace, c='m')
        plt.show()



def get_activity_tensor(activity_matrix, onset_list, start_window, stop_window):

    activity_tensor = []

    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window
        trial_data = activity_matrix[trial_start:trial_stop]
        activity_tensor.append(trial_data)

    activity_tensor = np.array(activity_tensor)
    return activity_tensor

def reshape_tensor(tensor):
    number_of_trials, trial_length, pixels = np.shape(tensor)
    tensor = np.reshape(tensor, (number_of_trials * trial_length, pixels))
    return tensor


def run_regression(train_x, train_y, test_x, test_y):

    # Create Model
    model = Ridge()

    # Fit Model
    model.fit(X=train_x, y=train_y)

    # Make Prediciton
    y_pred = model.predict(test_x)

    # Score Model
    model_score = r2_score(y_true=test_y, y_pred=y_pred, multioutput='raw_values')

    weights = model.coef_

    return model_score, weights



def view_tensor_pair(motion_tensor, brain_activity_tensor):

    number_of_trials, trial_length, number_of_pixels = np.shape(motion_tensor)
    indicies, image_height, image_width = Regression_Utils.load_tight_mask_downsized()


    for trial_index in range(number_of_trials):

        figure_1 = plt.figure()
        gridspec_1 = GridSpec(nrows=2, ncols=trial_length)

        for timepoint_index in range(trial_length):

            motion_axis = figure_1.add_subplot(gridspec_1[0, timepoint_index])
            brain_axis = figure_1.add_subplot(gridspec_1[1, timepoint_index])

            brain_image = Regression_Utils.create_image_from_data(brain_activity_tensor[trial_index, timepoint_index], indicies, image_height, image_width)
            motion_image = np.reshape(motion_tensor[trial_index, timepoint_index], (480,640))

            motion_axis.imshow(motion_image, vmin=0, vmax=50)
            brain_axis.imshow(brain_image, vmin=0, vmax=40000)


        plt.show()


def visualise_brain_tensor(brain_activity_tensor):

    number_of_trials, trial_length, number_of_pixels = np.shape(brain_activity_tensor)
    indicies, image_height, image_width = Regression_Utils.load_tight_mask_downsized()

    figure_1 = plt.figure()
    gridspec_1 = GridSpec(nrows=number_of_trials, ncols=trial_length)

    for trial_index in range(number_of_trials):
        trail_baseline = brain_activity_tensor[trial_index, 0]

        for timepoint_index in range(trial_length):
            axis = figure_1.add_subplot(gridspec_1[trial_index, timepoint_index])
            brain_image = brain_activity_tensor[trial_index, timepoint_index]
            #brain_image = np.subtract(brain_image, trail_baseline)
            brain_image = Regression_Utils.create_image_from_data(brain_image, indicies, image_height, image_width)
            axis.imshow(brain_image, vmin=0, vmax=40000)
            axis.axis('off')

    plt.show()


def remove_early_onsets(onsets_list, cutoff=3000):

    threshold_onsets_list  = []
    for onset in onsets_list:
        if onset > cutoff:
            threshold_onsets_list.append(onset)

    return threshold_onsets_list



def vis_1_only_regression(base_directory, vis_1_onset_file, sample_size=30, start_window=-10, stop_window=20):

    # Load Stimuli Onsets
    vis_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", vis_1_onset_file))
    vis_1_onsets = remove_early_onsets(vis_1_onsets)
    print("Onsets", len(vis_1_onsets))

    # Get Sample
    vis_1_sample = random.sample(list(vis_1_onsets), sample_size*2)
    train_vis_1_sample = vis_1_sample[0:sample_size]
    test_vis_1_sample = vis_1_sample[sample_size:]

    # Load Widefield Frame Dict
    widefield_frame_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy"), allow_pickle=True)[()]

    # Get Mousecam Frame Tensors
    train_vis_1_frame_tensor = get_tensor_frames(train_vis_1_sample, start_window, stop_window, widefield_frame_dict)
    test_vis_1_frame_tensor = get_tensor_frames(test_vis_1_sample, start_window, stop_window, widefield_frame_dict)

    # Get Mousecam Tensors
    video_name = get_video_name(base_directory)
    train_vis_1_mousecam_tensor = get_mousecam_motion_tensor(base_directory, video_name, train_vis_1_frame_tensor)
    test_vis_1_mousecam_tensor = get_mousecam_motion_tensor(base_directory, video_name, test_vis_1_frame_tensor)

    # Load Brain Activity
    activity_matrix = np.load(os.path.join(base_directory, "Downsampled_Aligned_Data.npy"))

    #plt.imshow(np.transpose(activity_matrix), cmap='jet')
    #plt.show()

    train_activity_tensor = get_activity_tensor(activity_matrix, train_vis_1_sample, start_window, stop_window)
    test_activity_tensor = get_activity_tensor(activity_matrix, test_vis_1_sample, start_window, stop_window)

    #visualise_brain_tensor(train_activity_tensor)
    #visualise_brain_tensor(test_activity_tensor)
    #view_tensor_pair(test_vis_1_mousecam_tensor, test_activity_tensor)

    # Reshape Tensors
    train_activity_tensor = reshape_tensor(train_activity_tensor)
    train_vis_1_mousecam_tensor = reshape_tensor(train_vis_1_mousecam_tensor)
    test_activity_tensor = reshape_tensor(test_activity_tensor)
    test_vis_1_mousecam_tensor = reshape_tensor(test_vis_1_mousecam_tensor)

    # Run Regression
    model_score, weights = run_regression(train_x=train_vis_1_mousecam_tensor,
                                          train_y=train_activity_tensor,
                                          test_x=test_vis_1_mousecam_tensor,
                                          test_y=test_activity_tensor)

    # View R2 Map
    indicies, image_height, image_width = Regression_Utils.load_tight_mask_downsized()
    r2_map = Regression_Utils.create_image_from_data(model_score, indicies, image_height, image_width)

    plt.imshow(r2_map, cmap='jet')
    plt.show()




vis_1_onset_file = "visual_context_stable_vis_1_onsets.npy"
vis_2_onset_file = "visual_context_stable_vis_2_onsets.npy"
base_directory = r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging"
vis_1_only_regression(base_directory, vis_2_onset_file)
#display_traces(base_directory, vis_1_onset_file)
#check_bodycam_alignment(base_directory, vis_1_onset_file, vis_2_onset_file)