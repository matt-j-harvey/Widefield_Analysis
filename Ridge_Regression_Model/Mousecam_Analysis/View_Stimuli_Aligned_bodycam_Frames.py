import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec
import cv2
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


def get_video_name(base_directory):
    file_list = os.listdir(base_directory)
    for file in file_list:
        if "_cam_1.mp4" in file:
            return file


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


def get_mousecam_tensor(base_directory, video_name, frames_tensor):

    # Open Video File
    video_file = os.path.join(base_directory, video_name)
    cap = cv2.VideoCapture(video_file)
    print("Video name", video_name)
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


def view_mousecam_tensors(base_directory, tensor, stim_name):

    number_of_trials, trial_length, image_height, image_width = np.shape(tensor)


    figure_1 = plt.figure(figsize=(trial_length, number_of_trials*0.9))
    gridspec_1 = GridSpec(nrows=number_of_trials, ncols=trial_length)

    for trial_index in range(number_of_trials):
        for timepoint_index in range(trial_length):

            axis = figure_1.add_subplot(gridspec_1[trial_index, timepoint_index])
            axis.imshow(tensor[trial_index, timepoint_index], vmin=0, vmax=255)
            axis.axis('off')

            if timepoint_index == 0:
                axis.set_title(str(trial_index))

    save_directory = os.path.join(base_directory, "Mousecam_Analysis", str(stim_name) + "Aligned_Mousecam_Frames.png")
    plt.savefig(save_directory)
    plt.close()
    #plt.show()


def exclude_stimuli_prior_to_cutoff(stimuli_onsets, early_cutoff=3000):
    onsets_list = []
    for onset in stimuli_onsets:
        if onset >early_cutoff:
            onsets_list.append(onset)
    return onsets_list

def check_bodycam_alignment(base_directory, vis_1_onset_file, vis_2_onset_file, start_window=-3, stop_window=6):

    # Load Stimuli Onsets
    vis_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", vis_1_onset_file))
    vis_2_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", vis_2_onset_file))

    # Exclude Stimuli Prior To Cutoff
    vis_1_onsets = exclude_stimuli_prior_to_cutoff(vis_1_onsets)
    vis_2_onsets = exclude_stimuli_prior_to_cutoff(vis_2_onsets)

    # Load Widefield Frame Dict
    widefield_frame_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy"), allow_pickle=True)[()]

    # Get Mousecam Frame Times
    vis_1_frame_tensor = get_tensor_frames(vis_1_onsets, start_window, stop_window, widefield_frame_dict)
    vis_2_frame_tensor = get_tensor_frames(vis_2_onsets, start_window, stop_window, widefield_frame_dict)

    # Get Video Name
    video_name = get_video_name(base_directory)

    # Get Mousecam Tensors
    vis_1_mousecam_tensor = get_mousecam_tensor(base_directory, video_name, vis_1_frame_tensor)
    vis_2_mousecam_tensor = get_mousecam_tensor(base_directory, video_name, vis_2_frame_tensor)

    vis_1_stim_name = vis_1_onset_file.replace(".npy", "")
    vis_2_stim_name = vis_2_onset_file.replace(".npy", "")
    view_mousecam_tensors(base_directory, vis_1_mousecam_tensor, vis_1_stim_name + "_")
    view_mousecam_tensors(base_directory, vis_2_mousecam_tensor, vis_2_stim_name + "_")




session_list = [

    r"NRXN78.1D/2020_11_29_Switching_Imaging",
    r"NRXN78.1D/2020_12_07_Switching_Imaging",
    r"NRXN78.1D/2020_12_05_Switching_Imaging",

    r"NRXN78.1A/2020_11_28_Switching_Imaging",
    r"NRXN78.1A/2020_12_05_Switching_Imaging",
    r"NRXN78.1A/2020_12_09_Switching_Imaging",

    r"NXAK14.1A/2021_05_21_Switching_Imaging",
    r"NXAK14.1A/2021_05_23_Switching_Imaging",
    r"NXAK14.1A/2021_06_11_Switching_Imaging",
    r"NXAK14.1A/2021_06_13_Transition_Imaging",
    r"NXAK14.1A/2021_06_15_Transition_Imaging",
    r"NXAK14.1A/2021_06_17_Transition_Imaging",

    r"NXAK22.1A/2021_10_14_Switching_Imaging",
    r"NXAK22.1A/2021_10_20_Switching_Imaging",
    r"NXAK22.1A/2021_10_22_Switching_Imaging",
    r"NXAK22.1A/2021_10_29_Transition_Imaging",
    r"NXAK22.1A/2021_11_03_Transition_Imaging",
    r"NXAK22.1A/2021_11_05_Transition_Imaging",

    r"NXAK4.1B/2021_03_02_Switching_Imaging",
    r"NXAK4.1B/2021_03_04_Switching_Imaging",
    r"NXAK4.1B/2021_03_06_Switching_Imaging",
    r"NXAK4.1B/2021_04_02_Transition_Imaging",
    r"NXAK4.1B/2021_04_08_Transition_Imaging",
    r"NXAK4.1B/2021_04_10_Transition_Imaging",

    r"NXAK7.1B/2021_02_26_Switching_Imaging",
    r"NXAK7.1B/2021_02_28_Switching_Imaging",
    r"NXAK7.1B/2021_03_02_Switching_Imaging",
    r"NXAK7.1B/2021_03_23_Transition_Imaging",
    r"NXAK7.1B/2021_03_31_Transition_Imaging",
    r"NXAK7.1B/2021_04_02_Transition_Imaging",
]


visual_context_vis_1_onset_file = "visual_context_stable_vis_1_onsets.npy"
visual_context_vis_2_onset_file = "visual_context_stable_vis_2_onsets.npy"
odour_context_vis_1_onset_file = "odour_context_stable_vis_1_onsets.npy"
odour_context_vis_2_onset_file = "odour_context_stable_vis_2_onsets.npy"

for base_directory in session_list:
    check_bodycam_alignment(os.path.join("/media/matthew/Expansion/Control_Data", base_directory), visual_context_vis_1_onset_file, visual_context_vis_2_onset_file, start_window=-2, stop_window=5)
    check_bodycam_alignment(os.path.join("/media/matthew/Expansion/Control_Data", base_directory), odour_context_vis_1_onset_file, odour_context_vis_2_onset_file, start_window=-2, stop_window=5)
