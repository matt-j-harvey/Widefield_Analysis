import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import matplotlib.gridspec as gridspec
import sys
sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions


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




def create_mousecam_tensor(base_directory, video_file, onsets, start_window, stop_window):

    # Get Number Of Trials
    number_of_trials = len(onsets)

    # Load Mousecam Offset
    mousecam_offset = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Mousecam_Frame_Offset.npy"))[0]
    print("Mousecam Offset", mousecam_offset)


    # Get The Selected Mousecam Frames For Each Trial
    selected_frames = []
    for trial_index in range(number_of_trials):
        trial_start = onsets[trial_index] + start_window + mousecam_offset
        trial_stop = onsets[trial_index] + stop_window + mousecam_offset
        trial_frames = list(range(trial_start, trial_stop))
        selected_frames.append(trial_frames)
    selected_frames = np.array(selected_frames)

    full_video_filepath = os.path.join(base_directory, video_file)
    mousecam_tensor = load_video_as_numpy_array(full_video_filepath, selected_frames)


    return mousecam_tensor




def normalise_trace(trace):

    # Set Minimum To Zero
    trace_min = np.min(trace)
    trace = np.subtract(trace, trace_min)

    # Set Max To One
    trace_max = np.max(trace)
    trace = np.divide(trace, trace_max)

    return trace


def visualise_mousecam_onsets(base_directory, mousecam_tensor, widefield_onsets, start_window, stop_window):

    mousecam_expsoure_time = 5

    # Create Output Directory
    save_directory = os.path.join(base_directory, "Mousecam_Timing_Checks")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)


    # Load Ai Data
    ai_file = Widefield_General_Functions.get_ai_filename(base_directory)
    print(ai_file)
    ai_data = Widefield_General_Functions.load_ai_recorder_file(base_directory + ai_file)

    stimuli_dictionary = Widefield_General_Functions.create_stimuli_dictionary()

    led_1_trace = ai_data[stimuli_dictionary['LED 1']]
    mousecam_trace = ai_data[stimuli_dictionary['Mousecam']]
    photodiode_trace = ai_data[stimuli_dictionary['Photodiode']]
    vis_2_trace = ai_data[stimuli_dictionary['Visual 2']]

    led_1_trace = normalise_trace(led_1_trace)
    mousecam_trace = normalise_trace(mousecam_trace)
    photodiode_trace = normalise_trace(photodiode_trace)
    vis_2_trace = normalise_trace(vis_2_trace)

    # Load Frame Time Dictionaries
    widefield_time_frame_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    widefield_frame_time_dict = Widefield_General_Functions.invert_dictionary(widefield_time_frame_dict)

    mousecam_frame_time_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Mousecam_Frame_Times.npy"), allow_pickle=True)[()]
    mousecam_frame_time_dict = Widefield_General_Functions.invert_dictionary(mousecam_frame_time_dict)

    widefield_to_mousecam_frame_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy"), allow_pickle=True)[()]


    # Get Tensor Details
    number_of_trials = np.shape(mousecam_tensor)[0]
    trial_length = np.shape(mousecam_tensor)[1]

    trial_window = 100

    for trial_index in range(number_of_trials):
        print(trial_index)

        figure_1 = plt.figure(figsize=(80, 60))
        figure_1.suptitle("Trial: " + str(trial_index))
        grid_spec_1 = gridspec.GridSpec(ncols=5, nrows=2, figure=figure_1)
        grid_spec_1.tight_layout(figure_1)
        graph_axis = figure_1.add_subplot(grid_spec_1[0, :])

        # Plot Ai Data
        trial_start = widefield_frame_time_dict[widefield_onsets[trial_index]] - trial_window
        trial_stop = widefield_frame_time_dict[widefield_onsets[trial_index]] + trial_window

        trial_mousecam_trace    = mousecam_trace[trial_start:trial_stop]
        trial_led_1_trace       = led_1_trace[trial_start:trial_stop]
        trial_photodiode_trace  = photodiode_trace[trial_start:trial_stop]
        trial_vis_2_trace       = vis_2_trace[trial_start:trial_stop]

        relative_widefield_onset = widefield_frame_time_dict[widefield_onsets[trial_index]] - trial_start
        graph_axis.axvline(x=relative_widefield_onset, ymin=0, ymax=1, c='darkblue', linewidth=5)

        # Selected |Mousecam Onset
        widefield_onset_frame = widefield_onsets[trial_index]
        print("Widefield onset Frame", widefield_onset_frame)

        print("Trial start", trial_start)

        selected_mousecam_onset = widefield_to_mousecam_frame_dict[widefield_onset_frame]
        print("Selected mousecam onset", selected_mousecam_onset)

        selected_mousecam_onset_time = mousecam_frame_time_dict[selected_mousecam_onset]
        print("Selcted Mousecam Time", selected_mousecam_onset_time)


        # Mark Mousecam Exposure Time
        trial_mousecam_onset_list = Widefield_General_Functions.get_step_onsets(trial_mousecam_trace, threshold=0.5, window=1)
        print("Trial Mousecam List", trial_mousecam_onset_list)
        for trial_mousecam_onset in trial_mousecam_onset_list:
            xmin = trial_mousecam_onset
            xmax = trial_mousecam_onset + mousecam_expsoure_time
            graph_axis.axvspan(xmin, xmax, ymin=0, ymax=1, color='tab:orange', alpha=0.5)



        relative_mousecam_onset = selected_mousecam_onset_time - trial_start
        graph_axis.axvline(x=relative_mousecam_onset, ymin=0, ymax=1, c='k', linewidth=5)

        graph_axis.plot(trial_mousecam_trace, c='tab:orange', alpha=0.5)
        graph_axis.plot(trial_led_1_trace, c='b', alpha=0.2)
        graph_axis.plot(trial_photodiode_trace, c='g', linewidth=5)
        graph_axis.plot(trial_vis_2_trace, c='m', alpha=0.2)

        # Plot Mousecam Frames
        timepoint_title_list = list(range(start_window, stop_window))
        print("Timepoint title list", timepoint_title_list)
        for timepoint in range(trial_length):
            axis = figure_1.add_subplot(grid_spec_1[1, timepoint])
            axis.imshow(mousecam_tensor[trial_index][timepoint], cmap='jet')
            axis.set_title(str(timepoint_title_list[timepoint]), fontsize=200)
            axis.axis('off')

        #plt.show()
        plt.savefig(os.path.join(save_directory, str(trial_index).zfill(3) + ".png"))
        plt.clf()


def check_mousecam_timings(base_directory, video_file, onsets_file_list, start_window, stop_window):

    # Load Widefield Frame Indexes of Trial Starts
    widefield_onsets = load_onsets(base_directory, onsets_file_list)

    # Convert Widefield Frames Into Mousecam Frames
    mousecam_onsets = convert_widefield_onsets_into_mousecam_onsets(base_directory, widefield_onsets)

    # Load Video Data Into A Tensor Of Shape (N_trials , Trial Length, Image Height, Image Width)
    mousecam_tensor = create_mousecam_tensor(base_directory, video_file, mousecam_onsets, start_window, stop_window)

    # Visualuse Mousecam Onsets
    visualise_mousecam_onsets(base_directory, mousecam_tensor, widefield_onsets, start_window, stop_window)



