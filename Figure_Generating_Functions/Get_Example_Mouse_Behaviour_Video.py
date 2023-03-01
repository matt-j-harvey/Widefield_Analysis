import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def extract_video_data(video_file, selected_frame, start_window, stop_window):

    window_size = stop_window - start_window
    trial_start = selected_frame + start_window

    # Open Video File
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Extract Selected Frames
    data = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, trial_start - 1)

    for frame_index in range(window_size):
        ret, frame = cap.read()
        frame = frame[:, :, 0]
        data.append(frame)

    cap.release()
    data = np.array(data)
    return data


def get_bodycam_filename(base_directory):
    file_list = os.listdir(base_directory)
    for file_name in file_list:
        if "_cam_1.mp4" in file_name:
            return file_name



def save_trial_video(video_data, save_directory, trial_index):

    # Get Data Shape
    n_frames, image_height, image_width = np.shape(video_data)

    # Create Video File
    video_file = os.path.join(save_directory, "Trial_" + str(trial_index).zfill(3) + ".avi")
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_file, video_codec, frameSize=(image_width, image_height), fps=30)  # 0, 12

    for frame in range(n_frames):
        frame_data = video_data[frame]
        frame_data = np.ndarray.astype(frame_data, np.uint8)
        frame_data = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)
        video.write(frame_data)

    cv2.destroyAllWindows()
    video.release()

def create_example_video(base_directory, onsets_file, start_window, stop_window, save_directory):

    # Load Mouseframe Frame dict
    widefield_to_mousecam_frame_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy"), allow_pickle=True)[()]


    # Load Onsets
    # Check File Exists
    onsets_file_path = os.path.join(base_directory,  "Stimuli_Onsets", onsets_file)
    if not os.path.exists(onsets_file_path):
        print("No Onsets File")
        return

    else:
        onsets_list = np.load(os.path.join(base_directory,  "Stimuli_Onsets", onsets_file))
        if len(onsets_list) < 1:
            print("NO Onsets in File")
            return

    # Get Bodycam Filename
    bodycam_filename = get_bodycam_filename(base_directory)
    print("Bodycam Filename", bodycam_filename)
    video_file = os.path.join(base_directory, bodycam_filename)


    n_onsets = len(onsets_list)
    for onset_index in range(n_onsets):

        onset = onsets_list[onset_index]
        print("Widefield Onset", onset)

        # Convert To Mousecam Onset
        mousecam_onset = widefield_to_mousecam_frame_dict[onset]
        print("Mousecam Onset", mousecam_onset)

        # Extract Video Data
        trial_video = extract_video_data(video_file, mousecam_onset, start_window, stop_window)

        # Create Video
        save_trial_video(trial_video, save_directory, onset_index)


# Maybe r"//media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_29_Transition_Imaging"

base_directory = r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_15_Transition_Imaging"
save_directory = r"/home/matthew/Documents/Example_Perfect_Switch_Video"
onsets_file_name = "perfect_transition_onsets.npy"
start_window = -2000
stop_window = 1100
create_example_video(base_directory, onsets_file_name, start_window, stop_window, save_directory)
