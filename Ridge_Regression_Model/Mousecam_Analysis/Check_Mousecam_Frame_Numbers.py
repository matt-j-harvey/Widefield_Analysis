import numpy as np
import os
import cv2
from tqdm import tqdm


def get_video_frames(video_file):
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frameCount


def get_bodycam_filename(base_directory):
    file_list = os.listdir(base_directory)
    for file_name in file_list:
        if "_cam_1.mp4" in file_name:
            return file_name


def check_mousecam_frame_times(base_directory):

    # Load Mousecam Frame Times
    mousecam_frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Mousecam_Frame_Times.npy"), allow_pickle=True)[()]
    mousecam_frame_values = list(mousecam_frame_times.values())
    number_of_mousecam_triggers = len(mousecam_frame_values)

    # Load Mousecam Frame Data
    bodycam_name = get_bodycam_filename(base_directory)
    bodycam_frames = get_video_frames(os.path.join(base_directory, bodycam_name))

    # Check Match
    if number_of_mousecam_triggers == bodycam_frames:
        match_message = "Frame Numbrrs Match :) "
        filename = "Frame_Check_Passed.txt"


    else:
        match_message = "Frame Numbers Dont Match :( "
        filename = "Frame_Check_Failed.txt"
        print("Failed :( " + base_directory + "Frame Difference " + str(number_of_mousecam_triggers - bodycam_frames))

        # Save As Text File
    text_filename = os.path.join(base_directory, "Mousecam_Analysis", filename)
    with open(text_filename, 'w') as f:
        f.write('Bodycam Frame Triggers: ' + str(number_of_mousecam_triggers) + "\n")
        f.write('Bodycam Frames: ' + str(bodycam_frames) + "\n")
        f.write(match_message  + "\n")





session_list = [
    r"NRXN78.1A/2020_11_28_Switching_Imaging",
    r"NRXN78.1A/2020_12_05_Switching_Imaging",
    r"NRXN78.1A/2020_12_09_Switching_Imaging",

    r"NRXN78.1D/2020_12_07_Switching_Imaging",
    r"NRXN78.1D/2020_11_29_Switching_Imaging",
    r"NRXN78.1D/2020_12_05_Switching_Imaging",

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

    r"NRXN78.1A/2020_11_14_Discrimination_Imaging",
    r"NRXN78.1A/2020_11_15_Discrimination_Imaging",
    r"NRXN78.1A/2020_11_24_Discrimination_Imaging",
    r"NRXN78.1A/2020_11_21_Discrimination_Imaging",

    r"NRXN78.1D/2020_11_14_Discrimination_Imaging",
    r"NRXN78.1D/2020_11_15_Discrimination_Imaging",
    r"NRXN78.1D/2020_11_25_Discrimination_Imaging",
    r"NRXN78.1D/2020_11_23_Discrimination_Imaging",

    r"NXAK4.1B/2021_02_04_Discrimination_Imaging",
    r"NXAK4.1B/2021_02_06_Discrimination_Imaging",
    r"NXAK4.1B/2021_02_22_Discrimination_Imaging",
    r"NXAK4.1B/2021_02_14_Discrimination_Imaging",

    r"NXAK7.1B/2021_02_01_Discrimination_Imaging",
    r"NXAK7.1B/2021_02_03_Discrimination_Imaging",
    r"NXAK7.1B/2021_02_24_Discrimination_Imaging",
    r"NXAK7.1B/2021_02_22_Discrimination_Imaging",

    r"NXAK14.1A/2021_04_29_Discrimination_Imaging",
    r"NXAK14.1A/2021_05_01_Discrimination_Imaging",
    r"NXAK14.1A/2021_05_09_Discrimination_Imaging",
    r"NXAK14.1A/2021_05_07_Discrimination_Imaging",

    r"NXAK22.1A/2021_09_25_Discrimination_Imaging",
    r"NXAK22.1A/2021_09_29_Discrimination_Imaging",
    r"NXAK22.1A/2021_10_08_Discrimination_Imaging",
    r"NXAK22.1A/2021_10_07_Discrimination_Imaging",

]

full_session_list = []
for item in session_list:
    full_session_list.append(os.path.join("/media/matthew/Expansion/Control_Data", item))

for base_directory in full_session_list:
    check_mousecam_frame_times(base_directory)