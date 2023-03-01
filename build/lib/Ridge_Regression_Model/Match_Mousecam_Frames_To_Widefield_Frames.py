import os
import sys
import numpy as np
from bisect import bisect_left

def invert_dictionary(dictionary):
    inv_map = {v: k for k, v in dictionary.items()}
    return inv_map

def take_closest(myList, myNumber):

    """
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.
    """

    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before


def match_mousecam_to_widefield_frames(base_directory):

    # Load Frame Times
    widefield_frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()] # Keys are Times, Values Are Frames
    mousecam_frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Mousecam_Frame_Times.npy"), allow_pickle=True)[()] # Keys are Times, Values Are Frames

    print("Min Widefield Frame Time", np.min(list(widefield_frame_times.keys())), "Maximum Widefield Frame Time", np.max(list(widefield_frame_times.keys())))
    print("Min Mousecam Frame Time", np.min(list(mousecam_frame_times.keys())), "Maximum Mousecam Frame Time", np.max(list(mousecam_frame_times.keys())))

    widefield_frame_times = invert_dictionary(widefield_frame_times)
    widefield_frame_time_keys = list(widefield_frame_times.keys())
    mousecam_frame_times_keys = list(mousecam_frame_times.keys())
    mousecam_frame_times_keys.sort()

    # Get Number of Frames
    number_of_widefield_frames = len(widefield_frame_time_keys)

    # Dictionary - Keys are Widefield Frame Indexes, Values are Closest Mousecam Frame Indexes
    widfield_to_mousecam_frame_dict = {}

    for widefield_frame in range(number_of_widefield_frames):
        frame_time = widefield_frame_times[widefield_frame]
        closest_mousecam_time = take_closest(mousecam_frame_times_keys, frame_time)
        closest_mousecam_frame = mousecam_frame_times[closest_mousecam_time]
        widfield_to_mousecam_frame_dict[widefield_frame] = closest_mousecam_frame
        #print("Widefield Frame: ", widefield_frame, " Closest Mousecam Frame: ", closest_mousecam_frame)

    # Save Directory
    save_directoy = os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy")
    np.save(save_directoy, widfield_to_mousecam_frame_dict)




session_list = [

        r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging",
         r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_24_Discrimination_Imaging",

        r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_15_Discrimination_Imaging",
         r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_25_Discrimination_Imaging",

        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_06_Discrimination_Imaging",
         r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_22_Discrimination_Imaging",

        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_03_Discrimination_Imaging",
         r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_24_Discrimination_Imaging",

        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_01_Discrimination_Imaging",
         r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_09_Discrimination_Imaging",

        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_29_Discrimination_Imaging",
         r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging"

    ]

session_list = [
    r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_14_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_21_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_14_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_23_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_01_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_22_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_04_29_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_07_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_25_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_07_Discrimination_Imaging"
]
for session in session_list:
    match_mousecam_to_widefield_frames(session)
