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


session_list = [

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_14_Discrimination_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_09_Discrimination_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_04_Discrimination_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_02_Discrimination_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_14_Discrimination_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_02_Discrimination_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_15_Discrimination_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_30_Discrimination_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_22_Discrimination_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_08_Discrimination_Imaging",
    ]
session_list = ["/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_15_Discrimination_Imaging"]

full_session_list = [

    "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_17_Discrimination_Imaging",
    "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_19_Discrimination_Imaging",
    "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_21_Discrimination_Imaging",

    "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_21_Discrimination_Imaging",
    "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_23_Discrimination_Imaging",
    "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_25_Discrimination_Imaging",

    "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_22_Discrimination_Imaging",

    "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_24_Discrimination_Imaging",

    "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_05_Discrimination_Imaging",
    "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_07_Discrimination_Imaging",
    "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_09_Discrimination_Imaging",

    "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_07_Discrimination_Imaging",
    "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging",

    "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_05_Discrimination_Imaging",

    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_03_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging",

    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_12_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_14_Discrimination_Imaging",

    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_15_Discrimination_Imaging",

    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_11_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_13_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_15_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_17_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging",

    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_02_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_04_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_06_Discrimination_Imaging",
]

full_session_list = [

    "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging",
    "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_15_Discrimination_Imaging",

    "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_08_Discrimination_Imaging",
    "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_10_Discrimination_Imaging",

    "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_01_Discrimination_Imaging",
    "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_03_Discrimination_Imaging",
    "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_05_Discrimination_Imaging",
    "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_07_Discrimination_Imaging",
    "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_09_Discrimination_Imaging",

    "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_04_29_Discrimination_Imaging",
    "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_01_Discrimination_Imaging",
    "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_03_Discrimination_Imaging",

    "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_25_Discrimination_Imaging",

    "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_13_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_14_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_15_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_16_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_17_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_19_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_21_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_23_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_25_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_27_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_29_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_01_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_03_Discrimination_Imaging",

    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_02_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_04_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_06_Discrimination_Imaging",

    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_04_30_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_02_Discrimination_Imaging",

    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_04_30_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_02_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_04_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_06_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_08_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_10_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_12_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_14_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_16_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_18_Discrimination_Imaging",

    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_28_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_30_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_02_Discrimination_Imaging",

    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_20_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_22_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_24_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_26_Discrimination_Imaging",

]

for session in full_session_list:
    match_mousecam_to_widefield_frames(session)
