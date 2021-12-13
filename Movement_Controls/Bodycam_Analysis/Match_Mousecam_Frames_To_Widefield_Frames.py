import os
import sys
import numpy as np
from bisect import bisect_left

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions


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
    widefield_frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    mousecam_frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Mousecam_Frame_Times.npy"), allow_pickle=True)[()]

    widefield_frame_times = Widefield_General_Functions.invert_dictionary(widefield_frame_times)
    widefield_frame_time_keys = list(widefield_frame_times.keys())
    mousecam_frame_times_keys = list(mousecam_frame_times.keys())
    mousecam_frame_times_keys.sort()

    # Get Number of Frames
    number_of_widefield_frames = len(widefield_frame_time_keys)

    # Invert
    print("Widefield Frame Time Keys", list(widefield_frame_times.keys())[0:10])
    print("Widefield Frame Times Values", list(widefield_frame_times.values())[0:10])
    print("Mousecam Frame Times Keys", list(mousecam_frame_times.keys())[0:10])
    print("Mousecam Frame Times Values", list(mousecam_frame_times.values())[0:10])


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
