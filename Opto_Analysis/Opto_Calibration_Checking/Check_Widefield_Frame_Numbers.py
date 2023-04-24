import numpy as np
import os
import cv2
from tqdm import tqdm
import tables



def check_widefield_frame_times(base_directory):

    # Load Widfield Frame Times
    widefield_frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    widefield_frame_times = list(widefield_frame_times.values())
    number_of_widefield_triggers = len(widefield_frame_times)

    # Load Widefield Frame Data
    widefield_filename = os.path.join(base_directory, "Downsampled_Delta_F.h5")
    widefield_file_container = tables.open_file(widefield_filename, mode="r")
    widefield_data = widefield_file_container.root["Data"]
    widefield_frames = np.shape(widefield_data)[0]
    widefield_file_container.close()


    # Check Match
    if number_of_widefield_triggers == widefield_frames:
        match_message = "Frame Numbrrs Match :) "
        filename = "Frame_Check_Passed.txt"
        print("Passed :)")

    else:
        match_message = "Frame Numbers Dont Match :( "
        filename = "Frame_Check_Failed.txt"
        print("Failed :( " + base_directory + "Frame Difference " + str(number_of_widefield_triggers - widefield_frames))

        # Save As Text File
    text_filename = os.path.join(base_directory, filename)
    with open(text_filename, 'w') as f:
        f.write('Widefield Frame Triggers: ' + str(number_of_widefield_triggers) + "\n")
        f.write('Widefield Frames: ' + str(widefield_frames) + "\n")
        f.write(match_message  + "\n")

    return match_message
