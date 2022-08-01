import numpy as np
import os
import sys

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions
import Check_Bodycam_Timing



def save_mousecam_offset(base_directory, offset):
    offset_array = [offset]
    offset_array = np.array(offset_array)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "Mousecam_Frame_Offset.npy"), offset_array)


controls = ["/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging", # Checked :)
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging", # Checked :)
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NRXN78.1A/2020_12_09_Switching_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NRXN78.1D/2020_11_29_Switching_Imaging"]

mutants =  ["/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK12.1F/2021_09_22_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NRXN71.2A/2020_12_17_Switching_Imaging"]

all_mice = controls + mutants

for base_directory in all_mice:
    save_mousecam_offset(base_directory, 1)

    start_window = -1
    stop_window = 4
    onset_files = [["visual_context_stable_vis_2_onsets.npy"], ["odour_context_stable_vis_2_onsets.npy"]]
    tensor_names = ["Vis_2_Stable_Visual", "Vis_2_Stable_Odour"]


    bodycam_file, eyecam_file = Widefield_General_Functions.get_mousecam_files(base_directory)
    Check_Bodycam_Timing.check_mousecam_timings(base_directory, bodycam_file, onset_files[0], start_window, stop_window)
