import numpy as np
import os

def save_mousecam_offset(base_directory, offset):
    offset_array = [offset]
    offset_array = np.array(offset_array)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "Mousecam_Frame_Offset.npy"), offset_array)


base_directory = r"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging"
save_mousecam_offset(base_directory, 1)