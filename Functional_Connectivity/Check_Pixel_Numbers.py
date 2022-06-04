import h5py
import numpy as np
import sys
import os

controls = ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_02_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_08_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Switching_Analysis/Wildtype/NXAK4.1B/2021_03_04_Switching_Imaging",

            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_23_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_31_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_15_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_10_29_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_11_05_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive1/Switching_Analysis/Wildtype/NXAK7.1B/2021_03_02_Switching_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Switching_Analysis/Wildtype/NXAK14.1A/2021_06_09_Switching_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Switching_Analysis/Wildtype/NRXN78.1A/2020_12_05_Switching_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Switching_Analysis/Wildtype/NRXN78.1A/2020_12_09_Switching_Imaging"]


for base_directory in controls:
    print(base_directory)

    # Load Delta F Matrix
    delta_f_matrix_filepath = os.path.join(base_directory, "Delta_F_Registered.hdf5")
    delta_f_matrix_container = h5py.File(delta_f_matrix_filepath, 'r')
    delta_f_matrix = delta_f_matrix_container['Data']

    print(np.shape(delta_f_matrix))