import numpy as np
import os
from tqdm import tqdm
import Behaviour_Utils

def save_behaviour_matrix_as_csv(base_directory):

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)


    header = [
        "0 trial_index," 
        "1 trial_type,"
        "2 lick,"
        "3 correct,"
        "4 rewarded,"
        "5 preeceded_by_irrel,"
        "6 irrel_type,"
        "7 ignore_irrel,"
        "8 block_number,"
        "9 first_in_block,"
        "10 in_block_of_stable_performance,"
        "11 stimuli_onset,"
        "12 stimuli_offset,"
        "13 irrel_onset,"
        "14 irrel_offset,"
        "15 trial_end,"
        "16 Photodiode Onset,"
        "17 Photodiode Offset,"
        "18 Onset closest Frame,"
        "19 Offset Closest Frame,"
        "20 Irrel Onset Closest Frame,"
        "21 Irrel Offset Closest Frame,"
        "22 Lick Onset,"
        "23 Reaction Time,"
        ]
    header = " ".join(header)

    np.savetxt(os.path.join(base_directory, "Behaviour_Matrix.csv"), behaviour_matrix, delimiter=",", fmt="%s", header=header, comments="",  newline='\n')


# Load Sessions
"""
session_type = "Switching"
session_list = Behaviour_Utils.load_all_sessions_of_type(session_type)
for session in tqdm(session_list):
    save_behaviour_matrix_as_csv(session)
"""


mutant_mouse_list = [

    #Mutants
    # 72.1A - Slow Learner
    [r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN71.2A/2020_11_13_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN71.2A/2020_11_14_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN71.2A/2020_11_15_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN71.2A/2020_11_16_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN71.2A/2020_11_17_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN71.2A/2020_11_18_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN71.2A/2020_11_19_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN71.2A/2020_11_20_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN71.2A/2020_11_21_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN71.2A/2020_11_22_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN71.2A/2020_11_23_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN71.2A/2020_11_24_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN71.2A/2020_11_25_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN71.2A/2020_11_26_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN71.2A/2020_11_27_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN71.2A/2020_11_28_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN71.2A/2020_11_29_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN71.2A/2020_11_30_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN71.2A/2020_12_01_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN71.2A/2020_12_02_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN71.2A/2020_12_03_Discrimination_Imaging"],


    #4.1A Slow Learner
    [r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_02_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_03_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_04_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_05_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_06_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_07_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_08_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_09_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_10_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_11_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_12_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_13_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_14_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_15_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_16_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_17_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_18_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_19_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_20_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_21_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_22_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_23_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_24_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_25_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_26_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_27_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_02_28_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1A/2021_03_01_Discrimination_Imaging"],

    #10.1A Fast learner
    [r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK10.1A/2021_04_30_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK10.1A/2021_05_01_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK10.1A/2021_05_02_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK10.1A/2021_05_04_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK10.1A/2021_05_05_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK10.1A/2021_05_06_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK10.1A/2021_05_07_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK10.1A/2021_05_08_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK10.1A/2021_05_09_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK10.1A/2021_05_10_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK10.1A/2021_05_11_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK10.1A/2021_05_12_Discrimination_Imaging",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK10.1A/2021_05_13_Discrimination_Behaviour",
    r"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK10.1A/2021_05_14_Discrimination_Imaging",
    ],

    # 16.1B Slow Learner
    [
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_04_30_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_05_01_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_05_02_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_05_03_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_05_04_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_05_05_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_05_06_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_05_07_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_05_08_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_05_09_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_05_10_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_05_11_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_05_12_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_05_13_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_05_14_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_05_16_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_05_17_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_05_18_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_05_19_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_05_20_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_05_21_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_05_22_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_05_23_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_05_24_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_05_25_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_05_26_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK16.1B/2021_05_27_Discrimination_Behaviour",
    ],

    #24.1C Fast Learner
    [
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK24.1C/2021_09_20_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK24.1C/2021_09_21_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK24.1C/2021_09_22_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK24.1C/2021_09_23_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK24.1C/2021_09_24_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK24.1C/2021_09_25_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK24.1C/2021_09_26_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK24.1C/2021_09_27_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK24.1C/2021_09_28_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK24.1C/2021_09_29_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK24.1C/2021_09_30_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK24.1C/2021_10_01_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK24.1C/2021_10_02_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK24.1C/2021_10_03_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK24.1C/2021_10_04_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK24.1C/2021_10_05_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK24.1C/2021_10_06_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK24.1C/2021_10_08_Discrimination_Imaging",
    ],

    # 20.1B Slow Leaner
    [
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK20.1B/2021_09_28_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK20.1B/2021_09_29_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK20.1B/2021_09_30_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK20.1B/2021_10_01_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK20.1B/2021_10_02_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK20.1B/2021_10_03_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK20.1B/2021_10_04_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK20.1B/2021_10_05_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK20.1B/2021_10_06_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK20.1B/2021_10_07_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK20.1B/2021_10_08_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK20.1B/2021_10_09_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK20.1B/2021_10_10_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK20.1B/2021_10_11_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK20.1B/2021_10_12_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK20.1B/2021_10_13_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK20.1B/2021_10_14_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK20.1B/2021_10_15_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK20.1B/2021_10_16_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK20.1B/2021_10_17_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK20.1B/2021_10_18_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK20.1B/2021_10_20_Discrimination_Behaviour",
    ],
    ]

control_mice_list = [ # Control Mice

    #78.1A - Fast Learner
    ["/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN78.1A/2020_11_14_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN78.1A/2020_11_16_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN78.1A/2020_11_17_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN78.1A/2020_11_18_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN78.1A/2020_11_19_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN78.1A/2020_11_20_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN78.1A/2020_11_21_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN78.1A/2020_11_22_Discrimination_Behaviour",],

    # 78.1D - Fast learner
    ["/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN78.1D/2020_11_14_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN78.1D/2020_11_15_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN78.1D/2020_11_16_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN78.1D/2020_11_17_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN78.1D/2020_11_18_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN78.1D/2020_11_19_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN78.1D/2020_11_20_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN78.1D/2020_11_21_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN78.1D/2020_11_22_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN78.1D/2020_11_23_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN78.1D/2020_11_24_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN78.1D/2020_11_25_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NRXN78.1D/2020_11_26_Discrimination_Behaviour"],

    #4.1B
    ["/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1B/2021_02_05_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1B/2021_02_07_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1B/2021_02_08_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1B/2021_02_09_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1B/2021_02_10_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1B/2021_02_11_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1B/2021_02_12_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1B/2021_02_13_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1B/2021_02_15_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1B/2021_02_18_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1B/2021_02_19_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1B/2021_02_20_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1B/2021_02_21_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK4.1B/2021_02_22_Discrimination_Imaging"],


    #14.1A FSt learner
    ["/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK14.1A/2021_04_29_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK14.1A/2021_04_30_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK14.1A/2021_05_01_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK14.1A/2021_05_02_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK14.1A/2021_05_03_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK14.1A/2021_05_04_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK14.1A/2021_05_05_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK14.1A/2021_05_06_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK14.1A/2021_05_07_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK14.1A/2021_05_08_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK14.1A/2021_05_09_Discrimination_Imaging"],

    # 22.1A
    ["/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK22.1A/2021_09_25_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK22.1A/2021_09_26_Discrimination_Behaviour",
    #"/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK22.1A/2021_09_27_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK22.1A/2021_09_28_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK22.1A/2021_09_29_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK22.1A/2021_09_30_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK22.1A/2021_10_01_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK22.1A/2021_10_02_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK22.1A/2021_10_03_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK22.1A/2021_10_04_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK22.1A/2021_10_05_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK22.1A/2021_10_06_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK22.1A/2021_10_07_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK22.1A/2021_10_09_Discrimination_Behaviour"],

    #7.1B Slow learner
    ["/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK7.1B/2021_02_01_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK7.1B/2021_02_02_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK7.1B/2021_02_03_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK7.1B/2021_02_04_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK7.1B/2021_02_05_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK7.1B/2021_02_06_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK7.1B/2021_02_07_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK7.1B/2021_02_08_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK7.1B/2021_02_09_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK7.1B/2021_02_10_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK7.1B/2021_02_11_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK7.1B/2021_02_12_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK7.1B/2021_02_13_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK7.1B/2021_02_14_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK7.1B/2021_02_15_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK7.1B/2021_02_16_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK7.1B/2021_02_17_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK7.1B/2021_02_18_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK7.1B/2021_02_19_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK7.1B/2021_02_20_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK7.1B/2021_02_21_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK7.1B/2021_02_22_Discrimination_Imaging",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK7.1B/2021_02_23_Discrimination_Behaviour",
    "/media/matthew/External_Harddrive_2/Learning_Behaviour_Data/NXAK7.1B/2021_02_24_Discrimination_Imaging"],

]

base_directory = r"//media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_29_Switching_Imaging"

save_behaviour_matrix_as_csv(base_directory)