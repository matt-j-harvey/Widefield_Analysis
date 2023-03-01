import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from tqdm import tqdm


def z_score_df(base_directory, early_cutoff=3000):

    # Load Delta F
    delta_f_matrix = np.load(os.path.join(base_directory, "Delta_F_Matrix_100_by_100_SVD.npy"))
    print("Raw Delta F Matrix Shape", np.shape(delta_f_matrix))

    number_of_timepoints, number_of_pixels = np.shape(delta_f_matrix)

    # Remove Early Cutoff
    delta_f_matrix = delta_f_matrix[early_cutoff:]
    delta_f_matrix = np.nan_to_num(delta_f_matrix)

    z_scored_delta_f_matrix = stats.zscore(delta_f_matrix, axis=0)
    z_scored_delta_f_matrix = np.nan_to_num(z_scored_delta_f_matrix)

    # Insert Back
    zero_padding = np.zeros((early_cutoff, number_of_pixels))
    delta_f_matrix = np.vstack([zero_padding, z_scored_delta_f_matrix])
    print("Z Score Delta F Matrix Shape", np.shape(delta_f_matrix))

    np.save(os.path.join(base_directory, "Delta_F_Matrix_100_by_100_SVD_Z_Score.npy"), delta_f_matrix)



session_list = [

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



    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_13_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_15_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_17_Switching_Imaging",

    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_31_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_02_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_04_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_08_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_10_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_12_Transition_Imaging",

    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_20_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_22_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_24_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_14_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_16_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_18_Transition_Imaging",

    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_17_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_19_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_23_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_30_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_06_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_08_Transition_Imaging",

    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_15_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_17_Switching_Imaging",  ## here on 28/01
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_19_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_22_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_24_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_26_Transition_Imaging",

    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_14_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_20_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_26_Switching_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_05_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_08_Transition_Imaging",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_10_Transition_Imaging",
]


for session in tqdm(session_list):
    z_score_df(session, early_cutoff=3000)