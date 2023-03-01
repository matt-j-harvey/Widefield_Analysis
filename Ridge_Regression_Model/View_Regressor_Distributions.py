import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from Widefield_Utils import widefield_utils

def view_behaviour_regressor_distributions(tensor_save_directory, base_directory, design_matrix, design_matrix_dict, number_of_stimuli):

    print("Design Matrix", np.shape(design_matrix))

    # Unpack Dictiionaries
    number_of_behaviour_regressor_groups = design_matrix_dict["number_of_regressor_groups"]
    group_sizes = design_matrix_dict[ "coef_group_sizes"]
    group_starts = design_matrix_dict["coef_group_starts"]
    group_stops = design_matrix_dict["coef_group_stops"]
    coef_names = design_matrix_dict["coefs_names"]

    # Get Number Of Regressor Groups
    number_of_behavioural_regressors = np.sum(group_sizes[number_of_stimuli:])
    print("Number of behavioural regressors", number_of_behavioural_regressors)

    n_rows, n_columns = widefield_utils.get_best_grid(number_of_behavioural_regressors)
    figure_1 = plt.figure(figsize=(50, 50))

    regressor_count = 0
    for behaviour_regressor_group_index in range(number_of_stimuli, number_of_behaviour_regressor_groups):
        behaviour_group_start = group_starts[behaviour_regressor_group_index]
        behaviour_group_stop = group_stops[behaviour_regressor_group_index]

        for regressor_index in range(behaviour_group_start, behaviour_group_stop):
            regressor_name = coef_names[regressor_index]
            regressor_data = design_matrix[:, regressor_index]

            axis = figure_1.add_subplot(n_rows, n_columns, regressor_count + 1)
            axis.set_title(regressor_name)
            axis.hist(regressor_data)

            regressor_count += 1

    plt.savefig(os.path.join(tensor_save_directory, base_directory, "Regressor_Distributions.png"))
    plt.close()


# Select Sessions
selected_session_list = [

    [r"NRXN78.1A/2020_11_28_Switching_Imaging",
    r"NRXN78.1A/2020_12_05_Switching_Imaging",
    r"NRXN78.1A/2020_12_09_Switching_Imaging"],

    ["NRXN78.1D/2020_12_07_Switching_Imaging",
     r"NRXN78.1D/2020_11_29_Switching_Imaging",
    r"NRXN78.1D/2020_12_05_Switching_Imaging"],

    [r"NXAK14.1A/2021_05_21_Switching_Imaging",
    r"NXAK14.1A/2021_05_23_Switching_Imaging",
    r"NXAK14.1A/2021_06_11_Switching_Imaging"],

    [r"NXAK22.1A/2021_10_14_Switching_Imaging",
    r"NXAK22.1A/2021_10_20_Switching_Imaging",
    r"NXAK22.1A/2021_10_22_Switching_Imaging"],

    [r"NXAK4.1B/2021_03_02_Switching_Imaging",
    r"NXAK4.1B/2021_03_04_Switching_Imaging",
    r"NXAK4.1B/2021_03_06_Switching_Imaging"],

    [r"NXAK7.1B/2021_02_26_Switching_Imaging",
    r"NXAK7.1B/2021_02_28_Switching_Imaging",
    #r"NXAK7.1B/2021_03_02_Switching_Imaging" - Falied Mousecam Check
     ],

]


tensor_save_directory = r"/media/matthew/External_Harddrive_2/Control_Switching_Analysis/Full_Model"
number_of_stimuli = 4

for mouse in tqdm(selected_session_list, leave=True, position=0, desc="Mouse"):
  for base_directory in tqdm(mouse, leave=True, position=1, desc="Session"):
      print(base_directory)
      design_matrix = np.load(os.path.join(tensor_save_directory, base_directory, "Full_Model_Design_Matrix.npy"))
      design_matrix_dict = np.load(os.path.join(tensor_save_directory, base_directory, "design_matrix_key_dict.npy"), allow_pickle=True)[()]
      view_behaviour_regressor_distributions(tensor_save_directory, base_directory, design_matrix, design_matrix_dict, number_of_stimuli)
