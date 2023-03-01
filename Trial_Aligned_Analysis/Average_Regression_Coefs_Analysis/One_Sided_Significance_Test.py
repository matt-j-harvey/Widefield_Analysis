import numpy as np
import h5py
import tables
import matplotlib.pyplot as plt
from scipy import stats
from skimage.transform import resize
import os
from statsmodels.stats.multitest import fdrcorrection

from Widefield_Utils import widefield_utils


def get_condition_regressors(regression_coefs, number_of_regressors, trial_length):

    condition_regressor_list = []

    for condition_index in range(number_of_regressors):

        # Get Stimuli Regressor For Each Condition
        condition_regressor_start = condition_index * trial_length
        condition_regressor_stop = condition_regressor_start + trial_length
        condition_regressors = regression_coefs[condition_regressor_start:condition_regressor_stop]
        condition_regressor_list.append(condition_regressors)

    return condition_regressor_list


def baseline_correct_trace(trace, baseline_window):
    baseline_values = trace[baseline_window]
    baseline_mean = np.mean(baseline_values)
    trace = np.subtract(trace, baseline_mean)
    return trace


def get_average_response(trace, response_window):
    trace_response = trace[response_window]
    average_response = np.mean(trace_response, axis=0)
    return average_response


def get_individual_session_modulation(session_list, tensor_directory, baseline_window, onset_files, start_window, stop_window, response_window, condition_1_index, condition_2_index):

    # Create Save Directory
    save_directory = os.path.join(tensor_directory, "Response_Modulation")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Get Regression Dict Structure
    number_of_regressors = len(onset_files)
    trial_length = stop_window - start_window

    # Load mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

    # Load Colourmap
    colourmap = widefield_utils.get_musall_cmap()

    for mouse in session_list:
        mouse_modulation = []

        for session in mouse:

            # Load Regressor Dict
            full_tensor_directory = os.path.join(tensor_directory, session)
            regressor_dict = np.load(os.path.join(full_tensor_directory, "Regression_Dictionary_Simple.npy"), allow_pickle=True)[()]
            regression_coefs = regressor_dict['Coefs']
            regression_coefs = np.transpose(regression_coefs)

            # Get Stimuli Regressors
            condition_regressors = get_condition_regressors(regression_coefs, number_of_regressors, trial_length)

            # Get Selected Conditions
            condition_1_regressor = condition_regressors[condition_1_index]
            condition_2_regressor = condition_regressors[condition_2_index]

            # Baseline Correct
            condition_1_regressor = baseline_correct_trace(condition_1_regressor, baseline_window)
            condition_2_regressor = baseline_correct_trace(condition_2_regressor, baseline_window)

            # Get Average Across Response Window
            condition_1_response = get_average_response(condition_1_regressor, response_window)
            condition_2_response = get_average_response(condition_2_regressor, response_window)

            # Get Average Modulation
            moduation = np.subtract(condition_1_response, condition_2_response)
            mouse_modulation.append(moduation)

            # Create Image
            moduation_image = widefield_utils.create_image_from_data(moduation, indicies, image_height, image_width)

            # Save Image
            session_name = session.replace("/", "_")
            plt.title(session_name + "Modulation")
            plt.imshow(moduation_image, cmap=colourmap, vmin=-0.02, vmax=0.02)
            plt.colorbar()
            plt.savefig(os.path.join(save_directory, session_name + ".png"))
            plt.close()

        # Get Mouse Average
        mouse_modulation = np.array(mouse_modulation)
        mean_mouse_modulation = np.mean(mouse_modulation, axis=0)
        mean_mouse_modulation_image = widefield_utils.create_image_from_data(mean_mouse_modulation, indicies, image_height, image_width)
        plt.imshow(mean_mouse_modulation_image, cmap=colourmap, vmin=-0.02, vmax=0.02)
        plt.colorbar()
        #plt.savefig(os.path.join(save_directory, session_name + ".png"))
        plt.show()


    # T Te
    mouse_modulation = np.array(mouse_modulation)
    t_stats, p_values = stats.ttest_1samp(mouse_modulation, axis=0, popmean=0)
    p_values = np.nan_to_num(p_values)
    t_stat_image = widefield_utils.create_image_from_data(t_stats,  indicies, image_height, image_width)
    plt.imshow(t_stat_image, cmap=colourmap, vmin=-2.5, vmax=2.5)
    plt.colorbar()
    plt.show()
    print("T stats", t_stats)

    inverse_p_values = 1 - p_values
    inverse_p_image = widefield_utils.create_image_from_data(inverse_p_values, indicies, image_height, image_width)
    plt.title("Inverse P")
    plt.imshow(inverse_p_image, cmap="inferno", vmin=0, vmax=1)
    plt.colorbar()
    plt.show()

    # Multiple Comparisons Correction
    rejected, corrrected_p_values = fdrcorrection(p_values, alpha=0.1)
    print("corredted p values", corrrected_p_values)

    inverse_p_values = 1 - corrrected_p_values
    inverse_p_image = widefield_utils.create_image_from_data(inverse_p_values, indicies, image_height, image_width)
    plt.title("Corrected Inverse P")
    plt.imshow(inverse_p_image, cmap="inferno", vmin=0, vmax=1)
    plt.colorbar()
    plt.show()



    corrected_effects = np.where(rejected == 1, t_stats, 0)
    corrected_effects_image = widefield_utils.create_image_from_data(corrected_effects, indicies, image_height, image_width)
    plt.title("Multiple Comparisons Correction")
    plt.imshow(corrected_effects_image, cmap=colourmap, vmin=-4, vmax=4)
    plt.colorbar()
    plt.show()



control_switching_only_nested = [


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
    r"NXAK7.1B/2021_02_28_Switching_Imaging",],
    #r"NXAK7.1B/2021_03_02_Switching_Imaging" - Falied Mousecam Check


]

# Set Directories
selected_session_list = control_switching_only_nested
data_root_diretory = r"/media/matthew/Expansion/Control_Data"
tensor_directory = r"/media/matthew/External_Harddrive_2/Control_Switching_Analysis/Full_Model"
analysis_name = "Full_Model"





"""
# Mutant Switching
mutant_switching_only_sessions_nested = [

    [r"NRXN71.2A/2020_12_13_Switching_Imaging",
    r"NRXN71.2A/2020_12_15_Switching_Imaging",
    r"NRXN71.2A/2020_12_17_Switching_Imaging"],

    [r"NXAK4.1A/2021_03_31_Switching_Imaging",
    r"NXAK4.1A/2021_04_02_Switching_Imaging",
    r"NXAK4.1A/2021_04_04_Switching_Imaging"],

    [r"NXAK10.1A/2021_05_20_Switching_Imaging",
    r"NXAK10.1A/2021_05_22_Switching_Imaging",
    r"NXAK10.1A/2021_05_24_Switching_Imaging"],

    [r"NXAK16.1B/2021_06_17_Switching_Imaging",
    r"NXAK16.1B/2021_06_19_Switching_Imaging",
    r"NXAK16.1B/2021_06_23_Switching_Imaging"],

    [r"NXAK20.1B/2021_11_15_Switching_Imaging",
    r"NXAK20.1B/2021_11_17_Switching_Imaging",
    r"NXAK20.1B/2021_11_19_Switching_Imaging"],

    [r"NXAK24.1C/2021_10_14_Switching_Imaging",
    r"NXAK24.1C/2021_10_20_Switching_Imaging",
    r"NXAK24.1C/2021_10_26_Switching_Imaging"],

]

selected_session_list = mutant_switching_only_sessions_nested
data_root_diretory = r"/media/matthew/External_Harddrive_1/Neurexin_Data"
tensor_directory = r"/media/matthew/External_Harddrive_2/Neurexin_Switching_Analysis/Full_Model"
analysis_name = "Full_Model"
"""


# Select Analysis Details
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)

condition_1_index = 1
condition_2_index = 3
baseline_window = list(range(0, 14)) # -2500 to - 2000
response_window = list(range(69, 97)) # 0 - 1000

get_individual_session_modulation(selected_session_list, tensor_directory, baseline_window, onset_files, start_window, stop_window, response_window, condition_1_index, condition_2_index)