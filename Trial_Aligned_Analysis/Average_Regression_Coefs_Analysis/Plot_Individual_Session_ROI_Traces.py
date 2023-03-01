import numpy as np
import h5py
import tables
import matplotlib.pyplot as plt
from scipy import stats
from skimage.transform import resize
import os

from Widefield_Utils import widefield_utils
import ROI_Quantification_Functions


def get_roi_mean(session_trace, roi_pixels):
    roi_activity = session_trace[:, roi_pixels]
    roi_mean = np.mean(roi_activity, axis=1)
    return roi_mean


def baseline_correct_trace(trace, baseline_window):
    baseline_values = trace[baseline_window]
    baseline_mean = np.mean(baseline_values)
    trace = np.subtract(trace, baseline_mean)
    return trace

def plot_individual_mice_coefs(session_list, tensor_directory, start_window, stop_window, onset_files, roi_list, roi_name, baseline_window):


    # Create Save Directory
    save_directory = os.path.join(tensor_directory, "Individual_Session_Plots")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Get Regression Dict Structure
    number_of_regressors = len(onset_files)
    trial_length = stop_window - start_window
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, 36)

    # Get ROI Pixels
    roi_pixels = ROI_Quantification_Functions.get_pooled_roi_from_list(roi_list)

    for mouse in session_list:
        for session in mouse:

            # Load Regressor Dict
            full_tensor_directory = os.path.join(tensor_directory, session)
            regressor_dict = np.load(os.path.join(full_tensor_directory, "Regression_Dictionary_Simple.npy"), allow_pickle=True)[()]
            regression_coefs = regressor_dict['Coefs']
            regression_coefs = np.transpose(regression_coefs)

            print(session)

            for condition_index in range(number_of_regressors):

                # Get Stimuli Regressor For Each Condition
                condition_regressor_start = condition_index * trial_length
                condition_regressor_stop = condition_regressor_start + trial_length
                condition_regressors = regression_coefs[condition_regressor_start:condition_regressor_stop]

                # Get ROI Mean
                roi_mean = get_roi_mean(condition_regressors, roi_pixels)

                # Baseline Correct Trace
                roi_mean = baseline_correct_trace(roi_mean, baseline_window)

                # Baseline Correct
                condition_name = onset_files[condition_index].replace("onsets", "")
                condition_name = condition_name.replace(".npy", "")
                plt.plot(x_values, roi_mean, label=condition_name)

            print(session)
            session_name = session.replace("/", "_")
            plt.title(session_name + "_" + roi_name)
            plt.legend()
            plt.savefig(os.path.join(save_directory, session_name + "_" + roi_name + ".png"))
            plt.close()







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
tensor_directory = r"/media/matthew/External_Harddrive_2/Neurexin_Switching_Analysis/Full_Model"
analysis_name = "Full_Model"
nested_session_list = [selected_session_list]
baseline_window = list(range(0, 14))

# Select Analysis Details
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)

roi_list = ["m2_left",  "m2_right"]
roi_name = "M2"
plot_individual_mice_coefs(selected_session_list, tensor_directory, start_window, stop_window, onset_files, roi_list, roi_name, baseline_window)


roi_list = [ "primary_visual_left",  "primary_visual_right"]
roi_name = "V1"
plot_individual_mice_coefs(selected_session_list, tensor_directory, start_window, stop_window, onset_files, roi_list, roi_name, baseline_window)
