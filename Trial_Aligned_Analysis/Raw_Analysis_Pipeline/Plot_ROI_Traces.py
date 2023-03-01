import numpy as np
import h5py
import tables
import matplotlib.pyplot as plt
from scipy import stats
from skimage.transform import resize
import os

from Widefield_Utils import widefield_utils
import ROI_Quantification_Functions



def get_std_bars(data):

    mean = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    upper_bound = np.add(mean, sd)
    lower_bound = np.subtract(mean, sd)

    return lower_bound, upper_bound


def get_sem_bars(data):

    mean = np.mean(data, axis=0)
    sd = stats.sem(data, axis=0)
    upper_bound = np.add(mean, sd)
    lower_bound = np.subtract(mean, sd)

    return lower_bound, upper_bound


def get_signficiance_points(condition_1_data, condition_2_data, x_values):


def plot_roi_trace_average(condition_1_data, condition_2_data, roi_list, roi_name, start_window, stop_window, save_directory):

    # Get ROI Pixels
    roi_pixel_indicies = ROI_Quantification_Functions.get_pooled_roi_from_list(roi_list)

    # Get ROI Data
    print("Condition 1 data", np.shape(condition_1_data))
    condition_1_data = condition_1_data[:, :, roi_pixel_indicies]
    condition_2_data = condition_2_data[:, :, roi_pixel_indicies]

    # Get Mean Within Region
    condition_1_data = np.mean(condition_1_data, axis=2)
    condition_2_data = np.mean(condition_2_data, axis=2)

    # Get Mean Across Trials
    condition_1_mean = np.mean(condition_1_data, axis=0)
    condition_2_mean = np.mean(condition_2_data, axis=0)

    print("Condition 1 mean", np.shape(condition_1_mean))
    print("Conditon 2 mean", np.shape(condition_2_mean))

    # Get SD
    c1_lower_bound, c1_upper_bound = get_sem_bars(condition_1_data)
    c2_lower_bound, c2_upper_bound = get_sem_bars(condition_2_data)



    #  Test
    print("T Test Cond 1 data", np.shape(condition_1_data))
    print("T Test Cond 2 data", np.shape(condition_2_data))
    t_stats, p_values = stats.ttest_rel(a=condition_1_data, b=condition_2_data, axis=0)
    significance = np.where(p_values < 0.05, 1, 0)
    max_value = np.max([c1_upper_bound, c2_upper_bound])
    significance = np.multiply(significance, max_value)
    print("P Values", p_values)



    plt.plot(x_values, condition_1_mean)
    plt.plot(x_values, condition_2_mean)

    plt.fill_between(x=x_values, y1=c1_lower_bound, y2=c1_upper_bound, alpha=0.3)
    plt.fill_between(x=x_values, y1=c2_lower_bound, y2=c2_upper_bound, alpha=0.3)
    plt.scatter(x_values, significance, c='k')

    plt.axvline(0, color='k', linestyle='dashed')

    plt.xlabel("Time")
    plt.ylabel("Residual DF")

    plt.title(roi_name)
    plt.savefig(os.path.join(save_directory, roi_name + ".png"))
    plt.show()


def plot_individual_traces(condition_1_data, condition_2_data, roi_list, roi_name, start_window, stop_window):

    # Get ROI Pixels
    roi_pixel_indicies = ROI_Quantification_Functions.get_pooled_roi_from_list(roi_list)

    # Get ROI Data
    print("Condition 1 data", np.shape(condition_1_data))
    condition_1_data = condition_1_data[:, :, roi_pixel_indicies]
    condition_2_data = condition_2_data[:, :, roi_pixel_indicies]

    # Get Mean Within Region
    condition_1_data = np.mean(condition_1_data, axis=2)
    condition_2_data = np.mean(condition_2_data, axis=2)

    # Get Mean Across Trials
    number_of_mice = np.shape(condition_1_data)[0]
    for mouse_index in range(number_of_mice):
        mouse_condition_1 = condition_1_data[mouse_index]
        mouse_condition_2 = condition_2_data[mouse_index]
        plt.plot(mouse_condition_1)
        plt.plot(mouse_condition_2)
        plt.show()





def plot_roi_trace_individual(condition_1_data, condition_2_data, roi_name):

    roi_pixel_indicies = ROI_Quantification_Functions.get_roi_pixels(roi_name)

    # Get ROI Data
    condition_1_data = condition_1_data[:, :, roi_pixel_indicies]
    condition_2_data = condition_2_data[:, :, roi_pixel_indicies]

    # Get Mean Within Region
    condition_1_data = np.mean(condition_1_data, axis=2)
    condition_2_data = np.mean(condition_2_data, axis=2)

    for trace in condition_1_data:
        plt.plot(trace, c='b')


    for trace in condition_2_data:
        plt.plot(trace, c='g')

    plt.title(roi_name)
    plt.show()






def quantify_roi_activity_n_mouse(tensor_directory, analysis_name, roi_list):

    # Open Analysis Dataframe
    analysis_file = tables.open_file(os.path.join(tensor_directory, analysis_name + "_Trialwise_.h5"), mode="r")
    activity_dataset = analysis_file.root["Data"]
    metadata_dataset = analysis_file.root["Trial_Details"]
    activity_dataset = np.array(activity_dataset)
    metadata_dataset = np.array(metadata_dataset)

    # Get Average Session Response Per Mouse
    condition_1_mouse_average_list, condition_2_mouse_average_list = Trial_Aligned_Utils.get_mouse_session_averages(activity_dataset, metadata_dataset)

    n_mice = len(condition_1_mouse_average_list)

    for mouse_index in range(n_mice):
        mouse_condition_1_data = np.array(condition_1_mouse_average_list[mouse_index])
        mouse_condition_2_data = np.array(condition_2_mouse_average_list[mouse_index])
        print("Condition 1 mouse data", np.shape(mouse_condition_1_data))

        plot_roi_trace_individual(mouse_condition_1_data, mouse_condition_2_data, "primary_visual_left")
        plot_roi_trace_average(mouse_condition_1_data, mouse_condition_2_data, "primary_visual_left")



def baseline_correct(condition_data, baseline_window):

    # Shape N Mice, N Timepoints, N Pixels
    number_of_mice, number_of_timepoints, number_of_pixels = np.shape(condition_data)

    baseline_corrected_data = []
    for mouse_index in range(number_of_mice):
        mouse_activity = condition_data[mouse_index]
        mouse_baseline = mouse_activity[baseline_window]
        mouse_baseline = np.mean(mouse_baseline, axis=0)
        print("Mouse baseline", np.shape(mouse_baseline))
        mouse_activity = np.subtract(mouse_activity, mouse_baseline)
        baseline_corrected_data.append(mouse_activity)

    baseline_corrected_data = np.array(baseline_corrected_data)
    print("Baseline Corrected Data", np.shape(baseline_corrected_data))

    return baseline_corrected_data



def quantify_roi_activity_across_conditions(tensor_directory, condition_1_index, condition_2_index, roi_list, roi_name, start_window, stop_window):

    """
    This Test Is Run Pixelwise - All Brains Must Be In Same Pixel Space
    """

    # Create Save Directory
    save_directory = os.path.join(tensor_directory, "Group_Average_Graphs")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Load Data
    condition_averages = np.load(os.path.join(tensor_directory, "Average_Activity", "Mouse_Condition_Average_Matrix.npy"), allow_pickle=True)
    print("Condition Averafes", np.shape(condition_averages))

    condition_1_data = condition_averages[:, condition_1_index]
    condition_2_data = condition_averages[:, condition_2_index]

    print("Loadings")
    print("Condition 1 Data", np.shape(condition_1_data))

    # Plot ROI Trace
    plot_roi_trace_average(condition_1_data, condition_2_data, roi_list, roi_name, start_window, stop_window, save_directory)

    plot_individual_traces(condition_1_data, condition_2_data, roi_list, roi_name, start_window, stop_window)



def quantify_roi_activity_across_conditions_individuals(tensor_directory, condition_1_index, condition_2_index, roi_list, roi_name, start_window, stop_window):

    """
    This Test Is Run Pixelwise - All Brains Must Be In Same Pixel Space
    """

    # Create Save Directory
    save_directory = os.path.join(tensor_directory, "Individual_Graphs")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Load Data
    condition_averages = np.load(os.path.join(tensor_directory, "Average_Coefs", "Mouse_Condition_Average_Matrix.npy"))
    condition_1_data = condition_averages[:, condition_1_index]
    condition_2_data = condition_averages[:, condition_2_index]

    # Plot ROI Trace
    plot_individual_traces(condition_1_data, condition_2_data, roi_list, roi_name, start_window, stop_window)





#tensor_directory = r"/media/matthew/External_Harddrive_2/Control_Switching_Analysis/Full_Model"
#tensor_directory = r"/media/matthew/External_Harddrive_2/Neurexin_Switching_Analysis/Full_Model"
#tensor_directory = r"/media/matthew/External_Harddrive_2/Regression_Modelling_Results/Control_Switching_Residual_Only"
tensor_directory = r"/media/matthew/External_Harddrive_2/Regression_Modelling_Results/Mutant_Switching_Residual_Only"
analysis_name = "Full_Model"
condition_1_index = 1
condition_2_index = 3


tensor_directory = r"/media/matthew/External_Harddrive_2/Regression_Modelling_Results/Neurexin_Learning_Full_Prestim"
#tensor_directory = r"/media/matthew/External_Harddrive_2/Raw_Pipeline_Results/Neurexin_Learning_Vis_1_Full_Prestim"

analysis_name = "Hits_Vis_1_Aligned_Post"
condition_1_index = 0
condition_2_index = 1

#tensor_directory = r"/media/matthew/External_Harddrive_2/Regression_Modelling_Results/Control_Learning_Full_Prestim"


[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)

roi_list = [ "m2_left",  "m2_right"]
roi_name = "M2"
quantify_roi_activity_across_conditions(tensor_directory, condition_1_index, condition_2_index, roi_list, roi_name,  start_window, stop_window)


roi_list = [ "primary_visual_left",  "primary_visual_right"]
roi_name = "V1"
quantify_roi_activity_across_conditions(tensor_directory, condition_1_index, condition_2_index, roi_list, roi_name, start_window, stop_window)
