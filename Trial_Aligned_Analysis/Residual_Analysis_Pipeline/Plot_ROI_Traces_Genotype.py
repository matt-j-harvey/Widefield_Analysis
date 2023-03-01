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



def plot_roi_trace_average(condition_1_data, condition_2_data, roi_list, roi_name, start_window, stop_window, save_directory, tensor_directory):

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


    #t_stats, p_values = stats.ttest_ind(a=condition_1_data, b=condition_2_data, axis=0)
    p_values = np.load(os.path.join(tensor_directory, "Mixed_Effects_Model_Results", roi_name + "_LME_P_Values.npy"))
    significance = np.where(p_values < 0.05, 1, 0)
    max_value = np.max([c1_upper_bound, c2_upper_bound])
    scaled_significance = np.multiply(significance, max_value) + (max_value * 0.05)
    #print("P Values", p_values)

    x_values = list(range(int(start_window), int(stop_window)))
    x_values = np.multiply(x_values, 36)

    plt.plot(x_values, condition_1_mean)
    plt.plot(x_values, condition_2_mean, c='g')

    plt.fill_between(x=x_values, y1=c1_lower_bound, y2=c1_upper_bound, alpha=0.3)
    plt.fill_between(x=x_values, y1=c2_lower_bound, y2=c2_upper_bound, alpha=0.3, color='g')
    plt.scatter(x_values, scaled_significance, c='gray', alpha=significance, marker='s')

    plt.axvline(0, color='k', linestyle='dashed')

    plt.xlabel("Time")
    plt.ylabel("Residual DF")

    plt.title(roi_name)
    plt.savefig(os.path.join(save_directory, roi_name + ".png"))
    plt.show()










def quantify_roi_activity_across_genotypes(tensor_directory, roi_list, roi_name, start_window, stop_window):

    """
    This Test Is Run Pixelwise - All Brains Must Be In Same Pixel Space
    """

    # Create Save Directory
    save_directory = os.path.join(tensor_directory, "Group_Average_Graphs")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Load Data
    condition_averages = np.load(os.path.join(tensor_directory, "Average_Activity", "Mouse_Genotype_Average_Matrix.npy"))
    print("Condition Averages", np.shape(condition_averages))

    condition_1_data = condition_averages[0]
    condition_2_data = condition_averages[1]
    print("Condition 1 Data", np.shape(condition_1_data))

    # Plot ROI Trace
    plot_roi_trace_average(condition_1_data, condition_2_data, roi_list, roi_name, start_window, stop_window, save_directory, tensor_directory)





#tensor_directory = r"/media/matthew/External_Harddrive_2/Regression_Modelling_Results/Genotype_Comparison_CR"
tensor_directory = r"/media/matthew/External_Harddrive_2/Regression_Modelling_Results/Genotype_Comparison_CR_All_Post"
analysis_name = "Correct_Rejections"

#tensor_directory = r"//media/matthew/External_Harddrive_2/Regression_Modelling_Results/Genotype_RT_Matched_Post_Learning"
#analysis_name = "Genotype_Vis_1_Comparison_Post"

#analysis_name = "Hits_Vis_1_Aligned_Post"
#tensor_directory = r"//media/matthew/External_Harddrive_2/Regression_Modelling_Results/Genotype_Comparisons_Hits_Post_Vis_Alinged"


[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)

roi_list = [ "m2_left",  "m2_right"]
roi_name = "M2"
quantify_roi_activity_across_genotypes(tensor_directory, roi_list, roi_name,  start_window, stop_window)



roi_list = [ "primary_visual_left",  "primary_visual_right"]
roi_name = "V1"
quantify_roi_activity_across_genotypes(tensor_directory, roi_list, roi_name,  start_window, stop_window)
