import os

number_of_threads = 3
os.environ["OMP_NUM_THREADS"] = str(number_of_threads) # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = str(number_of_threads) # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = str(number_of_threads) # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(number_of_threads) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(number_of_threads) # export NUMEXPR_NUM_THREADS=1


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import fdrcorrection


from Files import Session_List
from Widefield_Utils import widefield_utils


import Create_Analysis_Dataset_Regression_Cross_Genotypes
import Extract_Average_Coefs_Genotype_Comparison
import Visualise_Average_Coefs_Genotype

#import Test_Significance_Mouse_Average_Coefs
#import Mixed_Effects_Modelling_Coefs


#Control Switching
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


# Mutant Switching
mutant_switching_only_nested = [

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

# Set Directories
#data_root_directory_list = [r"/media/matthew/Expansion/Control_Data", "/media/matthew/External_Harddrive_1/Neurexin_Data"]

tensor_directory_list = [r"/media/matthew/External_Harddrive_2/Control_Switching_Analysis/Full_Model",
                         r"/media/matthew/External_Harddrive_2/Neurexin_Switching_Analysis/Full_Model"]


output_directory = r"/media/matthew/External_Harddrive_2/Full_Pipeline_Results/Genotype_Comparison_Switching"
analysis_name = "Full_Model"

nested_session_list = [control_switching_only_nested,
                       mutant_switching_only_nested]

baseline_window = list(range(0, 14))


# Select Analysis Details
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)
print("start window", start_window)
print("Stop Window", stop_window)

# Create Analysis Dataset
#Create_Analysis_Dataset_Regression_Cross_Genotypes.create_analysis_dataset(tensor_directory_list, output_directory, nested_session_list, onset_files, analysis_name, start_window, stop_window)

# Extract Average Conditions
#Extract_Average_Coefs_Genotype_Comparison.extract_condition_averages(output_directory, analysis_name, baseline_correct=True, baseline_window=baseline_window)

# View Average Coefs
#Visualise_Average_Coefs_Genotype.view_average_difference(tensor_directory, 1, 3, "Unrewarded_Contextual_Modulation", delta_f_vmin=-0.02, delta_f_vmax=0.02, diff_scale_factor=0.5)
Visualise_Average_Coefs_Genotype.view_average_difference(output_directory, 1, "Vis Context Vis 2", start_window, stop_window,  vmin=-0, vmax=0.015, diff_magnitude=0.007, title_list=["Neurexin-1a KO", "Wildtype", "Difference"])
Visualise_Average_Coefs_Genotype.view_average_difference(output_directory, 3, "Odour Context Vis 2", start_window, stop_window,  vmin=-0, vmax=0.015, diff_magnitude=0.007)


"""
# View Average Differences
condition_1_index = 1
condition_2_index = 3
#Visualise_Average_Coefs.view_average_difference(tensor_directory, analysis_name, condition_1_index, condition_2_index, vmin=-0.02, vmax=0.02)
#Visualise_Average_Coefs.view_average_difference_per_mouse(tensor_directory, analysis_name, condition_1_index, condition_2_index, vmin=-0.02, vmax=0.02)

# Test Significance
baseline_start_index = 0
baseline_stop_index = 14

response_start_index = abs(start_window)
response_stop_index = response_start_index + 14

baseline_window = list(range(baseline_start_index, baseline_stop_index))
response_window = list(range(response_start_index, response_stop_index))

print("baseline Window", baseline_window)
print("REsponse window", response_window)

Test_Significance_Mouse_Average_Coefs.test_signficance_mouse_average_window_baseline_correct(tensor_directory, analysis_name, response_window, condition_1_index, condition_2_index, baseline_window)

p_value_tensor, slope_tensor, t_stat_tensor = Mixed_Effects_Modelling_Coefs.test_significance_window(tensor_directory, analysis_name, condition_1_index, condition_2_index, baseline_window, response_window)

# Save These Tensors
np.save(os.path.join(tensor_directory, analysis_name + "_p_value_tensor_raw.npy"), p_value_tensor)
np.save(os.path.join(tensor_directory, analysis_name + "_slope_tensor_raw.npy"), slope_tensor)
np.save(os.path.join(tensor_directory, analysis_name + "_t_stat_tensor_raw.npy"), t_stat_tensor)

p_value_tensor = np.load(os.path.join(tensor_directory, analysis_name + "_p_value_tensor_raw.npy"))
slope_tensor = np.load(os.path.join(tensor_directory, analysis_name + "_slope_tensor_raw.npy"))
t_tensor = np.load(os.path.join(tensor_directory, analysis_name + "_t_stat_tensor_raw.npy"))
rejected = np.where(p_value_tensor < 0.05, 1, 0)

# Multiple Comparisons Correction
rejected, p_value_tensor = fdrcorrection(p_value_tensor, alpha=0.1)
p_value_tensor = float(1) - p_value_tensor

# View Images
indicies, image_height, image_width = widefield_utils.load_tight_mask()
indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

effect_map = widefield_utils.create_image_from_data(t_tensor, indicies, image_height, image_width)
p_value_map = widefield_utils.create_image_from_data(p_value_tensor, indicies, image_height, image_width)
rejcted_map = widefield_utils.create_image_from_data(rejected, indicies, image_height, image_width)

figure_1 = plt.figure()
effect_axis = figure_1.add_subplot(1,3,1)
signficance_axis = figure_1.add_subplot(1,3,2)
rejected_axis = figure_1.add_subplot(1,3,3)

effect_magnitude = 3
effect_axis.imshow(-1*effect_map, vmin=-effect_magnitude, vmax=effect_magnitude, cmap=widefield_utils.get_musall_cmap())
signficance_axis.imshow(p_value_map, vmin=0.95, vmax=1)
rejected_axis.imshow(rejcted_map)
plt.show()
"""
