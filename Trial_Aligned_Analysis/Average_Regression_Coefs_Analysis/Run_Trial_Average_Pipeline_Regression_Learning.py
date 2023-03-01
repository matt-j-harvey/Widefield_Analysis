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


import Create_Analysis_Dataset_Regression_Learning
import Test_Significance_Mouse_Average_Coefs
import Visualise_Average_Coefs
import Mixed_Effects_Modelling_Coefs




# Mutant Learning
mutant_session_tuples = [

    [   # 71.2A
        [r"NRXN71.2A/2020_11_14_Discrimination_Imaging"],
        [r"NRXN71.2A/2020_12_09_Discrimination_Imaging"],
    ],


    [   # 4.1a
        [r"NXAK4.1A/2021_02_04_Discrimination_Imaging"],
        [r"NXAK4.1A/2021_03_05_Discrimination_Imaging"],
    ],


    [   # 10.1a
        [r"NXAK10.1A/2021_05_02_Discrimination_Imaging"],
        [r"NXAK10.1A/2021_05_14_Discrimination_Imaging"],
    ],


    [   # 16.1b
        [r"NXAK16.1B/2021_05_02_Discrimination_Imaging"],
        [r"NXAK16.1B/2021_06_15_Discrimination_Imaging"],
    ],


    [   # 20.1b
        [r"NXAK20.1B/2021_09_30_Discrimination_Imaging"],
        [r"NXAK20.1B/2021_10_19_Discrimination_Imaging"],
    ],


    [   # 24.1c
        [r"NXAK24.1C/2021_09_22_Discrimination_Imaging"],
        [r"NXAK24.1C/2021_10_08_Discrimination_Imaging"],
    ]

]


selected_session_list = mutant_session_tuples
data_root_diretory = r"/media/matthew/External_Harddrive_1/Neurexin_Data"
tensor_directory = r"/media/matthew/External_Harddrive_2/Neurexin_Learning_Analysis/Full_Model"
analysis_name = "Hits_Vis_1_Aligned"


selected_session_list = mutant_session_tuples
data_root_diretory = r"/media/matthew/External_Harddrive_1/Neurexin_Data"
tensor_directory = r"//media/matthew/External_Harddrive_2/Regression_Modelling_Results/Control_Learning_Lick_Aligned"
analysis_name = "Hits_Lick_Aligned"


# Must Nest Session List Into Format - Group - Mouse - Session
nested_session_list = [selected_session_list]

# Load Analysis Details
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)
print("start window", start_window)
print("Stop Window", stop_window)

# Create Analysis Dataset
Create_Analysis_Dataset_Regression_Learning.create_analysis_dataset(tensor_directory, nested_session_list, onset_files, analysis_name, start_window, stop_window)

# View Average Differences
#Visualise_Average_Coefs.view_average_difference(tensor_directory, analysis_name, condition_1_index, condition_2_index, vmin=-0.02, vmax=0.02)
Visualise_Average_Coefs.view_average_difference_per_mouse_learning(tensor_directory, analysis_name, vmin=-0.05, vmax=0.05)

# View Average Differences
response_start_index = abs(start_window)
response_stop_index = response_start_index + 24
response_window = list(range(response_start_index, response_stop_index))
print("Response Window", response_window)
Test_Significance_Mouse_Average_Coefs.test_signficance_mouse_average_learning(tensor_directory, analysis_name, response_window)


"""
# Test Significance
baseline_start_index = abs(start_window)-14
baseline_stop_index = abs(start_window)



baseline_window = list(range(baseline_start_index, baseline_stop_index))
response_window = list(range(response_start_index, response_stop_index))

print("baseline Window", baseline_window)
print("REsponse window", response_window)

#Test_Significance_Mouse_Average_Coefs.test_signficance_mouse_average_window_baseline_correct(tensor_directory, analysis_name, response_window, condition_1_index, condition_2_index, baseline_window)

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