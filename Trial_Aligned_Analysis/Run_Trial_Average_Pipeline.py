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


import Create_Trial_Tensors
import Create_Analysis_Dataset
import Test_Significance_Mouse_Average
import Visualise_Trial_Average
import Mixed_Effects_Modelling_Session_Average

# Select Sessions
selected_session_list = [

    [r"NRXN78.1A/2020_12_05_Switching_Imaging",
    r"NRXN78.1A/2020_12_09_Switching_Imaging"],

    [r"NRXN78.1D/2020_11_29_Switching_Imaging",
    r"NRXN78.1D/2020_12_05_Switching_Imaging"],

    [r"NXAK14.1A/2021_05_21_Switching_Imaging",
    r"NXAK14.1A/2021_05_23_Switching_Imaging",
    r"NXAK14.1A/2021_06_11_Switching_Imaging",
    r"NXAK14.1A/2021_06_13_Transition_Imaging",
    r"NXAK14.1A/2021_06_15_Transition_Imaging",
    r"NXAK14.1A/2021_06_17_Transition_Imaging"],

    [r"NXAK22.1A/2021_10_14_Switching_Imaging",
    r"NXAK22.1A/2021_10_20_Switching_Imaging",
    r"NXAK22.1A/2021_10_22_Switching_Imaging",
    r"NXAK22.1A/2021_10_29_Transition_Imaging",
    r"NXAK22.1A/2021_11_03_Transition_Imaging",
    r"NXAK22.1A/2021_11_05_Transition_Imaging"],

    [r"NXAK4.1B/2021_03_02_Switching_Imaging",
    r"NXAK4.1B/2021_03_04_Switching_Imaging",
    r"NXAK4.1B/2021_03_06_Switching_Imaging",
    #r"NXAK4.1B/2021_04_02_Transition_Imaging", No Regression
    r"NXAK4.1B/2021_04_08_Transition_Imaging",
    r"NXAK4.1B/2021_04_10_Transition_Imaging"],

    [r"NXAK7.1B/2021_02_26_Switching_Imaging",
    r"NXAK7.1B/2021_02_28_Switching_Imaging",
    r"NXAK7.1B/2021_03_02_Switching_Imaging",
    r"NXAK7.1B/2021_03_23_Transition_Imaging",
    r"NXAK7.1B/2021_03_31_Transition_Imaging",
    #r"NXAK7.1B/2021_04_02_Transition_Imaging", No Regression
    ],

]

# Create Analysis Dataset
# Must Nest Session List Into Format - Group - Mouse - Session
print("create analysis dataset")
nested_session_list = [selected_session_list]

# Set Tensor Directory
data_root_directory = r"/media/matthew/Expansion/Control_Data"
tensor_directory = r"/media/matthew/External_Harddrive_2/Control_Switching_Analysis/Residual_Only"

# Select Analysis Details
analysis_name = "Unrewarded_Contextual_Modulation"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)
print("start window", start_window)
"""
# Create Trial Tensors
print("creating trial tensors")
for mouse in tqdm(selected_session_list, leave=True, position=0, desc="Mouse"):
    for base_directory in tqdm(mouse, leave=True, position=1, desc="Session"):
        for onsets_file in tqdm(onset_files, leave=False, position=2, desc="Condition"):
            Create_Trial_Tensors.create_trial_tensor(os.path.join(data_root_directory, base_directory), onsets_file, start_window, stop_window, tensor_directory,
                                start_cutoff=3000,
                                ridge_regression_correct=True,
                                gaussian_filter=False,
                                baseline_correct=True,
                                align_within_mice=False,
                                align_across_mice=False,
                                extended_tensor=False,
                                mean_only=False,
                                stop_stimuli=None,
                                use_100_df=True)

Create_Analysis_Dataset.create_analysis_dataset(tensor_directory, nested_session_list, onset_files, analysis_name, start_window, stop_window)

#Visualise_Trial_Average.view_raw_difference(tensor_directory, analysis_name, vmin=-0.05, vmax=0.05)
"""

# Get T Map
print("creating t map")
window = list(range(14, 42))
#p_value_tensor, slope_tensor, t_stat_tensor = Mixed_Effects_Modelling_Session_Average.test_significance_window(tensor_directory, analysis_name, window, random_effects="mouse_and_session")
Test_Significance_Mouse_Average.test_signficance_mouse_average_window(tensor_directory, analysis_name, window)
#Test_Significance_Mixed_Effects_Model.test_significance_window_session_average(tensor_directory, analysis_name)
p_value_tensor = np.nan_to_num(p_value_tensor)

# Save These Tensors
np.save(os.path.join(tensor_directory, analysis_name + "_p_value_tensor_raw.npy"), p_value_tensor)
np.save(os.path.join(tensor_directory, analysis_name + "_slope_tensor_raw.npy"), slope_tensor)
np.save(os.path.join(tensor_directory, analysis_name + "_t_stat_tensor_raw.npy"), t_stat_tensor)

p_value_tensor = np.load(os.path.join(tensor_directory, analysis_name + "_p_value_tensor_raw.npy"))
slope_tensor = np.load(os.path.join(tensor_directory, analysis_name + "_slope_tensor_raw.npy"))
t_tensor = np.load(os.path.join(tensor_directory, analysis_name + "_t_stat_tensor_raw.npy"))
rejected = np.where(p_value_tensor < 0.05, 1, 0)

# Multiple Comparisons Correction
rejected, p_value_tensor = fdrcorrection(p_value_tensor, alpha=0.05)
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
