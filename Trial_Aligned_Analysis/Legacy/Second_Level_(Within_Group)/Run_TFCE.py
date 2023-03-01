import math

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import statsmodels.stats.multitest
from tqdm import tqdm
from scipy import stats, ndimage
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import fdrcorrection
import mne
from datetime import datetime

from Files import Session_List
from Widefield_Utils import widefield_utils, Create_Activity_Tensor, Create_Video_From_Tensor



def paired_cluster_signficance_test(activity_tensor_list):

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()

    # Convert To Diffference Tensor For 1 Sample T Test
    modulation_list = np.subtract(activity_tensor_list[0], activity_tensor_list[1])

    # Convert To Array
    modulation_list = np.array(modulation_list)
    print("Modulation list shape", np.shape(modulation_list))
    threshold_tfce = dict(start=0, step=0.2)
    number_of_timepoints = np.shape(modulation_list)[1]

    mean_modulation = np.sum(modulation_list[:, 10:52], axis=1)
    print("Mean modulation shape", np.shape(mean_modulation))

    F_obs, clusters, cluster_pvs, H0 = mne.stats.permutation_cluster_1samp_test(X=mean_modulation, threshold=threshold_tfce, out_type='mask', n_permutations=1024)

    # Reshape P Values
    p_map = np.reshape(cluster_pvs, (image_height, image_width))


    return p_map


def view_p_maps(p_values):

    # Remove NaNs
    p_values = np.nan_to_num(p_values)

    # Create Axes
    figure_1 = plt.figure()
    p_axis = figure_1.add_subplot(1,1,1)

    # Inverse
    p_frame = 1 - p_values

    # Plot
    image_handle = p_axis.imshow(p_frame, vmin=0.95, vmax=1, cmap='inferno')
    figure_1.colorbar(image_handle)

    plt.show()



### Correct Rejections Post Learning ###
analysis_name = "Unrewarded_Contextual_Modulation"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)
tensor_save_directory = r"//media/matthew/Expansion/Widefield_Analysis/Learning_Analysis"

# Intermediate Significance Resting Folder
signficance_testing_folder = r"/media/matthew/29D46574463D2856/Significance_Testing"

# Load Session List
nested_session_list = Session_List.control_switching_nested
session_list = Session_List.control_switching_sessions


#mean_mouse_tensor = np.load(os.path.join(signficance_testing_folder, "Controls_Contextual_Modulation_Mouse_Average.npy"))
mean_session_tensor = np.load(os.path.join(signficance_testing_folder, "Controls_Contextual_Modulation_Session_Average.npy"))

print("mean mouse tensor", np.shape(mean_session_tensor))

p_map = paired_cluster_signficance_test(mean_session_tensor)
np.save("/media/matthew/29D46574463D2856/Significance_Testing/Control_Context/TFCE/p_map_session.npy", p_map)

p_map = np.load("/media/matthew/29D46574463D2856/Significance_Testing/Control_Context/TFCE/p_map_session.npy")
view_p_maps(p_map)