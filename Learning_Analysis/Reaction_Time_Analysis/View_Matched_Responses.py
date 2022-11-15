import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec
import numpy as np
import tables
import os
import pandas as pd
from scipy import ndimage
from skimage.transform import resize
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from scipy import stats
import RT_Strat_Utils



def correct_baseline(activity_tensor, trial_start):

    corrected_tensor = []
    for trial in activity_tensor:
        trial_baseline = trial[0: -1 * trial_start]
        trial_baseline = np.mean(trial_baseline, axis=0)
        trial = np.subtract(trial, trial_baseline)
        corrected_tensor.append(trial)
    corrected_tensor = np.array(corrected_tensor)

    return corrected_tensor



def view_matched_responses(turple_list, tensor_name, save_directory, subtract_baseline=True, trial_start=-10):

    # Load Mask
    indicies, image_height, image_width = RT_Strat_Utils.load_tight_mask()

    # Load Data
    pre_learning_list = []
    post_learning_list = []
    t_stats_list = []

    for session_tuple in tqdm(turple_list):

        # Load Each Tensor
        condition_1_activity = np.load(os.path.join(session_tuple[0], tensor_name))
        if subtract_baseline == True:
            condition_1_activity = correct_baseline(condition_1_activity, trial_start)


        condition_2_activity = np.load(os.path.join(session_tuple[1], tensor_name))
        if subtract_baseline == True:
            condition_2_activity = correct_baseline(condition_2_activity, trial_start)


        t_stats, p_values = stats.ttest_ind(condition_1_activity, condition_2_activity, axis=0)

        # Threshold T Stat By P Value
        t_stats = np.where(p_values < 0.05, t_stats, 0)

        condition_1_activity = np.mean(condition_1_activity, axis=0)
        condition_2_activity = np.mean(condition_2_activity, axis=0)

        t_stats_list.append(t_stats)
        pre_learning_list.append(condition_1_activity)
        post_learning_list.append(condition_2_activity)

    # Plot Data
    number_of_sessions = len(pre_learning_list)
    number_of_timepoints = np.shape(pre_learning_list[0])[0]

    figure_1 = plt.figure(figsize=(20,60))
    gridspec_1 = GridSpec(nrows=4, ncols=number_of_sessions)

    vmin = 0
    vmax = 10000
    diff_cmap = RT_Strat_Utils.get_mussal_cmap()

    # PLot Each Timepoint
    for timepoint in range(number_of_timepoints):

        for session_index in range(number_of_sessions):
            condition_1_axis = figure_1.add_subplot(gridspec_1[0, session_index])
            condition_2_axis = figure_1.add_subplot(gridspec_1[1, session_index])
            difference_axis = figure_1.add_subplot(gridspec_1[2, session_index])
            signficance_axis = figure_1.add_subplot(gridspec_1[3, session_index])

            condition_1_data = pre_learning_list[session_index][timepoint]
            condition_2_data = post_learning_list[session_index][timepoint]
            difference = np.subtract(condition_1_data, condition_2_data)

            condition_1_image = RT_Strat_Utils.create_image_from_data(condition_1_data, indicies, image_height, image_width)
            condition_2_image = RT_Strat_Utils.create_image_from_data(condition_2_data, indicies, image_height, image_width)
            diff_image = RT_Strat_Utils.create_image_from_data(difference, indicies, image_height, image_width)
            significance_image = RT_Strat_Utils.create_image_from_data(t_stats_list[session_index][timepoint], indicies, image_height, image_width)

            condition_1_axis.imshow(condition_1_image, vmax=vmax, vmin=vmin)
            condition_2_axis.imshow(condition_2_image, vmax=vmax, vmin=vmin)
            difference_axis.imshow(diff_image, vmax=0.5 * vmax, vmin=-0.5 * vmax, cmap=diff_cmap)
            signficance_axis.imshow(significance_image, vmin=-6, vmax=6, cmap=diff_cmap)

            condition_1_axis.axis('off')
            condition_2_axis.axis('off')
            difference_axis.axis('off')
            signficance_axis.axis('off')

        figure_1.suptitle(str(timepoint))

        plt.draw()
        plt.savefig(os.path.join(save_directory, str(timepoint).zfill(4) + ".png"))
        plt.clf()




control_session_tuples = [

    [r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits/NRXN78.1A/2020_11_15_Discrimination_Imaging",
     r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits/NRXN78.1A/2020_11_24_Discrimination_Imaging",],

    [r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits/NRXN78.1D/2020_11_15_Discrimination_Imaging",
     r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits/NRXN78.1D/2020_11_25_Discrimination_Imaging"],

    [r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits/NXAK4.1B/2021_02_06_Discrimination_Imaging",
     r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits/NXAK4.1B/2021_02_22_Discrimination_Imaging"],

    [r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits/NXAK7.1B/2021_02_03_Discrimination_Imaging",
     r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits/NXAK7.1B/2021_02_24_Discrimination_Imaging"],

    [r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits/NXAK14.1A/2021_05_01_Discrimination_Imaging",
     r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits/NXAK14.1A/2021_05_09_Discrimination_Imaging"],

    [r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits/NXAK22.1A/2021_09_29_Discrimination_Imaging",
     r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits/NXAK22.1A/2021_10_08_Discrimination_Imaging"]

]

mutant_session_tuples = [
    [r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits/NRXN71.2A/2020_11_14_Discrimination_Imaging",
     r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits/NRXN71.2A/2020_12_09_Discrimination_Imaging"],

    [r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits/NXAK4.1A/2021_02_04_Discrimination_Imaging",
     r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits/NXAK4.1A/2021_03_05_Discrimination_Imaging"],

    [r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits/NXAK10.1A/2021_05_02_Discrimination_Imaging",
     r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits/NXAK10.1A/2021_05_14_Discrimination_Imaging"],

    [r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits/NXAK16.1B/2021_05_02_Discrimination_Imaging",
     r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits/NXAK16.1B/2021_06_15_Discrimination_Imaging"],

    [r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits/NXAK20.1B/2021_09_30_Discrimination_Imaging",
     r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits/NXAK20.1B/2021_10_19_Discrimination_Imaging"],

    [r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits/NXAK24.1C/2021_09_22_Discrimination_Imaging",
     r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits/NXAK24.1C/2021_10_08_Discrimination_Imaging"],
    ]


tensor_name = "Hits_RT_Matched_Activity_Tensor.npy"
"""
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits/Mutants"
view_matched_responses(mutant_session_tuples, tensor_name, save_directory)
"""

save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Pre_Post_Hits/Controls"
view_matched_responses(control_session_tuples, tensor_name, save_directory)
