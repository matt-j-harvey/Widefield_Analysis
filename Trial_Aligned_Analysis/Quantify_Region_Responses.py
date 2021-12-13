import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
import os
import math
import scipy
import tables
from bisect import bisect_left
import cv2
from sklearn.decomposition import TruncatedSVD
from pathlib import Path
import joblib
from scipy import signal, ndimage, stats
from skimage.transform import resize
from scipy.interpolate import interp1d

def get_selected_pixels(selected_regions, pixel_assignments):

    # Get Pixels Within Selected Regions
    selected_pixels = []
    for region in selected_regions:
        region_mask = np.where(pixel_assignments == region, 1, 0)
        region_indicies = np.nonzero(region_mask)[0]
        for index in region_indicies:
            selected_pixels.append(index)
    selected_pixels.sort()

    return selected_pixels



def get_region_response(selected_pixels, activity_tensor, baseline_normalise=False):


    # Get Mean Response For Region For Each Trial
    region_pixel_responses = activity_tensor[:, :, selected_pixels]
    region_pixel_responses = np.nan_to_num(region_pixel_responses)
    region_responses = np.mean(region_pixel_responses, axis=2)
    region_responses = np.nan_to_num(region_responses)

    if baseline_normalise == False:
        return region_responses

    elif baseline_normalise == True:
        baseline = region_responses[:, 0: -1 * trial_start]
        baseline = np.mean(baseline, axis=1)

        number_of_trials = len(onsets)
        trial_length = trial_stop - trial_start
        normalised_region_responses = np.zeros((number_of_trials, trial_length))

        for timepoint in range(trial_length):
            timepoint_response = region_responses[:, timepoint]
            normalised_response = np.subtract(timepoint_response, baseline)
            normalised_region_responses[:, timepoint] = normalised_response

        return normalised_region_responses


def get_region_response_single_mouse(base_directory, activity_tensor_list, start_window, stop_window, condition_names, save_directory, baseline_normalise=False):

    v1 = [45, 46]
    pmv = [47, 48]
    amv = [39, 40]
    rsc = [32, 28]
    s1 = [21, 24]
    m2 = [8, 9]
    visual_cortex = v1 + pmv + amv

    region_index_list = [v1, pmv, amv, visual_cortex, rsc, s1, m2]
    region_name_list = ["V1", "PMV", "AMV", "Visual_Cortex", "RSC", "S1", "M2"]

    # Load Region Assignments
    pixel_assignments = np.load(os.path.join(base_directory, "Pixel_Assignmnets.npy"))

    # Quantify Responses For Each Region
    number_of_regions = len(region_index_list)
    for region_index in range(number_of_regions):

        region_numbers = region_index_list[region_index]
        region_name = region_name_list[region_index]
        selected_pixels = get_selected_pixels(region_numbers, pixel_assignments)

        print(pixel_assignments)
        condition_1_region_response = get_region_response(selected_pixels, activity_tensor_list[0], baseline_normalise=baseline_normalise)
        condition_2_region_response = get_region_response(selected_pixels, activity_tensor_list[1], baseline_normalise=baseline_normalise)

        """
        print("Region responses", np.shape(condition_1_region_response))
        plt.imshow(condition_1_region_response)
        plt.show()

        plt.imshow(condition_2_region_response)
        plt.show()
        """
        plot_individual_response(condition_1_region_response, condition_2_region_response, start_window, stop_window, condition_names, region_name, save_directory)


def plot_individual_response(condition_1_responses, condition_2_responses, start_window, stop_window, condition_names, region_name, save_directory):

    # Get X Values
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, 36)

    # Get Average Trace
    condition_1_average = np.mean(condition_1_responses, axis=0)
    condition_2_average = np.mean(condition_2_responses, axis=0)
    condition_1_average = np.nan_to_num(condition_1_average)
    condition_2_average = np.nan_to_num(condition_2_average)

    # Get SEM
    condition_1_sem = scipy.stats.sem(condition_1_responses, axis=0)
    condition_2_sem = scipy.stats.sem(condition_2_responses, axis=0)
    condition_1_sem = np.nan_to_num(condition_1_sem)
    condition_2_sem = np.nan_to_num(condition_2_sem)

    # Check For Significance
    t_stats, p_values = stats.ttest_ind(condition_1_responses, condition_2_responses)
    significant_points = []
    count = 0
    for timepoint in x_values:
        if p_values[count] < 0.05:
            significant_points.append(timepoint)
        count += 1
    max_value = np.max([np.add(condition_1_average, condition_1_sem), np.add(condition_2_average, condition_2_sem)])

    # Plot All This
    plt.plot(x_values, condition_1_average, c='b', marker='o', label=condition_names[0])
    plt.plot(x_values, condition_2_average, c='g', linestyle='dotted', marker='^', label=condition_names[1])
    plt.legend()

    plt.fill_between(x_values, np.add(condition_1_average, condition_1_sem), np.subtract(condition_1_average, condition_1_sem), alpha=0.1, color='b')
    plt.fill_between(x_values, np.add(condition_2_average, condition_2_sem), np.subtract(condition_2_average, condition_2_sem), alpha=0.1, color='g')

    plt.scatter(significant_points, np.ones(len(significant_points)) * max_value, c='k', marker='*')

    # Add Onset Line
    plt.axvline([0], c='k')

    # Set Plot Title
    plt.title(region_name)

    # Save Plot
    plt.savefig(os.path.join(save_directory, region_name + ".png"))
    plt.close()



def plot_group_reponse(trial_start, trial_stop, visual_responses, odour_responses, colour='b'):

    # Get X Values
    x_values = list(range(trial_start, trial_stop))
    x_values = np.multiply(x_values, 36)

    # Get Average Trace
    visual_average = np.mean(visual_responses, axis=0)
    odour_average = np.mean(odour_responses, axis=0)

    # Get STD
    visual_sd = scipy.stats.sem(visual_responses, axis=0)
    odour_sd = scipy.stats.sem(odour_responses, axis=0)

    plt.plot(x_values, visual_average, c=colour, marker='o')
    plt.plot(x_values, odour_average, c=colour, linestyle='dotted', marker='^')

    plt.fill_between(x_values, np.add(visual_average, visual_sd), np.subtract(visual_average, visual_sd), alpha=0.1, color=colour)
    plt.fill_between(x_values, np.add(odour_average, odour_sd), np.subtract(odour_average, odour_sd), alpha=0.1, color=colour)

    # Check For Significance
    t_stats, p_values = stats.ttest_ind(visual_responses, odour_responses)
    significant_points = []
    count = 0
    for timepoint in x_values:
        if p_values[count] < 0.05:
            significant_points.append(timepoint)
        count += 1

    max_value = np.max([np.add(visual_average, visual_sd), np.add(odour_average, odour_sd)])
    plt.scatter(significant_points, np.ones(len(significant_points))*max_value, c='k')

    # Add Onset Line
    plt.axvline([0], c='k')

    plt.show()

def get_session_name(base_directory):

    split_directory = base_directory.split('/')
    session_name = split_directory[-2] + "_" + split_directory[-1]
    return session_name

"""
visual_onset_file = ["Combined_Visual_Pre_Matched.npy"]
odour_onset_file = ["Combined_Odour_Pre_Matched.npy"]
plot_save_directory = "Pre_Stimuli_Mean_Visual_Cortex_Matched"


trial_start = -10
trial_stop = 40
visual_onset_file = ["visual_context_stable_vis_2_frame_onsets.npy"]
odour_onset_file = ["odour_context_stable_vis_2_frame_onsets.npy"]
plot_save_directory = "Vis_2_Response_Visual_Cortex_Matched"

x_values = list(range(trial_start, trial_stop))
x_values = np.multiply(x_values, 36)

# Get Responses
visual_control_responses = get_region_response(controls, visual_onset_file, trial_start, trial_stop, visual_cortex, baseline_normalise=True)
odour_control_responses = get_region_response(controls, odour_onset_file, trial_start, trial_stop,  visual_cortex, baseline_normalise=True)

# Plot Individual Responses
for control_index in range(len(controls)):
    base_directory = controls[control_index]
    session_name = get_session_name(base_directory)
    visual_responses = visual_control_responses[control_index]
    odour_responses = odour_control_responses[control_index]
    save_directory = plot_save_directory
    plot_individual_response(base_directory, trial_start, trial_stop, visual_responses, odour_responses, save_directory, plot_name=session_name)

visual_mutant_responses = get_region_response(mutants, visual_onset_file, trial_start, trial_stop,  visual_cortex, baseline_normalise=True)
odour_mutant_responses = get_region_response(mutants, odour_onset_file, trial_start, trial_stop,  visual_cortex, baseline_normalise=True)


# Plot Individual Responses
for mutant_index in range(len(mutants)):
    base_directory = mutants[mutant_index]
    session_name = get_session_name(base_directory)
    visual_responses = visual_mutant_responses[mutant_index]
    odour_responses = odour_mutant_responses[mutant_index]
    save_directory = plot_save_directory
    plot_individual_response(base_directory, trial_start, trial_stop, visual_responses, odour_responses, save_directory, plot_name=session_name,  colour='g')


plot_group_reponse(trial_start, trial_stop, visual_average_control_responses, odour_average_control_responses)
plot_group_reponse(trial_start, trial_stop, visual_average_mutant_responses, odour_average_mutant_responses, colour='g')



# Get Average Trace
visual_control_average = np.mean(visual_average_control_responses, axis=0)
visual_mutant_average = np.mean(visual_average_mutant_responses, axis=0)

olfactory_control_average = np.mean(odour_average_control_responses, axis=0)
olfactory_mutant_average = np.mean(odour_average_mutant_responses, axis=0)


# Get STD
visual_control_sd = scipy.stats.sem(visual_average_control_responses, axis=0)
visual_mutant_sd = scipy.stats.sem(visual_average_mutant_responses, axis=0)

olfactory_control_sd = scipy.stats.sem(odour_average_control_responses, axis=0)
olfactory_mutant_sd = scipy.stats.sem(odour_average_mutant_responses, axis=0)


plt.plot(x_values, visual_control_average, c='b')
plt.plot(x_values, olfactory_control_average, c='b', linestyle='dotted')

plt.plot(x_values, visual_mutant_average, c='g')
plt.plot(x_values, olfactory_mutant_average, c='g', linestyle='dotted')

plt.fill_between(x_values, np.add(visual_control_average, visual_control_sd), np.subtract(visual_control_average, visual_control_sd), alpha=0.1, color='b')
plt.fill_between(x_values, np.add(visual_mutant_average,  visual_mutant_sd), np.subtract( visual_mutant_average,  visual_mutant_sd),  alpha=0.1, color='g')

plt.fill_between(x_values, np.add(olfactory_control_average, olfactory_control_sd), np.subtract(olfactory_control_average, olfactory_control_sd), alpha=0.1, color='b')
plt.fill_between(x_values, np.add(olfactory_mutant_average,  olfactory_mutant_sd),  np.subtract(olfactory_mutant_average,  olfactory_mutant_sd), alpha=0.1, color='g')

plt.axvline([0], c='k')
plt.show()
#plt.vlines(10)

"""