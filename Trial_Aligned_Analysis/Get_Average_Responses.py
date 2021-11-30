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


def get_activity_tensor(activity_matrix, onsets, start_window, stop_window):

    number_of_pixels = np.shape(activity_matrix)[1]
    number_of_trials = np.shape(onsets)[0]
    number_of_timepoints = stop_window - start_window

    # Create Empty Tensor To Hold Data
    activity_tensor = np.zeros((number_of_trials, number_of_timepoints, number_of_pixels))

    # Get Correlation Matrix For Each Trial
    for trial_index in range(number_of_trials):

        # Get Trial Activity
        trial_start = onsets[trial_index] + start_window
        trial_stop = onsets[trial_index] + stop_window
        trial_activity = activity_matrix[trial_start:trial_stop]
        activity_tensor[trial_index] = trial_activity

    return activity_tensor


def get_average_response(session_list, onsets_file_list, trial_start, trial_stop, baseline_normalise=True):

    response_list = []

    # Open Atlas Labels
    atlas_labels = np.recfromcsv(r"/home/matthew/Documents/Allen_Atlas_Templates/Atlas_Labels.csv")
    #print(atlas_labels)

    #atlas_image = np.load(r"/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/Pixel_Assignmnets_Image.npy")
    #plt.imshow(atlas_image)
    #plt.show()

    for base_directory in session_list:
        print(base_directory)

        # Load Delta F Matrix
        delta_f_matrix_filepath = os.path.join(base_directory, "Delta_F.h5")
        delta_f_matrix_container = tables.open_file(delta_f_matrix_filepath, mode='r')
        delta_f_matrix = delta_f_matrix_container.root['Data']

        # Load Region Assigments
        pixel_assignments = np.load(os.path.join(base_directory, "Pixel_Assignmnets.npy"))
        print(pixel_assignments)

        # Load Onsets
        onsets = []
        for onsets_file in onsets_file_list:
            onsets_file_contents = np.load(os.path.join(base_directory, "Stimuli_Onsets", onsets_file))
            for onset in onsets_file_contents:
                onsets.append(onset)


        # Create Trial Tensor
        activity_tensor = get_activity_tensor(delta_f_matrix, onsets, trial_start, trial_stop)

        # Get Selected Pixels

        # Visual Cortex
        selected_regions = [45, 46, 47, 48]

        # Retrosplenial
        #selected_regions = [28, 30, 31]

        # Secondary Motor Cortex
        #selected_regions = [8, 9]

        selected_pixels = []
        for region in selected_regions:
            region_mask = np.where(pixel_assignments == region, 1, 0)
            region_indicies = np.nonzero(region_mask)[0]
            for index in region_indicies:
                selected_pixels.append(index)
        selected_pixels.sort()

        # Get Mean Response For Region For Each Trial
        region_pixel_responses = activity_tensor[:, :, selected_pixels]
        region_responses = np.mean(region_pixel_responses, axis=2)
        region_responses = np.nan_to_num(region_responses)

        if baseline_normalise == True:
            baseline = region_responses[:, 0: -1 * trial_start]
            baseline = np.mean(baseline, axis=1)

            number_of_trials = len(onsets)
            trial_length = trial_stop - trial_start
            normalised_region_responses = np.zeros((number_of_trials, trial_length))

            for timepoint in range(trial_length):
                timepoint_response = region_responses[:, timepoint]
                normalised_response = np.subtract(timepoint_response, baseline)
                normalised_region_responses[:, timepoint] = normalised_response

            region_responses = normalised_region_responses

        for response in region_responses:
            response_list.append(response)



    return response_list
    #average_response = np.array(average_response)
    #average_response = np.mean(average_response, axis=0)
    #np.save(save_directory, average_response)



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

    plt.plot(x_values, visual_average, c=colour)
    plt.plot(x_values, odour_average, c=colour, linestyle='dotted')

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



controls = ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN78.1A/2020_12_09_Switching_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN78.1D/2020_11_29_Switching_Imaging"]

mutants =  ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK12.1F/2021_09_22_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN71.2A/2020_12_17_Switching_Imaging"]



trial_start = -65
trial_stop = -4

x_values = list(range(trial_start, trial_stop))
x_values = np.multiply(x_values, 36)


#visual_onset_file = ["visual_context_stable_vis_1_frame_onsets.npy", "visual_context_stable_vis_2_frame_onsets.npy"]
#odour_onset_file = ["odour_context_stable_vis_1_frame_onsets.npy", "odour_context_stable_vis_2_frame_onsets.npy"]

visual_onset_file = ["Combined_Visual_Pre_Matched.npy"]
odour_onset_file = ["Combined_Odour_Pre_Matched.npy"]

#visual_onset_file = "visual_context_stable_vis_2_frame_onsets_Matched.npy"
#odour_onset_file = "odour_context_stable_vis_2_frame_onsets_Matched.npy"

visual_average_control_responses = get_average_response(controls, visual_onset_file, trial_start, trial_stop, baseline_normalise=False)
visual_average_mutant_responses = get_average_response(mutants, visual_onset_file, trial_start, trial_stop,  baseline_normalise=False)

odour_average_control_responses = get_average_response(controls, odour_onset_file, trial_start, trial_stop,  baseline_normalise=False)
odour_average_mutant_responses = get_average_response(mutants, odour_onset_file, trial_start, trial_stop,  baseline_normalise=False)

print("Visual Average Responses", np.shape(visual_average_control_responses))
print("Odour Average Responses", np.shape(odour_average_control_responses))

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

