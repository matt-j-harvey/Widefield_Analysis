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



def get_average_response(session_list, onsets_file, save_directory):

    response_list = []

    trial_start = -10
    trial_stop = 40

    # Open Atlas Labels
    """
    atlas_labels = np.recfromcsv(r"/home/matthew/Documents/Allen_Atlas_Templates/Atlas_Labels.csv")
    print(atlas_labels)
    atlas_image = np.load(r"/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/Pixel_Assignmnets_Image.npy")
    plt.imshow(atlas_image)
    plt.show()
    """
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
        onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", onsets_file))

        # Create Trial Tensor
        activity_tensor = get_activity_tensor(delta_f_matrix, onsets, trial_start, trial_stop)

        # Get Selected Pixels
        selected_regions = [40, 39, 45, 46, 47, 48]
        #selected_regions = [45, 46]
        #selected_regions = [28, 30, 31]
        selected_pixels = []
        for region in selected_regions:
            region_mask = np.where(pixel_assignments == region, 1, 0)
            region_indicies = np.nonzero(region_mask)[0]
            for index in region_indicies:
                selected_pixels.append(index)
        selected_pixels.sort()

        # Get Mean Response For Region For Each Trial
        region_pixel_responses = activity_tensor[:, :, selected_pixels]
        region_pixel_responses = np.nan_to_num(region_pixel_responses)
        region_responses = np.mean(region_pixel_responses, axis=2)
        region_responses = np.nan_to_num(region_responses)

        # Normalise To Baseline
        region_pixel_baseline = region_responses[:, 0:np.abs(trial_start)]
        region_pixel_baseline = np.mean(region_pixel_baseline, axis=1)

        normalised_traces = []
        number_of_trials = np.shape(region_pixel_baseline)[0]
        for trial_index in range(number_of_trials):
            trial_baseline = region_pixel_baseline[trial_index]
            trial_response = region_responses[trial_index]
            normalised_response = np.subtract(trial_response, trial_baseline)
            normalised_traces.append(normalised_response)
        normalised_traces = np.array(normalised_traces)

        response_list.append(normalised_traces)

        """
        region_response_std = np.std(normalised_traces, axis=0)
        x_values = list(range(len(mean_region_response)))
        plt.plot(mean_region_response)
        plt.fill_between(x_values, np.add(mean_region_response, region_response_std), np.subtract(mean_region_response, region_response_std), alpha=0.5)
        plt.title(str(base_directory))
        plt.show()
        """


    return response_list
    #average_response = np.array(average_response)
    #average_response = np.mean(average_response, axis=0)
    #np.save(save_directory, average_response)







controls = ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN78.1D/2020_11_29_Switching_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN78.1A/2020_12_09_Switching_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/"]

mutants = ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging",
           "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/",
           "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging/"]

#"/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK12.1F/2021_09_22_Transition_Imaging",
all_mice = controls + mutants

visual_responses = get_average_response(controls, "visual_context_stable_vis_2_frame_onsets.npy", None)
odour_responses = get_average_response(controls, "odour_context_stable_vis_2_frame_onsets.npy", None)

print("Visual Response Shape", np.shape(visual_responses))
print("Odour Responses", np.shape(odour_responses))

visual_responses = np.vstack(visual_responses)
odour_responses = np.vstack(odour_responses)

print("Visual Response Shape", np.shape(visual_responses))
print("Odour Responses", np.shape(odour_responses))


visual_average = np.mean(visual_responses, axis=0)
odour_average = np.mean(odour_responses, axis=0)

visual_sem = scipy.stats.sem(visual_responses, axis=0)
odour_sem = scipy.stats.sem(odour_responses, axis=0)
x_values = list(range(len(visual_sem)))

plt.plot(visual_average, c='b')
plt.plot(odour_average, c='g')
plt.fill_between(x_values, np.add(visual_average, visual_sem), np.subtract(visual_average, visual_sem), alpha=0.1, color='b')
plt.fill_between(x_values, np.add(odour_average, odour_sem), np.subtract(odour_average,  odour_sem), alpha=0.1,  color='g')



#Get X Labeles
start = - 10
stop = 40

x_labels = []
label_interval = 6
for x in range(start, stop, label_interval):
    time = x * 36
    x_labels.append(time)

count = 0
x_indexes = []
for x_value in x_values:
    if count % label_interval == 0:
        x_indexes.append(x_value)
    count += 1


plt.xticks(x_indexes, x_labels)
plt.vlines([10], ymin=0, ymax=0.32, color='k', linestyles='dashed')
plt.show()
