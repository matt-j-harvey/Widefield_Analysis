import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy import stats

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions


def match_trials(base_directory, visual_onsets_file, odour_onsets_file, trial_start_window, trial_stop_window, axis):

    # Load Onsets
    #visual_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", visual_onsets_file + ".npy"))
    #odour_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", odour_onsets_file + ".npy"))
    visual_onsets = get_onsets(base_directory, visual_onsets_file)
    odour_onsets = get_onsets(base_directory, odour_onsets_file)
    frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    frame_times = Widefield_General_Functions.invert_dictionary(frame_times)

    # Load AI Recorder File
    ai_filename = Widefield_General_Functions.get_ai_filename(base_directory)
    ai_data = Widefield_General_Functions.load_ai_recorder_file(base_directory + "/" + ai_filename)
    running_trace = ai_data[8]

    # Get Condition Running Traces
    visual_running_traces = get_condition_running_traces(running_trace, visual_onsets, frame_times, trial_start_window,trial_stop_window)
    odour_running_traces = get_condition_running_traces(running_trace, odour_onsets, frame_times, trial_start_window,trial_stop_window)

    # Get Mean Vectors
    visual_running_means = np.mean(visual_running_traces, axis=1)
    odour_running_means = np.mean(odour_running_traces, axis=1)

    # Sort These Vectors
    sorted_visual_means = np.copy(visual_running_means)
    sorted_visual_means = list(sorted_visual_means)
    sorted_visual_means.sort(reverse=True)

    sorted_odour_means = np.copy(odour_running_means)
    sorted_odour_means = list(sorted_odour_means)
    sorted_odour_means.sort()

    print("Sorted Visual Means", sorted_visual_means)
    print("Sorted Odour Means", sorted_odour_means)

    visual_running_means = list(visual_running_means)
    odour_running_means = list(odour_running_means)

    matched_visual_trial_indexes = []
    matched_odour_trial_indexes = []

    convergence = False
    count = 0
    while convergence == False:
        visual_mean = sorted_visual_means[count]
        odour_mean = sorted_odour_means[count]
        count += 1

        if visual_mean < odour_mean:
            convergence = True

        else:
            visual_trial_index = visual_running_means.index(visual_mean)
            odour_trial_index = odour_running_means.index(odour_mean)
            matched_visual_trial_indexes.append(visual_trial_index)
            matched_odour_trial_indexes.append(odour_trial_index)

    # Get Matched Subsets
    print("Matched Visual Trials", matched_visual_trial_indexes)
    print("Matched Odour Trials", matched_odour_trial_indexes)

    matched_visual_trials = visual_running_traces[matched_visual_trial_indexes]
    matched_odour_trials = odour_running_traces[matched_odour_trial_indexes]

    mean_matched_visual_trace = np.mean(matched_visual_trials, axis=0)
    mean_matched_odour_trace = np.mean(matched_odour_trials, axis=0)


    # Plot These

    # Get X Values
    visual_timepoints = np.shape(visual_running_traces)[1]
    odour_timepoints = np.shape(odour_running_traces)[1]

    visual_x_values = list(range(visual_timepoints))
    odour_x_values = list(range(odour_timepoints))

    visual_x_values = np.add(visual_x_values, 36 * trial_start)
    odour_x_values = np.add(odour_x_values, 36 * trial_start)

    axis.plot(visual_x_values, mean_matched_visual_trace, c='b')
    axis.plot(odour_x_values, mean_matched_odour_trace, c='r')
    #axis.axvline([0], c='k')

    plot_title = base_directory.split('/')[-3]
    axis.set_title(plot_title)

    # Convert Indexes To Onsets
    matched_visual_onsets = []
    matched_odour_onsets = []
    for trial_index in matched_visual_trial_indexes:
        onset = visual_onsets[trial_index]
        matched_visual_onsets.append(onset)

    for trial_index in matched_odour_trial_indexes:
        onset = odour_onsets[trial_index]
        matched_odour_onsets.append(onset)

    return matched_visual_onsets, matched_odour_onsets





def get_condition_running_traces(running_trace, onets, frame_times, trial_start_window, trial_stop_window):

    condition_running_traces = []

    # Get Realtime Start and Stops
    realtime_onsets = []
    realtime_durations = []

    for onset in onets:
        trial_start = onset + trial_start_window
        trial_stop = onset + trial_stop_window

        trial_start_realtime = frame_times[trial_start]
        trial_stop_realtime = frame_times[trial_stop]

        realtime_onsets.append(trial_start_realtime)
        realtime_durations.append(trial_stop_realtime - trial_start_realtime)

    minimum_duration = np.min(realtime_durations)

    for realtime_onset in realtime_onsets:
        trial_running_trace = running_trace[realtime_onset:realtime_onset + minimum_duration]
        condition_running_traces.append(trial_running_trace)

    condition_running_traces = np.array(condition_running_traces)
    return condition_running_traces



def get_onsets(base_directory, onsets_file_list):

    # Load Onsets
    onsets = []
    for onsets_file in onsets_file_list:
        onsets_file_contents = np.load(os.path.join(base_directory, "Stimuli_Onsets", onsets_file))
        for onset in onsets_file_contents:
            onsets.append(onset)

    return onsets


def get_mean_running_traces(base_directory, visual_onsets_file, odour_onsets_file, trial_start_window, trial_stop_window, axis):

    # Load Onsets
    visual_onsets = get_onsets(base_directory, visual_onsets_file)
    odour_onsets = get_onsets(base_directory, odour_onsets_file)
    frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    frame_times = Widefield_General_Functions.invert_dictionary(frame_times)

    # Load AI Recorder File
    ai_filename = Widefield_General_Functions.get_ai_filename(base_directory)
    ai_data = Widefield_General_Functions.load_ai_recorder_file(base_directory + "/" + ai_filename)
    running_trace = ai_data[8]

    # Get Condition Running Traces
    visual_running_traces = get_condition_running_traces(running_trace, visual_onsets, frame_times, trial_start_window, trial_stop_window)
    odour_running_traces = get_condition_running_traces(running_trace, odour_onsets, frame_times, trial_start_window, trial_stop_window)

    # Get Mean Running Traces
    visual_running_mean = np.mean(visual_running_traces, axis=0)
    odour_running_mean = np.mean(odour_running_traces, axis=0)

    # Get Running Trace SDs
    visual_running_sd = np.std(visual_running_traces, axis=0)
    odour_running_sd = np.std(odour_running_traces, axis=0)

    # Get X Values
    visual_timepoints = np.shape(visual_running_traces)[1]
    odour_timepoints = np.shape(odour_running_traces)[1]
    print("Visual timepoints", visual_timepoints)
    print("Odour timepoints", odour_timepoints)

    # Get Significance Values
    print("Odour running Traces", np.shape(odour_running_traces))
    print("Visual Running Traces", np.shape(visual_running_traces))
    t_stats, p_values = stats.ttest_ind(visual_running_traces, odour_running_traces, axis=0)
    print(np.shape(p_values))

    signficance_markers = []
    threshold = 0.05
    for p_value in p_values:
        if p_value < threshold:
            signficance_markers.append(1)
        else:
            signficance_markers.append(0)



    # Plot These
    visual_x_values = list(range(visual_timepoints))
    odour_x_values = list(range(odour_timepoints))

    visual_x_values = np.add(visual_x_values, 36 * trial_start)
    odour_x_values = np.add(odour_x_values, 36 * trial_start)

    axis.plot(visual_x_values, visual_running_mean, c='b')
    axis.fill_between(visual_x_values, np.subtract(visual_running_mean, visual_running_sd), np.add(visual_running_mean, visual_running_sd), color='b', alpha=0.2)

    axis.plot(odour_x_values, odour_running_mean, c='r')
    axis.fill_between(odour_x_values, np.subtract(odour_running_mean, odour_running_sd), np.add(odour_running_mean, odour_running_sd), color='r', alpha=0.2)

    max_value = np.max([np.add(visual_running_mean, visual_running_sd), np.add(odour_running_mean, odour_running_sd)])

    #axis.axvline([0], c='k')
    plot_title = base_directory.split('/')[-3]
    axis.set_title(plot_title)
    #axis.scatter(visual_x_values, np.multiply(signficance_markers, max_value))

    # Match Trials
    #match_trials(visual_running_traces, odour_running_traces)


controls = ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN78.1A/2020_12_09_Switching_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN78.1D/2020_11_29_Switching_Imaging/"]

mutants = [ "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK12.1F/2021_09_22_Transition_Imaging/",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN71.2A/2020_12_17_Switching_Imaging/"]


visual_onsets_file_list = ["visual_context_stable_vis_1_frame_onsets.npy", "visual_context_stable_vis_2_frame_onsets.npy"]
odour_onsets_file_list = ["odour_context_stable_vis_1_frame_onsets.npy", "odour_context_stable_vis_2_frame_onsets.npy"]

#trial_start = -10
#trial_stop = 40

trial_start = -65
trial_stop = -4




control_figure = plt.figure()
count = 1
for base_directory in controls:
    subplot = control_figure.add_subplot(1,5, count)
    get_mean_running_traces(base_directory, visual_onsets_file_list, odour_onsets_file_list, trial_start, trial_stop, subplot)
    count += 1

plt.show()

mutant_figure = plt.figure()
count = 1
for base_directory in mutants:
    subplot = mutant_figure.add_subplot(1,5, count)
    get_mean_running_traces(base_directory, visual_onsets_file_list, odour_onsets_file_list, trial_start, trial_stop, subplot)

    count += 1
plt.show()



# Now With Matching
control_figure = plt.figure()
count = 1
for base_directory in controls:
    subplot = control_figure.add_subplot(1,5, count)
    matched_visual_onsets, matched_odour_onsets = match_trials(base_directory, visual_onsets_file_list, odour_onsets_file_list, trial_start, trial_stop, subplot)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "Combined_Visual_Pre_Matched.npy"), matched_visual_onsets)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "Combined_Odour_Pre_Matched.npy"), matched_odour_onsets)
    count += 1
plt.show()


# Now With Matching
mutant_figure = plt.figure()
count = 1
for base_directory in mutants:
    subplot = mutant_figure.add_subplot(1,5, count)
    matched_visual_onsets, matched_odour_onsets = match_trials(base_directory, visual_onsets_file_list, odour_onsets_file_list, trial_start, trial_stop, subplot)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "Combined_Visual_Pre_Matched.npy"), matched_visual_onsets)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "Combined_Odour_Pre_Matched.npy"), matched_odour_onsets)
    count += 1
plt.show()
