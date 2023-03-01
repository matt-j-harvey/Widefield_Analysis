
import numpy as np
import os
import matplotlib.pyplot as plt
from Widefield_Utils import widefield_utils


def get_step_onsets(trace, threshold=1, window=10):
    state = 0
    number_of_timepoints = len(trace)
    onset_times = []
    time_below_threshold = 0

    onset_line = []

    for timepoint in range(number_of_timepoints):
        if state == 0:
            if trace[timepoint] > threshold:
                state = 1
                onset_times.append(timepoint)
                time_below_threshold = 0
            else:
                pass
        elif state == 1:
            if trace[timepoint] > threshold:
                time_below_threshold = 0
            else:
                time_below_threshold += 1
                if time_below_threshold > window:
                    state = 0
                    time_below_threshold = 0
        onset_line.append(state)

    return onset_times

def get_offset_list(onset_list, trace):
    offset_list = []

    for onset in onset_list:
        offset = get_offset(onset, trace)
        offset_list.append(offset)

    return offset_list


def get_offset(onset, stream, threshold=0.5):

    count = 0
    on = True
    while on:
        if onset + count < len(stream):
            if stream[onset + count] < threshold and count > 10:
                on = False
                return onset + count
            else:
                count += 1

        else:
            return np.nan


def extract_stimuli_portions_from_trace(trace, onset_list, offset_list):

    n_trials = len(onset_list)

    new_trace = np.zeros(len(trace))

    for trial_index in range(n_trials):
        trial_start = onset_list[trial_index]
        trial_stop = offset_list[trial_index]
        new_trace[trial_start:trial_stop] = trace[trial_start:trial_stop]

    return new_trace


def split_visual_onsets_by_context(visual_1_onsets, visual_2_onsets, odour_1_onsets, odour_2_onsets, following_window_size=140):

    combined_odour_onsets = odour_1_onsets + odour_2_onsets
    visual_block_stimuli_1, odour_block_stimuli_1 = split_stream_by_context(visual_1_onsets, combined_odour_onsets, following_window_size)
    visual_block_stimuli_2, odour_block_stimuli_2 = split_stream_by_context(visual_2_onsets, combined_odour_onsets, following_window_size)

    onsets_list = [visual_block_stimuli_1, visual_block_stimuli_2, odour_block_stimuli_1, odour_block_stimuli_2]

    return onsets_list




def split_stream_by_context(stimuli_onsets, context_onsets, context_window):
    context_negative_onsets = []
    context_positive_onsets = []

    # Iterate Through Visual 1 Onsets
    for stimuli_onset in stimuli_onsets:
        context = False
        window_start = stimuli_onset
        window_end = stimuli_onset + context_window

        for context_onset in context_onsets:
            if context_onset >= window_start and context_onset <= window_end:
                context = True

        if context == True:
            context_positive_onsets.append(stimuli_onset)
        else:
            context_negative_onsets.append(stimuli_onset)

    return context_negative_onsets, context_positive_onsets



def split_ai_channels_by_context(ai_data):

    # Create Stimuli Dictionary
    stimuli_dictionary = widefield_utils.create_stimuli_dictionary()

    # Load Traces
    vis_1_trace = ai_data[stimuli_dictionary["Visual 1"]]
    vis_2_trace = ai_data[stimuli_dictionary["Visual 2"]]
    odour_1_trace = ai_data[stimuli_dictionary["Odour 1"]]
    odour_2_trace = ai_data[stimuli_dictionary["Odour 2"]]

    # Get Onsets
    vis_1_onsets = get_step_onsets(vis_1_trace)
    vis_2_onsets = get_step_onsets(vis_2_trace)
    odour_1_onsets = get_step_onsets(odour_1_trace)
    odour_2_onsets = get_step_onsets(odour_2_trace)

    # Split Visual Onsets By Context
    visual_onsets_by_context = split_visual_onsets_by_context(vis_1_onsets, vis_2_onsets, odour_1_onsets, odour_2_onsets)
    vis_context_vis_1_onsets = visual_onsets_by_context[0]
    vis_context_vis_2_onsets = visual_onsets_by_context[1]
    odour_context_vis_1_onsets = visual_onsets_by_context[2]
    odour_context_vis_2_onsets = visual_onsets_by_context[3]

    # Get Offsets
    vis_context_vis_1_offsets = get_offset_list(vis_context_vis_1_onsets, vis_1_trace)
    odour_context_vis_1_offsets = get_offset_list(odour_context_vis_1_onsets, vis_1_trace)

    vis_context_vis_2_offsets = get_offset_list(vis_context_vis_2_onsets, vis_2_trace)
    odour_context_vis_2_offsets = get_offset_list(odour_context_vis_2_onsets, vis_2_trace)

    # Extract Traces
    vis_context_vis_1_trace = extract_stimuli_portions_from_trace(vis_1_trace, vis_context_vis_1_onsets, vis_context_vis_1_offsets)
    vis_context_vis_2_trace = extract_stimuli_portions_from_trace(vis_2_trace, vis_context_vis_2_onsets, vis_context_vis_2_offsets)
    odour_context_vis_1_trace = extract_stimuli_portions_from_trace(vis_1_trace, odour_context_vis_1_onsets, odour_context_vis_1_offsets)
    odour_context_vis_2_trace = extract_stimuli_portions_from_trace(vis_2_trace, odour_context_vis_2_onsets, odour_context_vis_2_offsets)

    #plt.plot(vis_1_trace, c='b', alpha=0.5)
    #plt.plot(vis_2_trace, c='r', alpha=0.5)

    """
    combined_odour_trace = np.add(odour_1_trace, odour_2_trace)

    # Visualise
    jitter = 0.2
    norm_vis_context_vis_1_trace = np.divide(vis_context_vis_1_trace, np.max(vis_context_vis_1_trace)+jitter) + 1
    norm_vis_context_vis_2_trace = np.divide(vis_context_vis_2_trace, np.max(vis_context_vis_2_trace)+jitter) + 2
    norm_odour_context_vis_1_trace = np.divide(odour_context_vis_1_trace, np.max(odour_context_vis_1_trace)+jitter) + 3
    norm_odour_context_vis_2_trace = np.divide(odour_context_vis_2_trace, np.max(odour_context_vis_2_trace)+jitter) + 4
    norm_odour_trace = np.divide(combined_odour_trace, np.max(combined_odour_trace)+jitter) + 5

    plt.plot(norm_vis_context_vis_1_trace, c='b', alpha=0.5)
    plt.plot(norm_vis_context_vis_2_trace, c='r', alpha=0.5)

    plt.plot(norm_odour_context_vis_1_trace, c='m', alpha=0.5)
    plt.plot(norm_odour_context_vis_2_trace, c='k', alpha=0.5)

    plt.plot(norm_odour_trace, c='g', alpha=0.5)


    plt.show()
    """
    return vis_context_vis_1_trace, vis_context_vis_2_trace, odour_context_vis_1_trace, odour_context_vis_2_trace
