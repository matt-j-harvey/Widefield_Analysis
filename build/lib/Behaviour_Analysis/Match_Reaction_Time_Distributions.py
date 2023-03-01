import matplotlib.pyplot as plt
import numpy as np
import os

import Session_List


def bin_trials_by_reaction_time(base_directory, window_start, window_stop, bin_size, early_cutoff=3000):

    behaviour_matrix = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

    # Get Lists Of All Trials
    binned_onsets = []
    reaction_time_list = []
    bin_n_list = []

    # Iterate Through Each Time Window
    for time_window in range(window_start, window_stop, bin_size):

        # Get Window Start and Stop
        window_start = time_window
        window_stop = window_start + bin_size

       # Create Empty List To Hold Onsets For This Window
        window_onsets = []
        window_reaction_times = []

        # Iterate Through Trial
        for trial in behaviour_matrix:

            # Get Trial Behavioural Characteristics
            trial_type = trial[1]
            correct = trial[3]
            reaction_time = trial[23]
            onset = trial[18]

            # If Trial Is Within Reaction time Window and Is Hit Add Onset To List
            if reaction_time > window_start and reaction_time < window_stop:
                if trial_type == 1 and correct == 1:
                    if onset != None and onset > early_cutoff:
                        window_onsets.append(onset)
                        window_reaction_times.append(reaction_time)


        binned_onsets.append(window_onsets)
        reaction_time_list.append(window_reaction_times)
        bin_n = len(window_onsets)
        bin_n_list.append(bin_n)

    binned_onsets = np.array(binned_onsets, dtype=object)

    return binned_onsets, reaction_time_list, bin_n_list



def get_matched_sample(onsets_list_1, onsets_list_2, rt_list_1, rt_list_2):

    # Create Empty Lists To Hold Selected Matched Onsets and RTs
    session_1_matched_onsets = []
    session_2_matched_onsets = []
    session_1_matched_rts = []
    session_2_matched_rts = []

    # Iterate Through Each Bin
    number_of_bins = len(onsets_list_1)
    for bin_index in range(number_of_bins):

        # Get Onsets and RTs For THis Time Bin
        session_1_bin_onsets = np.array(onsets_list_1[bin_index])
        session_2_bin_onsets = np.array(onsets_list_2[bin_index])
        session_1_bin_rts = np.array(rt_list_1[bin_index])
        session_2_bin_rts = np.array(rt_list_2[bin_index])

        # Get Smallest Number Of Trials
        session_1_trials = len(session_1_bin_onsets)
        session_2_trials = len(session_2_bin_onsets)
        smallest_number_of_trials = np.min([session_1_trials, session_2_trials])

        if smallest_number_of_trials > 0:

            # Select A Random Subset Of Trials From This Bin - Same Number of Trials For Each Condition
            session_1_pool = list(range(session_1_trials))
            session_2_pool = list(range(session_2_trials))

            selected_session_1_trials = np.random.choice(a=session_1_pool, size=smallest_number_of_trials, replace=False)
            selected_session_2_trials = np.random.choice(a=session_2_pool, size=smallest_number_of_trials, replace=False)

            # Get The Onsets and RTs For These Trials
            selected_session_1_onsets = session_1_bin_onsets[selected_session_1_trials]
            selected_session_2_onsets = session_2_bin_onsets[selected_session_2_trials]

            selected_session_1_rts = session_1_bin_rts[selected_session_1_trials]
            selected_session_2_rts = session_2_bin_rts[selected_session_2_trials]

            # Add These To The List
            session_1_matched_onsets.append(selected_session_1_onsets)
            session_2_matched_onsets.append(selected_session_2_onsets)
            session_1_matched_rts.append(selected_session_1_rts)
            session_2_matched_rts.append(selected_session_2_rts)

    return session_1_matched_onsets, session_2_matched_onsets, session_1_matched_rts, session_2_matched_rts



def view_trial_histograms(session_one, session_two, window_start, window_stop, window_size):

    session_one = np.concatenate(session_one)
    session_two = np.concatenate(session_two)

    bin_list = list(range(window_start, window_stop, window_size))

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.hist(session_one, alpha=0.2, color='b', bins=bin_list)
    axis_1.hist(session_two, alpha=0.2, color='g', bins=bin_list)
    axis_1.set_xticks(bin_list)

    plt.show()



def get_paired_matched_trials(session_tuple, window_start, window_stop, window_size, onsets_name):

    # Get Trial Distributions
    pre_learning_rt_distribution, pre_learning_reaction_time_list = bin_trials_by_reaction_time(session_tuple[0], window_start, window_stop, window_size)
    post_learning_rt_distribution, post_learning_reaction_time_list = bin_trials_by_reaction_time(session_tuple[1], window_start, window_stop, window_size)

    # View Raw Distributions
    view_trial_histograms(pre_learning_reaction_time_list, post_learning_reaction_time_list,  window_start, window_stop, window_size)

    # Match Distributions
    session_1_matched_onsets, session_2_matched_onsets, session_1_matched_rts, session_2_matched_rts = get_matched_sample(pre_learning_rt_distribution, post_learning_rt_distribution, pre_learning_reaction_time_list, post_learning_reaction_time_list)

    # View Matched Distributions
    view_trial_histograms(session_1_matched_rts, session_2_matched_rts, window_start, window_stop, window_size)

    # Save Onsets
    pre_learning_save_directory = os.path.join(session_tuple[0], "Stimuli_Onsets", onsets_name + ".npy")
    post_learning_save_directory = os.path.join(session_tuple[1], "Stimuli_Onsets", onsets_name + ".npy")

    np.save(pre_learning_save_directory, np.concatenate(session_1_matched_onsets))
    np.save(post_learning_save_directory, np.concatenate(session_2_matched_onsets))


def create_pool(group_1_n_list, group_2_n_list):

    group_1_pool_list = []
    group_2_pool_list = []

    for trial_number in group_1_n_list:
        group_1_pool_list.append(list(range(trial_number)))

    for trial_number in group_2_n_list:
        group_2_pool_list.append(list(range(trial_number)))

    return group_1_pool_list, group_2_pool_list

def create_empty_list(size):
    empty_list = []
    for item in range(size):
        empty_list.append([])
    return empty_list


def randomly_subsample_n_trials(group_size, group_pool_list, group_onsets_list, number_of_trials):

    # Create Empty List To Hold Selected Trials
    group_selected_onsets = create_empty_list(group_size)

    # Sample N Trials
    selected_trials = 0
    while selected_trials < number_of_trials:

        # Select Mouse
        mouse = np.random.randint(0, group_size)

        # Select Random Trial
        mouse_pool = group_pool_list[mouse]
        mouse_pool_size = len(mouse_pool)

        # If Mouse Pool Is Not Empty - Select A Random Trial
        if mouse_pool_size > 0:
            selected_index = np.random.randint(low=0, high=mouse_pool_size)
            selected_index = mouse_pool.pop(selected_index)
            group_pool_list[mouse] = mouse_pool
            mouse_onsets = group_onsets_list[mouse]
            selected_onset = mouse_onsets[selected_index]
            group_selected_onsets[mouse].append(selected_onset)
            selected_trials += 1

    return group_selected_onsets


def match_timepoint(group_1_onsets_list, group_2_onsets_list, group_1_n_list, group_2_n_list, selected_timepoint):

    # Get Group SIzes
    group_1_size = len(group_1_onsets_list)
    group_2_size = len(group_2_onsets_list)
    print("Group 1 size", group_1_size, "Group 2 size", group_2_size)

    # Get Onsets For This Timepoint
    group_1_onsets_list = np.array(group_1_onsets_list)
    print("Group 1 onsets list", group_1_onsets_list)
    group_2_onsets_list = np.array(group_2_onsets_list)
    group_1_onsets_list = group_1_onsets_list[:, selected_timepoint]
    group_2_onsets_list = group_2_onsets_list[:, selected_timepoint]
    print("Group 1 onsts list", group_1_onsets_list)
    print("Group 1 onsets list size", len(group_1_onsets_list))

    # Get Numbers Of Trials
    group_1_n_list = np.array(group_1_n_list)
    group_2_n_list = np.array(group_2_n_list)
    print("Group 1 n list", group_1_n_list)
    print("Group 2 n list", group_2_n_list)

    group_1_n = np.sum(group_1_n_list[:, selected_timepoint])
    group_2_n = np.sum(group_2_n_list[:, selected_timepoint])
    print("Group 1 n", group_1_n)
    print("group 2 n", group_2_n)

    number_of_trials = np.min([group_1_n, group_2_n])
    print("Number of trials", number_of_trials)

    # Create Pool List
    group_1_pool_list, group_2_pool_list = create_pool(group_1_n_list[:, selected_timepoint], group_2_n_list[:, selected_timepoint])
    print("group 1 pool list", group_1_pool_list)
    print("Group 2 pool list", group_2_pool_list)

    # Select Onsets
    group_1_onsets = randomly_subsample_n_trials(group_1_size, group_1_pool_list, group_1_onsets_list, number_of_trials)
    group_2_onsets = randomly_subsample_n_trials(group_2_size, group_2_pool_list, group_2_onsets_list, number_of_trials)

    return group_1_onsets, group_2_onsets


def get_matched_reaction_time_distributions(group_1_session_list, group_2_session_list, window_start, window_stop, window_size, onsets_name):

    # Get Group Sizes
    group_1_size = len(group_1_session_list)
    group_2_size = len(group_2_session_list)

    # Get Trial Distributions
    group_1_binned_onsets = []
    group_1_bin_sizes = []
    for session in group_1_session_list:
        reaction_time_distribution, reaction_time_list, bin_size_list = bin_trials_by_reaction_time(session, window_start, window_stop, window_size)
        group_1_binned_onsets.append(reaction_time_distribution)
        group_1_bin_sizes.append(bin_size_list)

    group_2_binned_onsets = []
    group_2_bin_sizes = []
    for session in group_2_session_list:
        reaction_time_distribution, reaction_time_list, bin_size_list = bin_trials_by_reaction_time(session, window_start, window_stop, window_size)
        group_2_binned_onsets.append(reaction_time_distribution)
        group_2_bin_sizes.append(bin_size_list)

    # Mactch Distributions
    match_timepoint(group_1_binned_onsets, group_2_binned_onsets, group_1_bin_sizes, group_2_bin_sizes, 0)
    number_of_time_bins = int((window_stop - window_start) / window_size)
    print("Number of time bins", number_of_time_bins)


    print()

    group_1_selected_onsets = create_empty_list(group_1_size)
    group_2_selected_onsets = create_empty_list(group_2_size)

    for time_bin in range(number_of_time_bins):
        group_1_onsets, group_2_onsets = match_timepoint(group_1_binned_onsets, group_2_binned_onsets, group_1_bin_sizes, group_2_bin_sizes, time_bin)

        for session in range(group_1_size):
            session_onsets = group_1_onsets[session]
            for onset in session_onsets:
                group_1_selected_onsets[session].append(onset)

        for session in range(group_2_size):
            session_onsets = group_2_onsets[session]
            for onset in session_onsets:
                group_2_selected_onsets[session].append(onset)

    return group_1_selected_onsets, group_2_selected_onsets


def save_onsets(session_list, matched_onsets_list, onsets_name):

    number_of_sessions = len(session_list)

    for session_index in range(number_of_sessions):
        session_onsets = matched_onsets_list[session_index]
        session_directory = session_list[session_index]
        np.save(os.path.join(session_directory, "Stimuli_Onsets", onsets_name), session_onsets)



# Window Settings
window_start = 500
window_stop = 2000
window_size = 100


control_sessions = Session_List.control_pre_learning_session_list
mutant_sessions = Session_List.mutant_pre_learning_session_list

control_sessions = Session_List.control_post_learning_session_list
mutant_sessions = Session_List.mutant_post_learning_session_list
onsets_name = "Genotype_RT_Matched_Vis_1_Onsets"



group_1_matched_onsets, group_2_matched_onsets = get_matched_reaction_time_distributions(control_sessions, mutant_sessions, window_start, window_stop, window_size, onsets_name)
save_onsets(control_sessions, group_1_matched_onsets, onsets_name)
save_onsets(mutant_sessions, group_2_matched_onsets, onsets_name)

"""
control_tuples = Session_List.control_session_tuples
for tuple in control_tuples:
    get_pre_and_post_matched_trials(tuple, window_start, window_stop, window_size)

mutant_tuples = Session_List.mutant_session_tuples
for tuple in mutant_tuples:
    get_pre_and_post_matched_trials(tuple, window_start, window_stop, window_size)
"""


