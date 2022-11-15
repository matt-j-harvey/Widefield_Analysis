import matplotlib.pyplot as plt
import numpy as np
import os


def bin_trials_by_reaction_time(base_directory, window_start, window_stop, bin_size, early_cutoff=3000):

    behaviour_matrix = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

    # Get Lists Of All Trials
    binned_onsets = []
    reaction_time_list = []

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
    binned_onsets = np.array(binned_onsets, dtype=object)

    return binned_onsets, reaction_time_list



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



def get_pre_and_post_matched_trials(session_tuple, window_start, window_stop, window_size):

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
    pre_learning_save_directory = os.path.join(session_tuple[0], "Stimuli_Onsets", "Hits_RT_Matched.npy")
    post_learning_save_directory = os.path.join(session_tuple[1], "Stimuli_Onsets", "Hits_RT_Matched.npy")
    np.save(pre_learning_save_directory, np.concatenate(session_1_matched_onsets))
    np.save(post_learning_save_directory, np.concatenate(session_2_matched_onsets))


session_tuples = [

    [r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging",
     r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_24_Discrimination_Imaging"],

    [r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_15_Discrimination_Imaging",
     r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_25_Discrimination_Imaging"],

    [r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_06_Discrimination_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_22_Discrimination_Imaging"],

    [r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_03_Discrimination_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_24_Discrimination_Imaging"],

    [r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_01_Discrimination_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_09_Discrimination_Imaging"],

    [r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_29_Discrimination_Imaging",
     r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging"]

]


session_tuples = [

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_14_Discrimination_Imaging",
     r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_09_Discrimination_Imaging"],

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_04_Discrimination_Imaging",
     r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging"],

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_02_Discrimination_Imaging",
     r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_14_Discrimination_Imaging"],

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_02_Discrimination_Imaging",
     r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_15_Discrimination_Imaging"],

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_30_Discrimination_Imaging",
     r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging"],

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_22_Discrimination_Imaging",
     r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_08_Discrimination_Imaging"],
    ]



window_start = 500
window_stop = 2000
window_size = 100

for tuple in session_tuples:
    get_pre_and_post_matched_trials(tuple, window_start, window_stop, window_size)