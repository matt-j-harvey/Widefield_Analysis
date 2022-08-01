import matplotlib.pyplot as plt
import numpy as np
import os


def save_samples(session_list, matched_samples, filename):

    number_of_sessions = len(session_list)

    for session_index in range(number_of_sessions):
        session_onsets = matched_samples[session_index]
        session_onsets = np.concatenate(session_onsets)

        session_name = session_list[session_index]
        full_filepath = os.path.join(session_name, "Stimuli_Onsets", filename + "_onsets.npy")
        print(full_filepath)
        np.save(full_filepath, session_onsets)

def get_matched_sample(binned_onsets):

    print("Binned Onsets Lenght", len(binned_onsets))
    number_of_sessions = len(binned_onsets[0])

    matched_onsets = []
    for time_window in binned_onsets:

        # Get Number Of Trials To Sample
        #print("Time Window", time_window)
        window_trial_numbers = []
        for session in time_window:
            number_of_trials = len(session)
            window_trial_numbers.append(number_of_trials)
        smallest_number_of_trials = np.min(window_trial_numbers)
        print("Smallest number", smallest_number_of_trials)
        selected_window_onsets = []
        for session_index in range(number_of_sessions):
            session_onsets = time_window[session_index]
            selected_session_onsets = np.random.choice(a=session_onsets, size=smallest_number_of_trials)
            selected_window_onsets.append(selected_session_onsets)
        matched_onsets.append(selected_window_onsets)

    return matched_onsets




def get_matched_reaction_times(session_list, window_start, window_stop, window_step=50):

    number_of_sessions = len(session_list)

    # Load Behaviour Matricies
    behaviour_matrix_list = []
    for base_directory in session_list:
        behaviour_matrix = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)
        behaviour_matrix_list.append(behaviour_matrix)

    # Get Lists Of All Trials
    binned_onsets = []
    binned_reaction_times = []

    # Iterate Through Each Time Window
    for time_window in range(window_start, window_stop, window_step):
        window_start = time_window
        window_stop = window_start + window_step

        # Iterate Through Each Session
        window_onsets = []
        window_reaction_times = []
        for session_index in range(number_of_sessions):
            session_onsets = []
            session_reaction_times = []

            # Load Behaviour Matrix
            behaviour_matrix = behaviour_matrix_list[session_index]

            # Iterate Through Trial
            for trial in behaviour_matrix:

                # Get Trial Behavioural Characteristics
                trial_type = trial[1]
                correct = trial[3]
                reaction_time = trial[23]
                onset = trial[18]

                if reaction_time > window_start and reaction_time <= window_stop:
                    if trial_type == 1 and correct == 1:
                        if onset != None:
                            session_onsets.append(onset)
                            session_reaction_times.append(reaction_time)

            window_onsets.append(session_onsets)
            window_reaction_times.append(session_reaction_times)
        binned_onsets.append(window_onsets)
        binned_reaction_times.append(window_reaction_times)

    binned_onsets = np.array(binned_onsets)
    print("binned_onsets", np.shape(binned_onsets))

    matched_onsets = get_matched_sample(binned_onsets)
    matched_onsets = np.transpose(matched_onsets)
    return matched_onsets

def display_mouse_reaction_times(session_list):

    figure_1 = plt.figure()
    rows = len(session_list)
    columns = 1

    bin_list = list(range(0, 5000, 100))
    print(bin_list)

    # Iterate Through Session List
    session_count = 1
    for base_directory in session_list:

        session_reaction_times = []

        # Load Behaviour Matrix
        behaviour_matrix = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

        # Iterate Through Trial
        for trial in behaviour_matrix:

            # Get Trial Behavioural Characteristics
            trial_type = trial[1]
            correct = trial[3]
            reaction_time = trial[23]

            if trial_type == 1 and correct == 1:
                session_reaction_times.append(reaction_time)

        axis = figure_1.add_subplot(rows, columns, session_count)
        axis.set_xlim([0, 5000])
        axis.hist(session_reaction_times, alpha=0.3, bins=bin_list)
        axis.set_title(str(session_list[0].split("/")[-2]))
        session_count += 1

    plt.show()









control_session_list = [

    # Controls 46 sessions

    # 78.1A - 6
    [
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_16_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_17_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_19_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_21_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_24_Discrimination_Imaging",
    ],

    # 78.1D - 8
    [
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_14_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_15_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_16_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_17_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_19_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_21_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_23_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_25_Discrimination_Imaging",
    ],

    # 4.1B - 7
    [
    #"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
    #"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging",
    ],

    # 22.1A - 7
    [
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_09_25_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_09_29_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_01_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_03_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_05_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_07_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging",
    ],

    # 14.1A - 6
    [
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_04_29_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_01_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_03_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_05_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_07_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_09_Discrimination_Imaging",
    ],

    # 7.1B - 12
    [
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_01_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_03_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_05_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_07_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_09_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_11_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_13_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_15_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_17_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_19_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_22_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_24_Discrimination_Imaging",
]]


mutant_session_list = [
    # Mutants

    # 4.1A - 15
    [
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_08_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_10_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_12_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_14_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_16_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_18_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_23_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_25_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_27_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_03_01_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_03_03_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging",
    ],

    # 20.1B - 11
    [
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_09_28_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_09_30_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_09_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_11_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_13_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_15_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_17_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging",
    ],

    # 24.1C - 10
    [
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_20_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_22_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_24_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_26_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_28_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_30_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_10_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_10_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_10_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_10_08_Discrimination_Imaging",
    ],

    # NXAK16.1B - 16
    [
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_04_30_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_08_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_10_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_12_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_14_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_16_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_18_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_20_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_22_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_24_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_26_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_06_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_06_15_Discrimination_Imaging",
    ],

    # 10.1A - 8
    [
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_04_30_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_08_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_10_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_12_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_14_Discrimination_Imaging",
    ]
]

# 78.1A 1000 - 1100
# 78.1D 900 - 1000
# 4.1B 900 - 1000
# 22.1A 1500 - 1600
# 14.1A 1100 - 1200
# 7.1B 1000 - 1100

# 4.1A 400 - 500
# 20.1B 800 - 900
# 24.1C 900 - 1000
# 16.1B 800 - 900
# 10.1A 1500 - 1600

"""
for mouse in control_session_list:
    get_specific_reaction_times(mouse)
"""

selected_mouse = control_session_list[2]
window = [800,1200]

display_mouse_reaction_times(selected_mouse)
matched_onsets = get_matched_reaction_times(selected_mouse, window[0], window[1], window_step=50)
print("Matched onsets", np.shape(matched_onsets))

filename = "matched_test"
save_samples(selected_mouse, matched_onsets, filename)


#get_mouse_reaction_times(selected_mouse, window)