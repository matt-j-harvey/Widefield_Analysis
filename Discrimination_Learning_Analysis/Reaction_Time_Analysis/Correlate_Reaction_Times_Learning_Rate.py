import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats

def get_specific_reaction_times(session_list, window=[900, 1000]):

    mouse_reaction_times = []

    # Iterate Through Session List
    for base_directory in session_list[1:]:

        session_reaction_times = []

        # Load Behaviour Matrix
        behaviour_matrix = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

        # Iterate Through Trial
        for trial in behaviour_matrix:

            # Get Trial Behavioural Characteristics
            trial_type = trial[1]
            correct = trial[3]
            reaction_time = trial[23]

            # Get Event Onsets
            onset_frame = trial[18]
            lick_frame = trial[22]

            if trial_type == 1 and correct == 1:
                session_reaction_times.append(reaction_time)

        mean_reaction_time = np.mean(session_reaction_times)
        mouse_reaction_times.append(mean_reaction_time)

    return np.mean(mouse_reaction_times)



def correlate_reaction_times_and_learning_rate(mouse_list, session_list, learning_rate_dict, color='b'):

    learning_rate_list = []
    reaction_time_list = []

    for mouse in mouse_list:
        learning_rate = learning_rate_dict[mouse]
        reaction_time = get_specific_reaction_times(session_list[mouse])

        learning_rate_list.append(learning_rate)
        reaction_time_list.append(reaction_time)

    plt.scatter(reaction_time_list, learning_rate_list, c=color)
    return learning_rate_list, reaction_time_list

control_session_list = {

    # Controls 46 sessions

    # 78.1A - 6
    "NRXN78.1A":[r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_16_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_17_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_19_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_21_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_24_Discrimination_Imaging",],

    # 78.1D - 8
    "NRXN78.1D":[r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_14_Discrimination_Imaging",
                r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_15_Discrimination_Imaging",
                r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_16_Discrimination_Imaging",
                r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_17_Discrimination_Imaging",
                r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_19_Discrimination_Imaging",
                r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_21_Discrimination_Imaging",
                r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_23_Discrimination_Imaging",
                r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_25_Discrimination_Imaging",],

    # 4.1B - 7
    "NXAK4.1B":["/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
                "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
                "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
                "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
                "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
                "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
                "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging",],

    # 22.1A - 7
    "NXAK22.1A":[r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_09_25_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_09_29_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_01_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_03_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_05_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_07_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging",],

    # 14.1A - 6
    "NXAK14.1A":[r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_04_29_Discrimination_Imaging",
                r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_01_Discrimination_Imaging",
                r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_03_Discrimination_Imaging",
                r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_05_Discrimination_Imaging",
                r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_07_Discrimination_Imaging",
                r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_09_Discrimination_Imaging",],

    # 7.1B - 12
    "NXAK7.1B":[r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_01_Discrimination_Imaging",
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
                r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_24_Discrimination_Imaging",],

    }


mutant_session_list = {
    # Mutants

    # 4.1A - 15
    "NXAK4.1A":[r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_02_Discrimination_Imaging",
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
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging",],

    # 20.1B - 11
    "NXAK20.1B":[r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_09_28_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_09_30_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_02_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_04_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_06_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_09_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_11_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_13_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_15_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_17_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging",],

    # 24.1C - 10
    "NXAK24.1C":[r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_20_Discrimination_Imaging",
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
    "NXAK16.1B":[r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_04_30_Discrimination_Imaging",
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
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_06_15_Discrimination_Imaging",],

    # 10.1A - 8
    "NXAK10.1A":[r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_04_30_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_02_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_04_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_06_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_08_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_10_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_12_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_14_Discrimination_Imaging",],
}


#    "NRXN71.2A":21,

mutants = {
    "NXAK4.1A":28,
    "NXAK16.1B":28,
    "NXAK10.1A": 15,
    "NXAK24.1C":18,
    "NXAK20.1B":23,
}

controls = {
    "NRXN78.1A":9,
    "NRXN78.1D":13,
    "NXAK4.1B":19,
    "NXAK7.1B":24,
    "NXAK14.1A":11,
    "NXAK22.1A":16,
}

control_learning_rates, control_reaction_times = correlate_reaction_times_and_learning_rate(list(mutants.keys()), mutant_session_list, mutants, color='g')

mutant_learning_rates, mutant_reaction_times = correlate_reaction_times_and_learning_rate(list(controls.keys()), control_session_list, controls, color='b')

combined_learning_rates = control_learning_rates + mutant_learning_rates
combined_reaction_times = control_reaction_times + mutant_reaction_times

r, p = stats.pearsonr(combined_learning_rates, combined_reaction_times)
print("r", r)
print("p", p)
plt.show()