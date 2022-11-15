import numpy as np
import os
import matplotlib.pyplot as plt

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def get_binned_reaction_time_trials(session_list, window_start, window_stop, bin_size, early_cutoff=3000):

    number_of_sessions = len(session_list)

    # Load Behaviour Matricies
    behaviour_matrix_list = []
    for base_directory in session_list:
        behaviour_matrix = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)
        behaviour_matrix_list.append(behaviour_matrix)

    # Get Lists Of All Trials
    binned_onsets = []
    binned_reaction_times = []
    trial_number_list = []

    # Iterate Through Each Time Window
    for time_window in range(window_start, window_stop, bin_size):
        window_start = time_window
        window_stop = window_start + bin_size

        # Iterate Through Each Session
        window_onsets = []
        window_reaction_times = []
        window_numbers = []
        for session_index in range(number_of_sessions):
            session_onsets = []
            session_reaction_times = []
            session_bin_trials = 0

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
                        if onset != None and onset > early_cutoff:
                            session_onsets.append(onset)
                            session_reaction_times.append(reaction_time)
                            session_bin_trials += 1


            window_onsets.append(session_onsets)
            window_reaction_times.append(session_reaction_times)
            window_numbers.append(session_bin_trials)
        trial_number_list.append(window_numbers)
        binned_onsets.append(window_onsets)
        binned_reaction_times.append(window_reaction_times)

    binned_onsets = np.array(binned_onsets)
    return trial_number_list, binned_onsets


def add_axis_text(ax, extent, data):

    y_size, x_size = np.shape(data)

    [x_start, x_end, y_end, y_start] = extent

    # Add the text
    jump_x = (x_end - x_start) / (2.0 * x_size)
    jump_y = (y_end - y_start) / (2.0 * y_size)
    x_positions = np.linspace(start=x_start, stop=x_end, num=x_size, endpoint=False)
    y_positions = np.linspace(start=y_start, stop=y_end, num=y_size, endpoint=False)

    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            label = data[y_index, x_index]
            text_x = x + jump_x
            text_y = y + jump_y


            ax.text(text_x, text_y, label, color='black', ha='center', va='center')


def view_trial_number_arrays(control_trial_number_list, mutant_trial_number_list):

    # Convert To Arrays
    control_trial_number_list = np.array(control_trial_number_list)
    mutant_trial_number_list = np.array(mutant_trial_number_list)

    control_trial_number_list = np.transpose(control_trial_number_list)
    mutant_trial_number_list = np.transpose(mutant_trial_number_list)

    # Get Extents
    control_extent = [window_start, window_stop, np.shape(control_trial_number_list)[0], 0]
    mutant_extent = [window_start, window_stop, np.shape(mutant_trial_number_list)[0], 0]

    #
    figure_1 = plt.figure(figsize=(12, 5))
    rows = 1
    columns = 2
    control_axis = figure_1.add_subplot(rows, columns, 1)
    mutant_axis = figure_1.add_subplot(rows, columns, 2)

    control_axis.imshow(control_trial_number_list, vmin=0, vmax=15, extent=control_extent)
    mutant_axis.imshow(mutant_trial_number_list, vmin=0, vmax=15, extent=mutant_extent)

    control_axis.set_ylabel("Session")
    mutant_axis.set_ylabel("Session")
    control_axis.set_xlabel("Reaction Time Bin")
    mutant_axis.set_xlabel("Reaction Time Bin")

    control_axis.set_title("Controls")
    mutant_axis.set_title("Mutants")

    forceAspect(control_axis)
    forceAspect(mutant_axis)

    add_axis_text(control_axis, control_extent, control_trial_number_list)
    add_axis_text(mutant_axis, mutant_extent, mutant_trial_number_list)

    plt.show()



control_session_list = [r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_14_Discrimination_Imaging",
                        r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging",

                        r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_14_Discrimination_Imaging",
                        r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_15_Discrimination_Imaging",

                        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_04_Discrimination_Imaging",
                        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_06_Discrimination_Imaging",

                        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_01_Discrimination_Imaging",
                        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_03_Discrimination_Imaging",

                        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_04_29_Discrimination_Imaging",
                        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_01_Discrimination_Imaging",

                        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_25_Discrimination_Imaging",
                        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_29_Discrimination_Imaging",

                        ]

mutant_session_list = [ r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_13_Discrimination_Imaging",
                        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_14_Discrimination_Imaging",

                        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_02_Discrimination_Imaging",
                        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_04_Discrimination_Imaging",

                        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_04_30_Discrimination_Imaging",
                        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_02_Discrimination_Imaging",

                        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_04_30_Discrimination_Imaging",
                        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_02_Discrimination_Imaging",

                        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_28_Discrimination_Imaging",
                        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_30_Discrimination_Imaging",

                        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_20_Discrimination_Imaging",
                        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_22_Discrimination_Imaging",
]


# Vis 1
# Lick Between 500 - 2000 seconds
# In 50ms Bins (will be 30)

window_start = 500
window_stop = 2000
bin_size = 100
bin_list = list(range(window_start, window_stop, bin_size))

control_trial_number_list, control_binned_onsets = get_binned_reaction_time_trials(control_session_list, window_start, window_stop, bin_size)
mutant_trial_number_list, mutant_binned_onsets = get_binned_reaction_time_trials(mutant_session_list, window_start, window_stop, bin_size)

view_trial_number_arrays(control_trial_number_list, mutant_trial_number_list)

# Save These Arrays
control_save_file = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Early_Learning/Control_Vis_1_RT_Binned.npy"
mutant_save_file = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Early_Learning/Mutant_Vis_1_RT_Binned.npy"
list_save_file = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Early_Learning/Bin_List.npy"
np.save(control_save_file, control_binned_onsets)
np.save(mutant_save_file, mutant_binned_onsets)
np.save(list_save_file, bin_list)