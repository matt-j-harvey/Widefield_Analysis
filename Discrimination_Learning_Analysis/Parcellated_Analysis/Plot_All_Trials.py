import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from matplotlib import gridspec
from scipy import stats

def get_trial_tensor(delta_f_matrix, onset_list, start_window, stop_window):

    number_of_timepoints = np.shape(delta_f_matrix)[0]

    trial_tensor = []
    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window

        if trial_start >= 0 and trial_stop < number_of_timepoints - 1:
            trial_data = delta_f_matrix[trial_start:trial_stop]
            trial_tensor.append(trial_data)

    trial_tensor = np.array(trial_tensor)

    return trial_tensor

def normalise_activity_matrix(activity_matrix):

    # Subtract Min
    min_vector = np.min(activity_matrix, axis=0)
    activity_matrix = np.subtract(activity_matrix, min_vector)

    # Divide By Max
    max_vector = np.max(activity_matrix, axis=0)
    activity_matrix = np.divide(activity_matrix, max_vector)

    return activity_matrix


def get_sig_differences(condition_1_tensor, condition_2_tensor):

    number_of_timepoints = np.shape(condition_1_tensor)[1]

    p_array = []
    t_array = []

    for timepoint_index in range(number_of_timepoints):
        condition_1_vector = condition_1_tensor[:, timepoint_index]
        condition_2_vector = condition_2_tensor[:, timepoint_index]

        print("Condition 1 vector", np.shape(condition_1_vector))
        print("Condition 2 vector", np.shape(condition_2_vector))

        t_vector, p_vector = stats.ttest_ind(condition_1_vector, condition_2_vector)
        p_array.append(p_vector)
        t_array.append(t_vector)

    p_array = np.array(p_array)
    t_array = np.array(t_array)

    return t_array, p_array


def check_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def plot_trial_tensor(trial_tensor, start_window, save_directory):

    number_of_trials = np.shape(trial_tensor)[0]
    for trial_index in range(number_of_trials):
        trial = trial_tensor[trial_index]
        plt.title("Trial: " + str(trial_index))
        plt.imshow(np.transpose(trial), cmap='jet', vmin=0, vmax=1)
        plt.axvline(x=-start_window, c='k', linestyle='--')
        plt.savefig(os.path.join(save_directory, str(trial_index).zfill(4) + ".png"))
        plt.close()



def plot_all_rasters(session_list, condition_1, condition_2, start_window, stop_window):

    # Iterate Through Each Session
    number_of_sessions = len(session_list)

    # Create Figure
    figure_1 = plt.figure(figsize=(20, 15), dpi=80)
    rows = 4
    columns = number_of_sessions
    gridspec_1 = gridspec.GridSpec(rows, columns, figure=figure_1)

    for session_index in range(number_of_sessions):
        base_directory = session_list[session_index]

        # Create Save Directories
        all_plot_directory = os.path.join(base_directory, "All_Trial_Rasters")
        vis_1_save_directory = os.path.join(all_plot_directory, "Vis_1_Trials")
        vis_2_save_directory = os.path.join(all_plot_directory, "Vis_2_Trials")

        check_directory(all_plot_directory)
        check_directory(vis_1_save_directory)
        check_directory(vis_2_save_directory)

        # Load Neural Data
        activity_matrix = np.load(os.path.join(base_directory, "Cluster_Activity_Matrix.npy"))

        # Normalise Activity Matrix
        activity_matrix = normalise_activity_matrix(activity_matrix)

        # Remove Background Activity
        activity_matrix = activity_matrix[:, 1:]

        # Load Onsets
        vis_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_1))
        vis_2_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_2))

        # Get Trial Tensors
        vis_1_tensor = get_trial_tensor(activity_matrix, vis_1_onsets, start_window, stop_window)
        vis_2_tensor = get_trial_tensor(activity_matrix, vis_2_onsets, start_window, stop_window)

        # Plot Each Tensor
        plot_trial_tensor(vis_1_tensor, start_window, vis_1_save_directory)
        plot_trial_tensor(vis_2_tensor, start_window, vis_2_save_directory)



        """
        plt.imshow(np.transpose(thresholded_t_array), cmap='bwr', vmin=-t_magnitude, vmax=t_magnitude)
        plt.show()
       

        vis_1_axis = figure_1.add_subplot(gridspec_1[0, session_index])
        vis_2_axis = figure_1.add_subplot(gridspec_1[1, session_index])
        diff_axis = figure_1.add_subplot(gridspec_1[2, session_index])
        sig_axis = figure_1.add_subplot(gridspec_1[3, session_index])

        vis_1_axis.set_title("Session: " + str(session_index) + " \n Vis 1")
        vis_2_axis.set_title("Session: " + str(session_index) + " \n Vis 2")
        diff_axis.set_title("Session: " + str(session_index) + " \n Diff")
        sig_axis.set_title("Session: " + str(session_index) + " \n Signficant Differences")

        vis_1_axis.imshow(np.transpose(vis_1_mean), cmap='jet', vmin=0, vmax=1)
        vis_2_axis.imshow(np.transpose(vis_2_mean), cmap='jet', vmin=0, vmax=1)
        diff_axis.imshow(np.transpose(np.subtract(vis_1_mean, vis_2_mean)), cmap='bwr', vmin=-0.5, vmax=0.5)
        sig_axis.imshow(np.transpose(thresholded_t_array), cmap='bwr', vmin=-t_magnitude, vmax=t_magnitude)

        vis_1_axis.axis('off')
        vis_2_axis.axis('off')
        diff_axis.axis('off')
        sig_axis.axis('off')
        """


session_list = [
    ["/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
     "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging"]

     #"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
     #"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
     #"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
     #"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging"],
]


# Decoding Parameters
start_window = -27
stop_window = 56
condition_1 = "visual_1_all_onsets.npy"
condition_2 = "visual_2_all_onsets.npy"
plot_save_directory = "/media/matthew/Expansion/Widefield_Analysis/Discrimination_Analysis/Average_Rasters"

for mouse in session_list:
    plot_all_rasters(mouse, condition_1, condition_2, start_window, stop_window)