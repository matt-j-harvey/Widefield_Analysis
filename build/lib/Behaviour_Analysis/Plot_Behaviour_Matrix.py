import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, patches


def get_patch_colour(trial_type):

    if trial_type == 1:
        patch_colour = 'b'

    elif trial_type == 2:
        patch_colour = 'r'

    elif trial_type == 3:
        patch_colour = 'g'

    elif trial_type == 4:
        patch_colour = 'm'

    return patch_colour


def plot_behaviour_maxtrix(base_directory, behaviour_matrix, onsets_dictioanry, block_boundaries, stable_windows, selected_trials):

    # Default Settings
    raster_start = 3000
    raster_stop = 5000
    border_size = 4
    patch_height = 1
    scatter_size = 5

    # Load Onsets
    lick_onsets = onsets_dictioanry["lick_onsets"]
    reward_onsets = onsets_dictioanry["reward_onsets"]
    number_of_trials = np.shape(behaviour_matrix)[0]

    # Setup Figure
    figure_1 = plt.figure(figsize=(12,17))
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.set_xlim([0, raster_start + raster_stop])
    axis_1.set_ylim([0, number_of_trials + 1])

    # Plot Each Trial
    for trial in range(number_of_trials):

        # Plot Trial Patch
        trial_type = behaviour_matrix[trial][1]
        trial_onset = behaviour_matrix[trial][11]
        trial_offset = behaviour_matrix[trial][12]

        patch_colour = get_patch_colour(trial_type)
        patch_y = (number_of_trials - trial) - patch_height
        patch_width = trial_offset - trial_onset
        trial_patch = patches.Rectangle(xy=(raster_start, patch_y), height=patch_height, width=patch_width, alpha=0.5, color=patch_colour)
        axis_1.add_patch(trial_patch)


        # Plot Irrel Patch
        preceeded_by_irrel = behaviour_matrix[trial][5]

        if preceeded_by_irrel:
            irrel_type = behaviour_matrix[trial][6]
            irrel_onset = behaviour_matrix[trial][13]
            irrel_offset = behaviour_matrix[trial][14]
            irrel_patch_width = irrel_offset - irrel_onset
            irrel_patch_start = raster_start - (trial_onset - irrel_onset)
            irrel_patch_colour = get_patch_colour(irrel_type)
            irrel_patch = patches.Rectangle(xy=(irrel_patch_start, patch_y), height=patch_height, width=irrel_patch_width, alpha=0.5, color=irrel_patch_colour)
            axis_1.add_patch(irrel_patch)


        # Plot Licks
        scatter_y = (number_of_trials - trial) - 0.5
        for lick in lick_onsets:
            if lick > trial_onset - raster_start and lick < trial_onset + raster_stop:
                axis_1.scatter(3000 - (trial_onset - lick), scatter_y, s=scatter_size, c='k')

        # Plot Reward Onsets
        for onset in reward_onsets:
            if onset > trial_onset - raster_start and onset < trial_onset + raster_stop:
                axis_1.scatter(3000 - (trial_onset - onset), scatter_y, s=scatter_size, c='tab:orange')

        # Plot Block Boundaries
        """
        if trial_onset in block_boundaries or irrel_onset in block_boundaries:
            axis_1.axhline(y=patch_y + 1, c='k')
        """
        # Plot Selected Onsets
        if trial in selected_trials[0]: axis_1.scatter(17, scatter_y, s=scatter_size, c='tab:purple')
        if trial in selected_trials[1]: axis_1.scatter(17, scatter_y, s=scatter_size, c='tab:purple')
        if trial in selected_trials[2]: axis_1.scatter(17, scatter_y, s=scatter_size, c='tab:purple')
        if trial in selected_trials[3]: axis_1.scatter(17, scatter_y, s=scatter_size, c='tab:purple')
        if trial in selected_trials[4]: axis_1.scatter(17, scatter_y, s=scatter_size, c='tab:cyan')


    # Plot Stable Windows
    for window in stable_windows:
        window_start = window[0]
        window_stop = window[-1]
        window_height = window_stop-window_start
        window_patch_y = (number_of_trials - window_stop)

        window_patch = patches.Rectangle(xy=(2800, window_patch_y), height=window_height, width=2, alpha=0.5, color='gold')
        axis_1.add_patch(window_patch)


    plt.savefig(base_directory + "/Session_Behaviour.png")
    plt.close()
    plt.show()




def plot_behaviour_maxtrix_discrimination(base_directory, behaviour_matrix, onsets_dictioanry):

    # Default Settings
    raster_start = 3000
    raster_stop = 5000
    patch_height = 1
    scatter_size = 5

    # Load Onsets
    lick_onsets = onsets_dictioanry["lick_onsets"]
    reward_onsets = onsets_dictioanry["reward_onsets"]
    number_of_trials = np.shape(behaviour_matrix)[0]

    # Setup Figure
    figure_1 = plt.figure(figsize=(12,17))
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.set_xlim([0, raster_start + raster_stop])
    axis_1.set_ylim([0, number_of_trials + 1])

    # Plot Each Trial
    for trial in range(number_of_trials):

        # Plot Trial Patch
        trial_type = behaviour_matrix[trial][1]
        trial_onset = behaviour_matrix[trial][11]
        trial_offset = behaviour_matrix[trial][12]

        patch_colour = get_patch_colour(trial_type)
        patch_y = (number_of_trials - trial) - patch_height
        patch_width = trial_offset - trial_onset
        trial_patch = patches.Rectangle(xy=(raster_start, patch_y), height=patch_height, width=patch_width, alpha=0.5, color=patch_colour)
        axis_1.add_patch(trial_patch)

        # Plot Licks
        scatter_y = (number_of_trials - trial) - 0.5
        for lick in lick_onsets:
            if lick > trial_onset - raster_start and lick < trial_onset + raster_stop:
                axis_1.scatter(3000 - (trial_onset - lick), scatter_y, s=scatter_size, c='k')

        # Plot Reward Onsets
        for onset in reward_onsets:
            if onset > trial_onset - raster_start and onset < trial_onset + raster_stop:
                axis_1.scatter(3000 - (trial_onset - onset), scatter_y, s=scatter_size, c='tab:orange')


    plt.savefig(base_directory + "/Session_Behaviour.png")
    plt.close()
    plt.show()

