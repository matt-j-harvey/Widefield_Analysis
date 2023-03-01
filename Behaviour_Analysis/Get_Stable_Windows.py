import numpy as np


def propagate_visual_block(trial_index, behaviour_matrix, accuracy_thershold):

    block_size = 0
    block_outcomes = []
    number_of_trials = np.shape(behaviour_matrix)[0]

    still_propagating = True
    while still_propagating:

        # If We Have Reached The End Then Stop
        if trial_index == number_of_trials:
            still_propagating = False

        else:
            current_trial = behaviour_matrix[trial_index]

            # If It Becomes an Odour Block Then Stop
            current_trial_type = current_trial[1]
            if current_trial_type != 1 and current_trial_type != 2:
                still_propagating = False

            # If The Addition of This Trial Causes Outcome Percentage To Drop Too Low Then Stop
            current_trial_outcome = current_trial[3]
            block_outcomes.append(current_trial_outcome)
            if np.mean(block_outcomes) < accuracy_thershold:
                still_propagating = False

            # Else Continue
            if still_propagating:
                trial_index += 1
                block_size += 1

    print("Visual block size", block_size)

    return block_size




def propagate_odour_block(trial_index, behaviour_matrix, accuracy_threshold, irrelevance_threshold):

    block_size = 0
    block_accuracy_outcomes = []
    block_irrelevance_outcomes = []

    number_of_trials = np.shape(behaviour_matrix)[0]

    still_propagating = True
    while still_propagating:

        # If We Have Reached The End Then Stop
        if trial_index == number_of_trials:
            still_propagating = False

        else:
            current_trial = behaviour_matrix[trial_index]

            # If It Becomes a Visual Block Then Stop
            current_trial_type = current_trial[1]
            if current_trial_type == 1 or current_trial_type == 2:
                still_propagating = False

            # If The Addition of This Trial Causes Outcome Percentage To Drop Too Low Then Stop
            current_trial_outcome = current_trial[3]
            block_accuracy_outcomes.append(current_trial_outcome)
            if np.mean(block_accuracy_outcomes) < accuracy_threshold:
                still_propagating = False

            # If The Addition Of THis Trial Causes Irrelevance Percentage To Drop Too Low Then Stop
            preceeded_by_irrel = current_trial[4]
            if preceeded_by_irrel:
                irrel_outcome = current_trial[6]
                block_irrelevance_outcomes.append(irrel_outcome)
                if np.mean(block_irrelevance_outcomes) < irrelevance_threshold:
                    still_propagating = False

            # Else Continue
            if still_propagating:
                trial_index += 1
                block_size += 1

    return block_size




def get_largest_window_per_block(potential_window_size_list, behaviour_matrix, window_min_size=10):

    # Get Largest Window In Block
    stable_windows = []

    current_largest_window = 0
    current_largest_window_index = 0
    current_block = behaviour_matrix[0][8]

    number_of_trials = np.shape(behaviour_matrix)[0]

    for trial in range(number_of_trials):
        trial_block = behaviour_matrix[trial][8]
        trial_window_size = potential_window_size_list[trial]

        if trial_block != current_block:
            print("New Block")

            # Check If Windows Large Enough
            if current_largest_window >= window_min_size:
                stable_windows.append(list(range(current_largest_window_index, current_largest_window_index + current_largest_window)))

            # Reset Everything
            current_block = trial_block
            current_largest_window = trial_window_size
            current_largest_window_index = trial

        elif trial_block == current_block:
            if trial_window_size > current_largest_window:
                current_largest_window = trial_window_size
                current_largest_window_index = trial

            if trial == number_of_trials-1:
                if current_largest_window >= window_min_size:
                    stable_windows.append(list(range(current_largest_window_index, current_largest_window_index + current_largest_window)))



    print("Potential Block Sizes:")
    print(potential_window_size_list)
    print("Trial Blocks:")
    print(list(behaviour_matrix[:, 8]))
    print("Stable Windows")
    print(stable_windows)
    print("Number of stable windows", len(stable_windows))

    print("Potential block sizes list")
    print(len(potential_window_size_list))

    print("Number of trials", np.shape(behaviour_matrix)[0])

    return stable_windows



def get_stable_windows(behaviour_matrix):

    """
    Visual Context -
    Moving Window
    75% correct Vis 1
    At least 10 trials

    Odour Context -
    Moving window
    75% both vis 1 and vis 2
    75% both odour 1 and odour 2
    At least 10 trials


    0 trial_index = int, index of trial
    1 trial_type = 1 - rewarded visual, 2 - unrewarded visual, 3 - rewarded odour, 4 - unrewarded odour
    2 lick = 1- lick, 0 - no lick
    3 correct = 1 - correct, 0 - incorrect
    4 rewarded = 1- yes, 0 - no
    5 preeceded_by_irrel = 0 - no, 1 - yes
    6 irrel_type = 1 - rewarded grating, 2 - unrearded grating
    7 ignore_irrel = 0 - licked to irrel, 1 - ignored irrel, nan - no irrel,
    8 block_number = int, index of block
    9 first_in_block = 1 - yes, 2- no
    10 in_block_of_stable_performance = 1 - yes, 2 - no
    11 onset = float onset of major stimuli
    12 stimuli_offset = float offset of major stimuli
    13 irrel_onset = float onset of any irrel stimuli, nan = no irrel stimuli
    14 irrel_offset = float offset of any irrel stimuli, nan = no irrel stimuli
    15 trial_end = float end of trial
    """

    accuracy_threshold = 0.75
    irrel_threshold    = 0.75

    # Get Number of Trials
    number_of_trials = np.shape(behaviour_matrix)[0]

    # For Each Trial Get Potential Block Size
    potential_window_size_list = []

    for trial in range(number_of_trials):
        trial_type = behaviour_matrix[trial][1]

        # Check If Visual
        if trial_type == 1 or trial_type == 2:
            potential_window_size = propagate_visual_block(trial, behaviour_matrix, accuracy_threshold)
            potential_window_size_list.append(potential_window_size)

        # Check If Odour
        elif trial_type == 3 or trial_type == 4:
            potential_window_size = propagate_odour_block(trial, behaviour_matrix, accuracy_threshold, irrel_threshold)
            potential_window_size_list.append(potential_window_size)

    stable_windows = get_largest_window_per_block(potential_window_size_list, behaviour_matrix)

    print("Stable_Windows", stable_windows)

    return stable_windows

