import numpy as np
import tables

def test_sampling():

    # Create Pseudo-data
    actual_control_trial_numbers = [3, 4, 5, 4, 6]
    actual_mutant_trial_numbers = [4, 6, 3, 5, 6]
    number_of_timepoints = 5
    number_of_pixels = 10

    control_activity_tensors = [
        np.random.randint(low=0, high=1, size=(actual_control_trial_numbers[0], number_of_timepoints, number_of_pixels)),
        np.random.randint(low=1, high=2, size=(actual_control_trial_numbers[1], number_of_timepoints, number_of_pixels)),
        np.random.randint(low=2, high=3, size=(actual_control_trial_numbers[2], number_of_timepoints, number_of_pixels)),
        np.random.randint(low=3, high=4, size=(actual_control_trial_numbers[3], number_of_timepoints, number_of_pixels)),
        np.random.randint(low=4, high=5, size=(actual_control_trial_numbers[4], number_of_timepoints, number_of_pixels)),
    ]

    mutant_activity_tensors = [
        np.random.randint(low=5, high=6, size=(actual_mutant_trial_numbers[0], number_of_timepoints, number_of_pixels)),
        np.random.randint(low=6, high=7, size=(actual_mutant_trial_numbers[1], number_of_timepoints, number_of_pixels)),
        np.random.randint(low=7, high=8, size=(actual_mutant_trial_numbers[2], number_of_timepoints, number_of_pixels)),
        np.random.randint(low=8, high=9, size=(actual_mutant_trial_numbers[3], number_of_timepoints, number_of_pixels)),
        np.random.randint(low=9, high=10, size=(actual_mutant_trial_numbers[4], number_of_timepoints, number_of_pixels)),
    ]

    sample_data, sample_labels = get_data_sample(control_activity_tensors, mutant_activity_tensors, timepoint_index=0)
    print("Sample Data")
    print(sample_data)
    print("Sample Labels")
    print(sample_labels)



def get_trial_numbers(control_activity_tensors, mutant_activity_tensors):

    # Get Trial Numbers
    control_trials_per_session = []
    for session in control_activity_tensors:
        control_trials_per_session.append(np.shape(session)[0])

    mutant_trials_per_session = []
    for session in mutant_activity_tensors:
        mutant_trials_per_session.append(np.shape(session)[0])

    # Get Smallest Number Of Trials In A Session
    combined_trial_numbers = control_trials_per_session + mutant_trials_per_session
    minimum_number_of_trials = np.min(combined_trial_numbers)
    print("Minimum Number Of Trials", minimum_number_of_trials)

    return control_trials_per_session, mutant_trials_per_session, minimum_number_of_trials


def create_trial_index_pools(control_trials_per_session, mutant_trials_per_session):

    control_trial_pool = []
    for mouse_trial_number in control_trials_per_session:
        mouse_pool = list(range(0, mouse_trial_number))
        control_trial_pool.append(mouse_pool)

    mutant_trial_pool = []
    for mouse_trial_number in mutant_trials_per_session:
        mouse_pool = list(range(0, mouse_trial_number))
        mutant_trial_pool.append(mouse_pool)

    return control_trial_pool, mutant_trial_pool



def get_data_sample(combined_file, control_trials_per_session, mutant_trials_per_session, timepoint_index, baseline_correction=False):

    # Input Data Are Lists of Tensors - of Shape (N_Trials, Trial_Length, N_Pixels)
    # Ensure Equal Number Of Samples Per Genotype
    # Ensure Equal Number Of Samples Per Mouse

    # Get Number Of Mice
    number_of_control_mice = len(control_trials_per_session)
    number_of_mutant_mice = len(mutant_trials_per_session)
    if number_of_control_mice != number_of_mutant_mice:
        print("Unequal Group Numbers")
        return None

    # Get The Minimum Number of Trials In A Session
    minimum_number_of_trials = np.min([control_trials_per_session, mutant_trials_per_session])
    print("Minimum Number Of Trials", minimum_number_of_trials)

    # Create Trial Index Pools
    control_trial_pool, mutant_trial_pool = create_trial_index_pools(control_trials_per_session, mutant_trials_per_session)

    # Open File
    file_container = tables.open_file(combined_file, "r")

    sample_data = []
    sample_labels = []
    for trial_index in range(minimum_number_of_trials):

        # Iterate Through Control Mice
        mouse_index = 0
        for array in file_container.list_nodes(where="/Controls"):

            # Select Random Trial
            selected_trial = control_trial_pool[mouse_index].pop(np.random.randint(low=0, high=len(control_trial_pool[mouse_index])))

            # Extract This Trials Data
            trial_data = array[selected_trial, timepoint_index]


            # Add This To Our Sample
            sample_data.append(trial_data)

            # Add A Corresponding Label To Our Smaple
            sample_labels.append(0)
            mouse_index += 1


        # Iterate Through Mutant Mice
        mouse_index = 0
        for array in file_container.list_nodes(where="/Mutants"):

            # Select Random Trial
            selected_trial = mutant_trial_pool[mouse_index].pop(np.random.randint(low=0, high=len(mutant_trial_pool[mouse_index])))

            # Extract This Trials Data
            trial_data = array[selected_trial, timepoint_index]

            # Add This To Our Sample
            sample_data.append(trial_data)

            # Add A Corresponding Label To Our Smaple
            sample_labels.append(1)
            mouse_index += 1

    sample_data = np.array(sample_data)
    sample_labels = np.array(sample_labels)
    file_container.close()

    return sample_data, sample_labels

