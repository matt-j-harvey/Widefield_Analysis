import numpy as np
import os
import random
import matplotlib.pyplot as plt

from Files import Session_List



def get_specific_reaction_times(session_list, window=[900, 1000]):

    session_onset_list = []
    session_reaction_times = []

    trial_count = 0
    # Iterate Through Session List
    for base_directory in session_list:

        # Create Lists To Hold Stimuli Onsets
        matched_reaction_times_stimuli_onsets = []
        reaction_times = []

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

                if onset_frame != None and lick_frame != None:
                    if onset_frame > 3000:

                        if reaction_time > window[0] and reaction_time < window[1]:
                            matched_reaction_times_stimuli_onsets.append(onset_frame)
                            reaction_times.append(reaction_time)
                            trial_count += 1

        session_onset_list.append(matched_reaction_times_stimuli_onsets)
        session_reaction_times.append(reaction_times)
    return trial_count, session_onset_list,session_reaction_times



def balance_interval(stimuli_to_remove, session_list, reaction_time_list):

    number_of_sessions = len(session_list)

    stimuli_removed = 0
    while stimuli_removed != stimuli_to_remove:

        # Get Largest List
        session_length_list = []
        for session_index in range(number_of_sessions):
            session_length = len(session_list[session_index])
            session_length_list.append(session_length)

        max_length = np.max(session_length_list)
        biggest_session_index = session_length_list.index(max_length)
        trial_to_remove = random.randrange(max_length)
        session_list[biggest_session_index].pop(trial_to_remove)
        reaction_time_list[biggest_session_index].pop(trial_to_remove)
        stimuli_removed += 1

    return session_list, reaction_time_list





def match_distributions(control_session_list, mutant_session_list):

    number_of_control_sessions = len(control_session_list)
    number_of_mutant_sessions = len(mutant_session_list)

    control_onsets_final = []
    for mouse in range(number_of_control_sessions):
        control_onsets_final.append([])

    mutant_onsets_final = []
    for mouse in range(number_of_mutant_sessions):
        mutant_onsets_final.append([])


    control_reaction_times = []
    mutant_reaction_times = []

    window_size = 50
    for x in range(500, 2000, window_size):
        window_start = x
        window_stop = x + window_size

        control_stimuli_count, control_session_onset_list, control_session_reaction_times = get_specific_reaction_times(control_session_list, window=[window_start, window_stop])
        mutant_stimuli_count, mutant_session_onset_list, mutant_session_reaction_times = get_specific_reaction_times(mutant_session_list, window=[window_start, window_stop])

        print("Pre Balance", "Mutant", mutant_stimuli_count, "Control", control_stimuli_count)

        if control_stimuli_count > mutant_stimuli_count:
            stimuli_to_remove = control_stimuli_count - mutant_stimuli_count
            control_session_onset_list, control_session_reaction_times = balance_interval(stimuli_to_remove, control_session_onset_list, control_session_reaction_times)

        if mutant_stimuli_count > control_stimuli_count:
            stimuli_to_remove = mutant_stimuli_count - control_stimuli_count
            mutant_session_onset_list, mutant_session_reaction_times = balance_interval(stimuli_to_remove, mutant_session_onset_list, mutant_session_reaction_times)


        for session in mutant_session_reaction_times:
            for time in session:
                mutant_reaction_times.append(time)

        for session in control_session_reaction_times:
            for time in session:
                control_reaction_times.append(time)

        #mutant_reaction_times = mutant_reaction_times + mutant_session_reaction_times
        #control_reaction_times = control_reaction_times + control_session_reaction_times

        # Add These To Lists To Save
        for session_index in range(number_of_control_sessions):
            onsets_to_add = control_session_onset_list[session_index]
            for onset in onsets_to_add:
                control_onsets_final[session_index].append(onset)

        # Add These To Lists To Save
        for session_index in range(number_of_mutant_sessions):
            onsets_to_add = mutant_session_onset_list[session_index]
            for onset in onsets_to_add:
                mutant_onsets_final[session_index].append(onset)

    # Save These
    for session_index in range(number_of_mutant_sessions):
        session = mutant_session_list[session_index]
        onsets = mutant_onsets_final[session_index]
        np.save(os.path.join(session, "Stimuli_Onsets", "Mixed_Effects_Distribution_Matched_Onsets.npy"), onsets)

    # Save These
    for session_index in range(number_of_control_sessions):
        session = control_session_list[session_index]
        onsets = control_onsets_final[session_index]
        np.save(os.path.join(session, "Stimuli_Onsets", "Mixed_Effects_Distribution_Matched_Onsets.npy"), onsets)

    print("Number OF Trials", len(control_reaction_times), "+", len(mutant_reaction_times))
    plt.hist(control_reaction_times, color='b', alpha=0.5)
    plt.hist(mutant_reaction_times, color='g', alpha=0.5)
    plt.show()



def split_session_list_into_pre_and_post(nested_session_list):

    pre_learning_sessions = []
    post_learning_sessions = []

    for mouse in nested_session_list:

        mouse_pre_learning_sessions = mouse[0]
        mouse_post_learning_sessions = mouse[1]

        for session in mouse_pre_learning_sessions:
            pre_learning_sessions.append(session)

        for session in mouse_post_learning_sessions:
            post_learning_sessions.append(session)

    return pre_learning_sessions, post_learning_sessions



# Load Nested Session List
controls_learning_nested_session_list = Session_List.expanded_controls_learning_nested
mutants_learning_nested_session_list = Session_List.expanded_mutants_learning_nested

# Split Into Pre and Post
print("Control session list", mutants_learning_nested_session_list)
pre_learning_sessions, post_learning_sessions = split_session_list_into_pre_and_post(mutants_learning_nested_session_list)

print("Pre Learning Sessions", pre_learning_sessions)
print("post Learning sessions", post_learning_sessions)
match_distributions(pre_learning_sessions, post_learning_sessions)