import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

"""
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
16 Photodiode Onset = Adjusted Visual stimuli onset to when the photodiode detects the stimulus
17 Photodiode Offset = Adjusted Visual Stimuli Offset to when the photodiode detects the stimulus
"""


## Contains All Nessecary Functions For Calculating Behavioural Measures

def extreme_value_corrections(selected_value, number_of_trials):

    if selected_value == 0:
        selected_value = float(1) / number_of_trials

    elif selected_value == 1:
        selected_value = float((number_of_trials - 1)) / number_of_trials

    return selected_value



def calculate_d_prime(hits, misses, false_alarms, correct_rejections):

    #print("Hits", hits)
    #print("Misses", misses)
    #print("False Alarms", false_alarms)
    #print("Correct Rejections", correct_rejections)

    # Calculate Hit Rates and False Alarm Rates
    number_of_rewarded_trials = hits + misses
    number_of_unrewarded_trials = false_alarms + correct_rejections

    if number_of_unrewarded_trials == 0 or number_of_rewarded_trials == 0:
        return np.nan
    else:

        hit_rate = float(hits) / number_of_rewarded_trials
        false_alarm_rate = float(false_alarms) / number_of_unrewarded_trials

        # Ensure Either Value Does Not Equal Zero or One
        hit_rate = extreme_value_corrections(hit_rate, number_of_rewarded_trials)
        false_alarm_rate = extreme_value_corrections(false_alarm_rate, number_of_unrewarded_trials)

        # Get The Standard Normal Distribution
        Z = norm.ppf

        # Z Transform Both The Hit Rates And The False Alarm Rates
        hit_rate_z_transform = Z(hit_rate)
        false_alarm_rate_z_transform = Z(false_alarm_rate)

        # Calculate D Prime
        d_prime = hit_rate_z_transform - false_alarm_rate_z_transform

    return d_prime

def analyse_visual_discrimination(behaviour_matrix):

    false_alarms = 0
    correct_rejections = 0
    hits = 0
    misses = 0
    trial_outcome_list = []

    # Get Matrix Strucutre
    number_of_trials = np.shape(behaviour_matrix)[0]

    for trial_index in range(number_of_trials):
        trial_data = behaviour_matrix[trial_index]

        trial_type = trial_data[1]

        # Get Outcome
        if trial_type == 1 or trial_type == 2:
            trial_outcome = trial_data[3]
            trial_outcome_list.append(trial_outcome)

        # Get Response
        lick = trial_data[2]

        # Rewarded Visual
        if trial_type == 1:
            if lick == 1:
                hits += 1
            elif lick == 0:
                misses += 1

        elif trial_type == 2:
            if lick == 1:
                false_alarms += 1
            elif lick == 0:
                correct_rejections += 1

    visual_d_prime = calculate_d_prime(hits, misses, false_alarms, correct_rejections)

    return trial_outcome_list, hits, misses, false_alarms, correct_rejections, visual_d_prime



def analyse_odour_discrimination(behaviour_matrix):

    false_alarms = 0
    correct_rejections = 0
    hits = 0
    misses = 0
    trial_outcome_list = []

    # Get Matrix Strucutre
    number_of_trials = np.shape(behaviour_matrix)[0]

    for trial_index in range(number_of_trials):
        trial_data = behaviour_matrix[trial_index]

        trial_type = trial_data[1]

        # Get Outcome
        if trial_type == 3 or trial_type == 4:
            trial_outcome = trial_data[3]
            trial_outcome_list.append(trial_outcome)

        # Get Response
        lick = trial_data[2]

        # Rewarded Visual
        if trial_type == 3:
            if lick == 1:
                hits += 1
            elif lick == 0:
                misses += 1

        elif trial_type == 4:
            if lick == 1:
                false_alarms += 1
            elif lick == 0:
                correct_rejections += 1

    odour_d_prime =calculate_d_prime(hits, misses, false_alarms, correct_rejections)

    return trial_outcome_list, hits, misses, false_alarms, correct_rejections, odour_d_prime


def analyse_irrelevant_performance(behaviour_matrix):

    # Get Matrix Strucutre
    number_of_trials = np.shape(behaviour_matrix)[0]

    irrel_responses = []

    for trial_index in range(number_of_trials):

        trial_data = behaviour_matrix[trial_index]
        trial_type = trial_data[1]

        if trial_type == 3 or trial_type == 4:
            preceeded_by_irrel = trial_data[5]

            if preceeded_by_irrel == 1:
                ignore_irrel = trial_data[7]

                irrel_responses.append(ignore_irrel)


    if len(irrel_responses) > 0:
        irrel_proportion = float(np.sum(irrel_responses)) / len(irrel_responses)
    else:
        irrel_proportion = np.nan
    return irrel_proportion


def get_outcome_of_next_n_trials(behaviour_matrix, trial_index, n=3):

    number_of_trials = np.shape(behaviour_matrix)[0]
    outcome_list = []

    for x in range(1, n+1):
        selected_trial = trial_index + x

        if selected_trial >= number_of_trials:
            return False

        else:
            trial_outcome = behaviour_matrix[selected_trial][3]
            outcome_list.append(trial_outcome)

    if np.sum(outcome_list) == n:
        return True
    else:
        return False




def analyse_transition_proportions(behaviour_matrix):

    transition_outcome_list = []
    # 1 for Perfect transition
    # 0 for Missed transition
    # -1 for Fluke

    # Get Matrix Structure
    number_of_trials = np.shape(behaviour_matrix)[0]

    for trial_index in range(number_of_trials):
        trial_data = behaviour_matrix[trial_index]

        # Check Is First in Block
        first_in_block = trial_data[9]

        # Check Is Vis 1
        trial_type = trial_data[1]

        # If Visual Block VIs 1 and First In BLock
        if trial_type == 1 and first_in_block == 1:

            # Trial Outcome
            trial_outcome = trial_data[3]

            # If Mouse Didnt Lick To IT
            if trial_outcome == 0:

                # Get Outcome Of Next N Trials
                outcome_of_next_n_trials = get_outcome_of_next_n_trials(behaviour_matrix, trial_index)

                # But Mouse Did Lick To Next 3 Trials
                if outcome_of_next_n_trials == True:

                    # Its A Perfect Switch
                    transition_outcome = 1

                # Otherwise Its a Missed Switch
                else:
                    transition_outcome = 0

            # If Mouse Did Lick To It, Its a Fluke
            else:
                transition_outcome = -1


            transition_outcome_list.append(transition_outcome)

    return transition_outcome_list



def check_if_transitioned(behaviour_matrix, count_start):

    index_1 = count_start + 0
    index_2 = count_start + 1
    index_3 = count_start + 2

    # Check There are still 3 Trials Left
    number_of_trials = np.shape(behaviour_matrix)[0]
    if index_3 >= number_of_trials:
        return "Error"

    # Check All 3 Are Rewarded Visuals
    if behaviour_matrix[index_1][1] == 1 and behaviour_matrix[index_2][1] == 1 and behaviour_matrix[index_3][1] == 1:
        if behaviour_matrix[index_1][3] == 1 and behaviour_matrix[index_2][3] == 1 and behaviour_matrix[index_3][3] == 1:
            return True
        else:
            return False
    else:
        return "Error"




def check_if_transitioned_visual_to_odour(behaviour_matrix, count_start):

    index_1 = count_start + 0
    index_2 = count_start + 1
    index_3 = count_start + 2

    # Check There are still 3 Trials Left
    number_of_trials = np.shape(behaviour_matrix)[0]
    if index_3 >= number_of_trials:
        return "Error"

    # Check All 3 Are Preceeded By Rewarded Visuals
    if behaviour_matrix[index_1][6] == 1 and behaviour_matrix[index_2][6] == 1 and behaviour_matrix[index_3][6] == 1:

        # Check All 3 Have Ignored Lick
        if behaviour_matrix[index_1][7] == 1 and behaviour_matrix[index_2][7] == 1 and behaviour_matrix[index_3][7] == 1:
            return True
        else:
            return False
    else:
        return "Error"



def get_odour_to_visual_transition_distribution(behaviour_matrix):

    # Get Transition Trials
    transition_trial_list = []
    number_of_trials = np.shape(behaviour_matrix)[0]
    for trial_index in range(number_of_trials):
        if behaviour_matrix[trial_index][1] == 1 and behaviour_matrix[trial_index][9] == 1:
            transition_trial_list.append(trial_index)

    # For Each Transition Trial
    transition_distribution = []
    for transition_trial in transition_trial_list:

        missed_vis_1_count = 0
        has_transitioned = False
        while has_transitioned == False:
                has_transitioned = check_if_transitioned(behaviour_matrix, transition_trial + missed_vis_1_count)

                if has_transitioned == True:
                    transition_distribution.append(missed_vis_1_count)

                elif has_transitioned == False:
                    missed_vis_1_count += 1

                elif has_transitioned == "Error":
                    transition_distribution.append(np.nan)
                    break


    return transition_distribution




def get_visual_to_odour_transition_distribution(behaviour_matrix):

    # Get Transition Trials
    # First Trial In Block + Rewarded Odour
    transition_trial_list = []
    number_of_trials = np.shape(behaviour_matrix)[0]
    for trial_index in range(number_of_trials):
        if behaviour_matrix[trial_index][1] == 3 and behaviour_matrix[trial_index][9] == 1:
            transition_trial_list.append(trial_index)

    # For Each Transition Trial
    transition_distribution = []
    for transition_trial in transition_trial_list:

        lick_to_irrel_vis_1_count = 0
        has_transitioned = False
        while has_transitioned == False:
                has_transitioned = check_if_transitioned_visual_to_odour(behaviour_matrix, transition_trial + lick_to_irrel_vis_1_count)

                if has_transitioned == True:
                    transition_distribution.append(lick_to_irrel_vis_1_count)

                elif has_transitioned == False:
                    lick_to_irrel_vis_1_count += 1

                elif has_transitioned == "Error":
                    transition_distribution.append(np.nan)
                    break


    return transition_distribution



def analyse_visual_performance_excluding_transitions(behaviour_matrix):

    # Exclude visual trials when going from odour to visual until mouse has transitioned

    # Get Indexes Of Trials To Exclude

    # Get Actual Transition Trials
    transition_trial_list = []
    number_of_trials = np.shape(behaviour_matrix)[0]
    for trial_index in range(number_of_trials):
        if behaviour_matrix[trial_index][1] == 1 and behaviour_matrix[trial_index][9] == 1:
            transition_trial_list.append(trial_index)

    # For Each Transition Trial
    pre_transition_misses = []
    for transition_trial in transition_trial_list:

        missed_vis_1_count = 0
        has_transitioned = False
        while has_transitioned == False:
            has_transitioned = check_if_transitioned(behaviour_matrix, transition_trial + missed_vis_1_count)

            if has_transitioned == False:
                pre_transition_misses.append(transition_trial + missed_vis_1_count)
                missed_vis_1_count += 1

            elif has_transitioned == "Error":
                break

    trials_to_exclude = transition_trial_list + pre_transition_misses

    # Now Analyse D Prime as Normal - Excluding These Trials

    false_alarms = 0
    correct_rejections = 0
    hits = 0
    misses = 0
    trial_outcome_list = []

    # Get Matrix Strucutre
    number_of_trials = np.shape(behaviour_matrix)[0]

    for trial_index in range(number_of_trials):

        if trial_index not in trials_to_exclude:
            trial_data = behaviour_matrix[trial_index]
            trial_type = trial_data[1]

            # Get Outcome
            if trial_type == 1 or trial_type == 2:
                trial_outcome = trial_data[3]
                trial_outcome_list.append(trial_outcome)

            # Get Response
            lick = trial_data[2]

            # Rewarded Visual
            if trial_type == 1:
                if lick == 1:
                    hits += 1
                elif lick == 0:
                    misses += 1

            elif trial_type == 2:
                if lick == 1:
                    false_alarms += 1
                elif lick == 0:
                    correct_rejections += 1

    visual_d_prime = calculate_d_prime(hits, misses, false_alarms, correct_rejections)

    return [trial_outcome_list, hits, misses, false_alarms, correct_rejections, visual_d_prime, trials_to_exclude]


def analyse_visual_d_prime_of_selected_trials(behaviour_matrix, selected_trials):

    false_alarms = 0
    correct_rejections = 0
    hits = 0
    misses = 0

    for trial_index in selected_trials:

        # Get Trial Data
        trial_data = behaviour_matrix[trial_index]

        # Get Trial Type
        trial_type = trial_data[1]
        lick = trial_data[2]

        # If Rewarded In Visual Block
        if trial_type == 1:
            if lick == 1:
                hits += 1
            elif lick == 0:
                misses += 1

        # If Unrewarded In Odour Block
        elif trial_type == 2:
            if lick == 1:
                false_alarms += 1
            elif lick == 0:
                correct_rejections += 1

        # Irrel Rewarded
        irrel_type = trial_data[6]
        ignore_irrel = trial_data[7]

        if irrel_type == 1:
            if ignore_irrel == 0:
                hits += 1
            elif ignore_irrel == 1:
                misses += 1

        elif irrel_type == 2:
            if ignore_irrel == 0:
                false_alarms += 1
            elif ignore_irrel == 1:
                correct_rejections += 1


    visual_d_prime = calculate_d_prime(hits, misses, false_alarms, correct_rejections)

    return visual_d_prime


def analyse_odour_d_prime_of_selected_trials(behaviour_matrix, selected_trials):
    false_alarms = 0
    correct_rejections = 0
    hits = 0
    misses = 0

    for trial_index in selected_trials:

        # Get Trial Data
        trial_data = behaviour_matrix[trial_index]

        # Get Trial Type
        trial_type = trial_data[1]

        # Get Response
        lick = trial_data[2]

        # Rewarded Visual
        if trial_type == 3:
            if lick == 1:
                hits += 1
            elif lick == 0:
                misses += 1

        elif trial_type == 4:
            if lick == 1:
                false_alarms += 1
            elif lick == 0:
                correct_rejections += 1

    odour_d_prime = calculate_d_prime(hits, misses, false_alarms, correct_rejections)
    return odour_d_prime


def calculate_blockwise_d_prime(behaviour_matrix):

    block_starts = []
    block_types = [] # 0 = visual block, 1 = odour block

    number_of_trials = np.shape(behaviour_matrix)[0]
    for trial_index in range(number_of_trials):
        trial_data = behaviour_matrix[trial_index]

        # Check If First In Block
        if trial_data[9] == 1:
            block_starts.append(trial_index)
            #print("First in trial")
            #print(trial_index)

            trial_type = trial_data[1]
            if trial_type == 1 or trial_type == 2:
                block_types.append(0)
            elif trial_type == 3 or trial_type == 4:
                block_types.append(1)

    block_stops = block_starts[1:]
    block_stops.append(number_of_trials-1)

    number_of_blocks = len(block_starts)
    block_odour_performance = []
    block_visual_performance = []

    #print("Block Starts", block_starts)
    #print("Block Stops", block_stops)
    #print("Block Types", block_types)

    for block_index in range(number_of_blocks):

        block_start = block_starts[block_index]
        block_stop = block_stops[block_index]
        block_trials = list(range(block_start, block_stop))
        block_type = block_types[block_index]

        if block_type == 0:
            block_visual_performance.append(analyse_visual_d_prime_of_selected_trials(behaviour_matrix, block_trials))
            block_odour_performance.append(np.nan)

        elif block_type == 1:
            block_visual_performance.append(analyse_visual_d_prime_of_selected_trials(behaviour_matrix, block_trials))
            block_odour_performance.append(analyse_odour_d_prime_of_selected_trials(behaviour_matrix, block_trials))


    return block_visual_performance, block_odour_performance


