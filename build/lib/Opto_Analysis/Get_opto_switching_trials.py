import os
from scipy.io import loadmat
import numpy as np

from Behaviour_Analysis import Create_Behaviour_Matrix_Switching


def get_matlab_filename(base_directory):
    file_list = os.listdir(base_directory)
    for file_name in file_list:
        if file_name[-4:] == ".mat":
            return file_name
    return None


def get_opto_trials(base_directory):

    # Get Matlab Filename
    matlab_filename = get_matlab_filename(base_directory)

    # Load Matlab Data
    matlab_data = loadmat(os.path.join(base_directory, matlab_filename))

    matlab_data = matlab_data['fsm'][0][0]

    count = 0
    for field in matlab_data:
        print("Field: ", str(count))
        print(field)
        count += 1

    laser_powers = matlab_data[50][0]
    laser_trials_binary = np.where(laser_powers > 0, 1, 0)
    laser_trial_numbers = np.nonzero(laser_trials_binary)[0]

    # Save Laser Trial Details
    save_directory = os.path.join(base_directory, "Stimuli_Onsets")
    np.save(os.path.join(save_directory, "Laser_Trial_Numbers.npy"), laser_trial_numbers)
    np.save(os.path.join(save_directory, "Laser_Trial_Powers.npy"), laser_powers)


def classify_opto_trials(base_directory):

    # Load Trial Matrix
    behaviour_matrix = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)
    print("Behaviour Matrix Shape", np.shape(behaviour_matrix))

    # Load Opto Trials
    laser_powers = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Laser_Trial_Powers.npy"))
    laser_trials = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Laser_Trial_Numbers.npy"))
    print("Laser Powers", len(laser_powers))
    print("laser_trials", laser_trials)

    # visual_context_vis_1_opto_correct
    # visual_context_vis_1_opto_incorrect
    # visual_context_vis_1_nonopto_correct
    # visual_context_vis_1_nonopto_incorrect

    # visual_context_vis_2_opto_correct
    # visual_context_vis_2_opto_incorrect
    # visual_context_vis_2_nonopto_correct
    # visual_context_vis_2_nonopto_incorrect

    # visual_context_vis_1_opto_correct
    # visual_context_vis_1_opto_incorrect
    # visual_context_vis_1_nonopto_correct
    # visual_context_vis_1_nonopto_incorrect

    # visual_context_vis_2_opto_correct
    # visual_context_vis_2_opto_incorrect
    # visual_context_vis_2_nonopto_correct
    # visual_context_vis_2_nonopto_incorrect

    visual_context_vis_1_opto_correct_list = []
    visual_context_vis_1_opto_incorrect_list = []
    visual_context_vis_1_nonopto_correct_list = []
    visual_context_vis_1_nonopto_incorrect_list = []

    visual_context_vis_2_opto_correct_list = []
    visual_context_vis_2_opto_incorrect_list = []
    visual_context_vis_2_nonopto_correct_list = []
    visual_context_vis_2_nonopto_incorrect_list = []

    odour_context_vis_1_opto_correct_list = []
    odour_context_vis_1_opto_incorrect_list = []
    odour_context_vis_1_nonopto_correct_list = []
    odour_context_vis_1_nonopto_incorrect_list = []

    odour_context_vis_2_opto_correct_list = []
    odour_context_vis_2_opto_incorrect_list = []
    odour_context_vis_2_nonopto_correct_list = []
    odour_context_vis_2_nonopto_incorrect_list = []

    number_of_trials = np.shape(behaviour_matrix)[0]
    for trial_index in range(number_of_trials):
        trial_data = behaviour_matrix[trial_index]

        # Extract Trial Data
        trial_type = trial_data[1]
        correct = trial_data[3]
        preceeded_by_visual = trial_data[5]
        irrel_type = trial_data[6]
        ignored_irrel = trial_data[7]
        stimuli_onset_frame = trial_data[18]
        irrel_onset_frame = trial_data[20]
        
        # Determine If Is Opto
        if trial_index in laser_trials:
            is_opto_trial = True
        else:
            is_opto_trial = False
            

        # Classify Trial


        # Vis Context Vis 1
        if trial_type == 1:

            if is_opto_trial == True:
                if correct == 1:
                    visual_context_vis_1_opto_correct_list.append(stimuli_onset_frame)
                elif correct == 0:
                    visual_context_vis_1_opto_incorrect_list.append(stimuli_onset_frame)

            if is_opto_trial == False:
                if correct == 1:
                    visual_context_vis_1_nonopto_correct_list.append(stimuli_onset_frame)
                elif correct == 0:
                    visual_context_vis_1_nonopto_incorrect_list.append(stimuli_onset_frame)


        # Vis Context Vis 2
        if trial_type == 2:

            if is_opto_trial == True:
                if correct == 1:
                    visual_context_vis_2_opto_correct_list.append(stimuli_onset_frame)
                elif correct == 0:
                    visual_context_vis_2_opto_incorrect_list.append(stimuli_onset_frame)

            if is_opto_trial == False:
                if correct == 1:
                    visual_context_vis_2_nonopto_correct_list.append(stimuli_onset_frame)
                elif correct == 0:
                    visual_context_vis_2_nonopto_incorrect_list.append(stimuli_onset_frame)


        # Odour Context Vis 1
        if trial_type == 3 or trial_type == 4:
            if preceeded_by_visual:

                if irrel_type == 1:

                    if is_opto_trial == 1:
                        if ignored_irrel == 1:
                            odour_context_vis_1_opto_correct_list.append(irrel_onset_frame)
                        elif ignored_irrel == 0:
                            odour_context_vis_1_opto_incorrect_list.append(irrel_onset_frame)

                    if is_opto_trial == 0:
                        if ignored_irrel == 1:
                            odour_context_vis_1_nonopto_correct_list.append(irrel_onset_frame)
                        elif ignored_irrel == 0:
                            odour_context_vis_1_nonopto_incorrect_list.append(irrel_onset_frame)

                elif irrel_type == 2:

                    if is_opto_trial == 1:
                        if ignored_irrel == 1:
                            odour_context_vis_2_opto_correct_list.append(irrel_onset_frame)
                        elif ignored_irrel == 0:
                            odour_context_vis_2_opto_incorrect_list.append(irrel_onset_frame)

                    if is_opto_trial == 0:
                        if ignored_irrel == 1:
                            odour_context_vis_2_nonopto_correct_list.append(irrel_onset_frame)
                        elif ignored_irrel == 0:
                            odour_context_vis_2_nonopto_incorrect_list.append(irrel_onset_frame)

    # Print Lengths
    print("visual_context_vis_1_opto_correct_list", len(visual_context_vis_1_opto_correct_list))
    print("visual_context_vis_1_opto_incorrect_list", len(visual_context_vis_1_opto_incorrect_list))
    print("visual_context_vis_1_nonopto_correct_list", len(visual_context_vis_1_nonopto_correct_list))
    print("visual_context_vis_1_nonopto_incorrect_list", len(visual_context_vis_1_nonopto_incorrect_list))

    print("visual_context_vis_2_opto_correct_list", len(visual_context_vis_2_opto_correct_list))
    print("visual_context_vis_2_opto_incorrect_list", len(visual_context_vis_2_opto_incorrect_list))
    print("visual_context_vis_2_nonopto_correct_list", len(visual_context_vis_2_nonopto_correct_list))
    print("visual_context_vis_2_nonopto_incorrect_list", len(visual_context_vis_2_nonopto_incorrect_list))

    print("odour_context_vis_1_opto_correct_list", len(odour_context_vis_1_opto_correct_list))
    print("odour_context_vis_1_opto_incorrect_list", len(odour_context_vis_1_opto_incorrect_list))
    print("odour_context_vis_1_nonopto_correct_list", len(odour_context_vis_1_nonopto_correct_list))
    print("odour_context_vis_1_nonopto_incorrect_list", len(odour_context_vis_1_nonopto_incorrect_list))

    print("odour_context_vis_2_opto_correct_list", len(odour_context_vis_2_opto_correct_list))
    print("odour_context_vis_2_opto_incorrect_list", len(odour_context_vis_2_opto_incorrect_list))
    print("odour_context_vis_2_nonopto_correct_list", len(odour_context_vis_2_nonopto_correct_list))
    print("odour_context_vis_2_nonopto_incorrect_list", len(odour_context_vis_2_nonopto_incorrect_list))



    # Save These
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "visual_context_vis_1_opto_correct_onset_frames.npy"), visual_context_vis_1_opto_correct_list)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "visual_context_vis_1_opto_incorrect_onset_frames.npy"), visual_context_vis_1_opto_incorrect_list)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "visual_context_vis_1_nonopto_correct_onset_frames.npy"), visual_context_vis_1_nonopto_correct_list)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "visual_context_vis_1_nonopto_incorrect_onset_frames.npy"), visual_context_vis_1_nonopto_incorrect_list)

    np.save(os.path.join(base_directory, "Stimuli_Onsets", "visual_context_vis_2_opto_correct_onset_frames.npy"), visual_context_vis_2_opto_correct_list)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "visual_context_vis_2_opto_incorrect_onset_frames.npy"), visual_context_vis_2_opto_incorrect_list)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "visual_context_vis_2_nonopto_correct_onset_frames.npy"), visual_context_vis_2_nonopto_correct_list)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "visual_context_vis_2_nonopto_incorrect_onset_frames.npy"), visual_context_vis_2_nonopto_incorrect_list)

    np.save(os.path.join(base_directory, "Stimuli_Onsets", "odour_context_vis_1_opto_correct_onset_frames.npy"), odour_context_vis_1_opto_correct_list)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "odour_context_vis_1_opto_incorrect_onset_frames.npy"), odour_context_vis_1_opto_incorrect_list)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "odour_context_vis_1_nonopto_correct_onset_frames.npy"), odour_context_vis_1_nonopto_correct_list)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "odour_context_vis_1_nonopto_incorrect_onset_frames.npy"), odour_context_vis_1_nonopto_incorrect_list)

    np.save(os.path.join(base_directory, "Stimuli_Onsets", "odour_context_vis_2_opto_correct_onset_frames.npy"), odour_context_vis_2_opto_correct_list)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "odour_context_vis_2_opto_incorrect_onset_frames.npy"), odour_context_vis_2_opto_incorrect_list)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "odour_context_vis_2_nonopto_correct_onset_frames.npy"), odour_context_vis_2_nonopto_correct_list)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "odour_context_vis_2_nonopto_incorrect_onset_frames.npy"), odour_context_vis_2_nonopto_incorrect_list)


# Set Base Directory
base_directory = r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Switching_Opto/KPGC2.2G/2022_12_08_Switching_Opto"

# Create Behaviour Matrix
#Create_Behaviour_Matrix_Switching.create_behaviour_matrix(base_directory)

# Get Opto Trials
get_opto_trials(base_directory)
classify_opto_trials(base_directory)