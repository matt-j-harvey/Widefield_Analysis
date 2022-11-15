import os
import matplotlib.pyplot as plt
import mat73
import numpy as np

def get_matlab_filename(base_directory):

    file_list = os.listdir(base_directory)
    for file in file_list:
        file_split = file.split(".")
        if file_split[-1] == 'mat':
            return "/" + file


def check_existing_file(base_directory):

    # Load Matlab Data
    matlab_filename = get_matlab_filename(base_directory)
    matlab_data = mat73.loadmat(base_directory + matlab_filename)

    print("Matlab Data")
    print(matlab_data)
    matlab_data = matlab_data['presentationData']
    sweeps_per_trial = int(matlab_data['sweeps'])
    number_of_trials = int(np.max(matlab_data['trialNumber']))
    trial_order = matlab_data['trialType']
    trials_per_direction = int(matlab_data['trialsPerDirection'])
    display_period = matlab_data['period']
    print("Trial Type", trial_order)


def create_log_with_new_order(base_directory_to_copy, new_save_directory, new_stim_order):
    
    # Load Matlab Data
    matlab_filename = get_matlab_filename(base_directory)
    matlab_data = mat73.loadmat(os.path.join(base_directory, matlab_filename))

    # Assign New Stim ORder
    matlab_data['trialType'] = new_stim_order
    
    # Save This
    mat73.savemat(matlab_data)
    
base_directory = r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Retinotopy/NXAK14.1A/Continous_Retinotopic_Mapping_Left"
check_existing_file(base_directory)

new_trial_order = [2,2,2,1,2,1,1,2,2,2,1,1,1,1,1,2,1,2,2,1]