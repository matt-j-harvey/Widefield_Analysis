import mat73
import numpy as np

# Load Matlab Data
base_directory = "/media/matthew/29D46574463D2856/Retinotopy_Jade_Mice/KGCA7.5A/2022_04_05_Retinotopy_Left"
base_directory = "/media/matthew/29D46574463D2856/Retinotopy_Jade_Mice/retinotopy_Jade_Mouse_1/KGCA7.1M/2022_04_04_Retinotopic_Mapping_Left"
matlab_filename = "/NXAK12.1F_Right_retinotopy.mat"
matlab_data = mat73.loadmat(base_directory + matlab_filename)




matlab_data = matlab_data['presentationData']
sweeps_per_trial = int(matlab_data['sweeps'])
number_of_trials = int(np.max(matlab_data['trialNumber']))
trial_order = matlab_data['trialType']
trials_per_direction = int(matlab_data['trialsPerDirection'])
display_period = matlab_data['period']


print("Trial Order", trial_order)