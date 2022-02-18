import sys
import numpy
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")
sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Analysis/Movement_Controls/Bodycam_Analysis")

import Widefield_General_Functions
import Get_Bodycam_SVD_Tensor



def decode_context_using_behaviour(base_directory, condition_1_onets, condition_2_onsets, start_window, stop_window):

    # Get Video File Name
    bodycam_file, eyecam_file = Widefield_General_Functions.get_mousecam_files(base_directory)

    # Get Mousecam Tensors
    condition_1_bodycam_tensor, condition_2_bodycam_tensor, bodycam_components = Get_Bodycam_SVD_Tensor.get_bodycam_tensor_multiple_conditions(base_directory, bodycam_file, [condition_1_onets], [condition_2_onsets], start_window, stop_window)
    print("condition 1 tensor", np.shape(condition_1_bodycam_tensor))
    print("condition 1 tensor", np.shape(condition_2_bodycam_tensor))

    # Get Data Structure
    number_of_condition_1_trials = np.shape(condition_1_bodycam_tensor)[0]
    number_of_condition_2_trials = np.shape(condition_2_bodycam_tensor)[0]
    number_of_timepoints = np.shape(condition_1_bodycam_tensor)[1]

    # Create Labels
    condition_1_labels = np.zeros(number_of_condition_1_trials)
    condition_2_labels = np.ones(number_of_condition_2_trials)
    combined_labels = np.concatenate([condition_1_labels, condition_2_labels])
    combined_labels = np.reshape(combined_labels, (number_of_condition_1_trials + number_of_condition_2_trials, 1))
    combined_data = np.vstack([condition_1_bodycam_tensor, condition_2_bodycam_tensor])

    print("Number of condition 1 trials", number_of_condition_1_trials)
    print("Number of condition 2 trials", number_of_condition_2_trials)
    print("Combined labels", np.shape(combined_labels))
    print("Combined Data", np.shape(combined_data))

    # Perform Decoding
    skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)

    mean_score_list = []
    for timepoint in range(number_of_timepoints):
        score_list = []
        timepoint_data = combined_data[:, timepoint]
        for train_index, test_index in skf.split(timepoint_data, combined_labels):

            # Split Data Into Train And Test Sets
            X_train, X_test = timepoint_data[train_index], timepoint_data[test_index]
            y_train, y_test = combined_labels[train_index], combined_labels[test_index]

            print("X Train Shape", np.shape(X_train))
            print("Y Train Shape", np.shape(y_train))
            # Create Model
            model = LogisticRegression()

            # Train Model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluate Model
            score = accuracy_score(y_test, y_pred)
            score_list.append(score)

        mean_score = np.mean(score_list)
        mean_score_list.append(mean_score)

    plt.plot(mean_score_list)
    plt.show()


controls = ["/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NRXN78.1A/2020_12_09_Switching_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NRXN78.1D/2020_11_29_Switching_Imaging"]

mutants =  ["/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK12.1F/2021_09_22_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NRXN71.2A/2020_12_17_Switching_Imaging"]

all_mice = controls + mutants


start_window = -10
stop_window = 40
onsets_list = ["visual_context_stable_vis_2", "odour_context_stable_vis_2"]

for base_directory in all_mice:
    print("Base Diretory: ", base_directory)
    decode_context_using_behaviour(base_directory, onsets_list[0], onsets_list[1], start_window, stop_window)