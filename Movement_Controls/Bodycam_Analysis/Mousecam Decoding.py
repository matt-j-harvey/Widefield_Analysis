import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Analysis/Movement_Controls/Bodycam_Analysis")
import Bodycam_SVD




start_window = -10
stop_window = 40
onset_files = [["visual_context_stable_vis_2_frame_onsets.npy"], ["odour_context_stable_vis_2_frame_onsets.npy"]]
tensor_names = ["Vis_2_Stable_Visual", "Vis_2_Stable_Odour"]

# Load Bodycam Tensor
condition_1_tensor = Bodycam_SVD.get_bodycam_tensor(base_directory, video_file, onset_files[0], start_window, stop_window, number_of_components=20)
condition_2_tensor = Bodycam_SVD.get_bodycam_tensor(base_directory, video_file, onset_files[1], start_window, stop_window, number_of_components=20)

# Get Data Structure
number_of_condition_1_trials = np.shape(condition_1_tensor)[0]
number_of_condition_2_trials = np.shape(condition_1_tensor)[0]
number_of_timepoints = stop_window - start_window

# Create Labels
condition_1_labels = np.zeros(number_of_condition_1_trials)
condition_2_labels = np.ones(number_of_condition_2_trials)

# Combine Labels and Tensors
combined_tensor = np.vstack([condition_1_tensor, condition_2_tensor])
combined_labels = np.vstack([condition_1_labels, condition_2_labels])

# Create Train Test Split
skf = StratifiedKFold(n_splits=5, Shuffle=True)
for train_index, test_index in skf.split(combined_tensor, combined_labels):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = LogisticRegression()
    model.fit()