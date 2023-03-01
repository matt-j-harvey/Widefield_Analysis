import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Widefield_Utils import widefield_utils, Create_Activity_Tensor
from Files import Session_List


# Load Analysis Details
### Correct Rejections Post Learning ###
analysis_name = "Unrewarded_Contextual_Modulation"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)
start_window = -14
tensor_save_directory = r"//media/matthew/Expansion/Widefield_Analysis/Variation_Prediction"

# Load Session List
control_switching_sessions = Session_List.control_switching_sessions

# Create Tensors Without Baseline Correction
for base_directory in tqdm(control_switching_sessions):
    print(base_directory)
    Create_Activity_Tensor.create_activity_tensor(base_directory, onset_files[0], start_window, stop_window, tensor_save_directory, start_cutoff=3000, align_within_mice=True, align_across_mice=True, baseline_correct=False)





# Average Activity IN 500ms preceeding Stimulus, Predict Average Activity in 1000ms Following Stimulus Onset

