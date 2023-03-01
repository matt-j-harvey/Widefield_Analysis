import numpy as np
from tqdm import tqdm
from Files import Session_List
from Widefield_Utils import widefield_utils, Create_Activity_Tensor
import Create_Movement_Regressor_Tensor

# Load Session List
session_list = Session_List.control_switching_sessions

# Load Analysis Details
analysis_name = "MVAR_Analysis"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)

# Generate Activity Tensors
tensor_save_directory = r"/media/matthew/External_Harddrive_2/Angus_Collab/Activity_Tensors"
print("Session List", session_list)
for base_directory in tqdm(session_list):
    for condition in onset_files:

        #Create_Activity_Tensor.create_activity_tensor(base_directory, condition, start_window, stop_window, tensor_save_directory, gaussian_filter=True, start_cutoff=3000, align_within_mice=True, align_across_mice=True, baseline_correct=False)
        Create_Movement_Regressor_Tensor.create_movement_tensor(base_directory, condition, start_window, stop_window, tensor_save_directory)







