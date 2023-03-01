import numpy as np
from tqdm import tqdm
from Files import Session_List
from Widefield_Utils import widefield_utils, Create_Activity_Tensor



# Load Session List
session_list = Session_List.control_switching_sessions


# Load Analysis Details

### Correct Rejections Post Learning ###
analysis_name = "Unrewarded_Contextual_Modulation"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)
tensor_save_directory = r"//media/matthew/Expansion/Widefield_Analysis/Learning_Analysis"

# 2 Seconds Prior To 1.5 Seconds Post
start_window = -55
stop_window = 42


tensor_save_directory = r"/media/matthew/External_Harddrive_2/Angus_Collaboration/Activity_Tensors"

session_list = session_list[0:3]
print("Session List", session_list)
for base_directory in tqdm(session_list):
    for condition in onset_files:
        Create_Activity_Tensor.create_activity_tensor(base_directory, condition, start_window, stop_window, tensor_save_directory, gaussian_filter=True, start_cutoff=3000, align_within_mice=True, align_across_mice=True, baseline_correct=False)







