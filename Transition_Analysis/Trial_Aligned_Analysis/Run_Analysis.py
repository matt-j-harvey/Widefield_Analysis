import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tables

import Trial_Aligned_Utils
import Create_Extended_Tensors
import Create_Behaviour_Tensor
import Create_Video_From_Tensor


def visualise_tensor(mean_tensor, indicies, image_height, image_width):

    blue_black_cmap = Trial_Aligned_Utils.get_musall_cmap()
    plt.ion()
    for frame in mean_tensor:
        frame = Trial_Aligned_Utils.create_image_from_data(frame, indicies, image_height, image_width)
        plt.imshow(frame, cmap=blue_black_cmap, vmin=-0.05, vmax=0.05)
        plt.draw()
        plt.pause(0.1)
        plt.clf()



def get_behaviour_dictionary_list(session_list, number_of_conditions, selected_behaviour_traces, onset_file_list, start_window, stop_window):

    """
    List Of Behaviour Dictionaries For Each Condition
    Each Dictionary Contains An Entry For A Selected AI Trace
    This Entry Is A List Of The Mean Behaviour Trace For Each Session
    """
    print("Selected Behaviour Traces", selected_behaviour_traces)
    behaviour_dict_list = []
    for condition_index in range(number_of_conditions):

        # Create Dictionary To Hold List Of Mean Traces
        mean_behaviour_trace_dict = {}
        for trace in selected_behaviour_traces:
            mean_behaviour_trace_dict[trace] = []

        # Get Mean For Each Session
        onsets_file = onset_file_list[condition_index]
        for base_directory in session_list:
            behaviour_tensor_dict = Create_Behaviour_Tensor.create_behaviour_tensor(base_directory, onsets_file, start_window, stop_window, selected_behaviour_traces)

            # Get Mean
            for trace in selected_behaviour_traces:
                print("Trace", trace)
                mean_behaviour_trace_dict[trace].append(np.mean(behaviour_tensor_dict[trace], axis=0))

        behaviour_dict_list.append(mean_behaviour_trace_dict)

    return behaviour_dict_list


def pad_ragged_tensor_with_nans(ragged_tensor):

    # Get Longest Trial
    length_list = []
    for trial in ragged_tensor:
        trial_length, number_of_pixels = np.shape(trial)
        length_list.append(trial_length)

    max_length = np.max(length_list)

    # Create Padded Tensor
    number_of_trials = len(length_list)
    padded_tensor = np.empty((number_of_trials, max_length, number_of_pixels))
    padded_tensor[:] = np.nan

    # Fill Padded Tensor
    for trial_index in range(number_of_trials):
        trial_data = ragged_tensor[trial_index]
        trial_length = np.shape(trial_data)[0]
        padded_tensor[trial_index, 0:trial_length] = trial_data

    return padded_tensor

def get_activity_tensor_list(session_list, tensor_names, tensor_save_directory):


    number_of_conditions = len(tensor_names)

    activity_tensor_list = []
    for condition_index in range(number_of_conditions):

        condition_name = tensor_names[condition_index]
        condition_name = condition_name.replace('_onsets.npy','')
        print("condition name", condition_name)
        condition_tensor_list = []

        for base_directory in session_list:
            print("Session: ", base_directory)

            # Get Path Details
            mouse_name, session_name = Trial_Aligned_Utils.get_mouse_name_and_session_name(base_directory)

            # Load Activity Tensor
            activity_tensor = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, condition_name + "_Extended_Activity_Tensor.npy"), allow_pickle=True)

            # Pad With NaNs
            activity_tensor = pad_ragged_tensor_with_nans(activity_tensor)
            print("Activity Tensor Shape", np.shape(activity_tensor))

            # Get Average
            mean_activity = np.nanmean(activity_tensor, axis=0)

            # Reconstruct and Align
            #mean_activity = Trial_Aligned_Utils.align_activity_tensor(base_directory, mean_activity)

            # Add To List
            condition_tensor_list.append(mean_activity)

        # Get Group Mean
        condition_tensor_list = np.array(condition_tensor_list)
        condition_mean_tensor = np.mean(condition_tensor_list, axis=0)
        activity_tensor_list.append(condition_mean_tensor)

    return activity_tensor_list



def create_activity_tensors(session_list, onset_file_list, tensor_name_list, start_window, stop_stimuli_list, selected_behaviour_traces, tensor_save_directory):

    number_of_conditions = len(tensor_name_list)
    for base_directory in session_list:
        for condition_index in range(number_of_conditions):
            condition_name = tensor_name_list[condition_index]
            condition_onset_file = onset_file_list[condition_index]
            stop_stimuli = stop_stimuli_list[condition_index]

            #Create_Activity_Tensor.create_activity_tensor(base_directory, condition_onsets, start_window, stop_window, condition_name, spatial_smoothing=False, save_tensor=True)
            Create_Extended_Tensors.create_extended_standard_alignment_tensor(base_directory, tensor_save_directory, condition_onset_file, start_window, stop_stimuli, selected_behaviour_traces, min_trial_number=1, max_trial_num=500, movement_correction=True)


def run_analysis_workflow(session_list, onset_file_list, tensor_names, start_window, stop_window, save_directory, selected_behaviour_traces, stop_stimuli_list, tensor_save_directory):

    # Check Save Directory
    Trial_Aligned_Utils.check_directory(save_directory)

    # Get Number Of Conditions
    number_of_conditions = len(tensor_names)

    # Create Activity Tensors
    #print("Creating Activity Tensors")
    #create_activity_tensors(session_list, onset_file_list, tensor_names, start_window, stop_stimuli_list, selected_behaviour_traces, tensor_save_directory)

    # Get Behavioural Dictionary For Each Condition
    #print("Creating Behavioural Tensors")
    #behaviour_dict_list = get_behaviour_dictionary_list(session_list, number_of_conditions, selected_behaviour_traces, onset_file_list, start_window, stop_window)

    # Load Activity Tensors
    print("Loading Activity Tensors")
    activity_tensor_list = get_activity_tensor_list(session_list, onset_file_list, tensor_save_directory)
    #for tensor in activity_tensor_list:
    # print("Tensor Shape", np.shape(tensor))

    # Create Activity Video
    indicies, image_height, image_width = Trial_Aligned_Utils.load_tight_mask()

    #visualise_tensor(activity_tensor_list[0], indicies, image_height, image_width)
    Create_Video_From_Tensor.create_activity_video(activity_tensor_list, start_window, tensor_names, save_directory, indicies, image_height, image_width, timestep=36)
    #Create_Video_From_Tensor.create_activity_video(indicies, image_height, image_width, activity_tensor_list, start_window, stop_window, plot_titles, save_directory, behaviour_dict_list, selected_behaviour_traces, difference_conditions=difference_conditions)


# Get Analysis Details
analysis_name = "Absence Of Expected Odour"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Trial_Aligned_Utils.load_analysis_container(analysis_name)
stop_stimuli_list = [["Odour 1", "Visual 1", "Visual 2"], ["Odour 1", "Odour 2", "Visual 1", "Visual 2"], ["Odour 1", "Odour 2", "Visual 1", "Visual 2"]]
tensor_save_directory = r"/media/matthew/29D46574463D2856/Nature_Transition_Analysis_Results/Extended_Tensors"

session_list = [

    r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_23_Transition_Imaging",
    r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_31_Transition_Imaging",
    r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_04_02_Transition_Imaging",

    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_02_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_08_Transition_Imaging",
    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_10_Transition_Imaging",

    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_13_Transition_Imaging",
    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_15_Transition_Imaging",
    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_17_Transition_Imaging",

    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_10_29_Transition_Imaging",
    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_11_03_Transition_Imaging",
    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_11_05_Transition_Imaging"

]

# Run Analysis
save_directory = r"/media/matthew/29D46574463D2856/Nature_Transition_Analysis_Results/Absence_of_expected_Odour"
#stop_window = 150
print("Tensor Names", tensor_names)
print("Onset Files", onset_files)
run_analysis_workflow(session_list, onset_files, tensor_names, start_window, stop_window, save_directory, behaviour_traces, stop_stimuli_list, tensor_save_directory)