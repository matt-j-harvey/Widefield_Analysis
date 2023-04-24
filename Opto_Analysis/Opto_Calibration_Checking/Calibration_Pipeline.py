import matplotlib.pyplot as plt
import os

from Preprocessing import Downsample_AI_Framewise
from Behaviour_Analysis import Create_Behaviour_Matrix_Opto

import Check_Widefield_Frame_Numbers
import Unpack_raw_calib_data
import Count_Calibration_Opto_Onsets
import Check_Opto_Classification
import Check_Projector_Delay


def plot_report(base_directory, max_projection_list, delay_list, widefield_match_message, opto_match_message):
    figure_1 = plt.figure()

    # Plot Pattern Max Projections
    n_patterns = len(max_projection_list)
    for pattern_index in range(n_patterns):
        axis = figure_1.add_subplot(1,n_patterns + 1, pattern_index + 1)
        axis.imshow(max_projection_list[pattern_index])

    # Plot Delay Jitter
    axis = figure_1.add_subplot(1, n_patterns + 1, n_patterns + 1)
    axis.hist(delay_list)

    figure_1.suptitle(str(widefield_match_message) + "_" + str(opto_match_message))

    plt.savefig(os.path.join(base_directory, "Calibration_Report.png"))
    plt.show()



def check_calibration_session(base_directory):

    # Create Behaviour Matrix
    #Create_Behaviour_Matrix_Opto.create_behaviour_matrix(base_directory)

    # Downsample AI Framewise
    #Downsample_AI_Framewise.downsample_ai_matrix(base_directory)

    # Check Widefield Frame Numbers
    widefield_match_message = Check_Widefield_Frame_Numbers.check_widefield_frame_times(base_directory)
    print("widefield_match_message", widefield_match_message)

    # Unpack Raw Calib Data
    #Unpack_raw_calib_data.unpack_calibration_data(base_directory)

    # Check Opto Onset Numbers
    opto_match_message = Count_Calibration_Opto_Onsets.check_opto_onsets(base_directory)
    print("opto_match_message", opto_match_message)

    # Check Stimuli Classification
    max_projection_list = Check_Opto_Classification.check_stim_classification(base_directory)

    # Check Onset Jitter
    delay_list = Check_Projector_Delay.check_projector_delay(base_directory)

    # Plot Report
    plot_report(base_directory, max_projection_list, delay_list, widefield_match_message, opto_match_message)

base_directory = r"/media/matthew/External_Harddrive_3/Opto_FSM_Calibration/Calibration/Opto_Test_2023_04_11"
check_calibration_session(base_directory)