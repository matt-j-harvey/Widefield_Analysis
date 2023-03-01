import numpy as np
import matplotlib.pyplot as plt
import os

from Widefield_Utils import widefield_utils
from Behaviour_Analysis import Behaviour_Utils



def check_running_traces(base_directory):

    # Load Ai Data
    ai_data = widefield_utils.load_ai_recorder_file(base_directory)
    stimulus_dictionary = widefield_utils.create_stimuli_dictionary()
    running_trace = ai_data[stimulus_dictionary["Running"]]

    # Get Running Differences
    running_derivatives = np.diff(running_trace)

    # Plot Histogram
    plt.hist(running_derivatives)
    plt.show()

    plt.plot(running_trace)
    plt.show()


control_switching_only_sessions_nested = [

    [r"NRXN78.1A/2020_11_28_Switching_Imaging",
     r"NRXN78.1A/2020_12_05_Switching_Imaging",
     r"NRXN78.1A/2020_12_09_Switching_Imaging"],

    [r"NRXN78.1D/2020_11_29_Switching_Imaging",
     r"NRXN78.1D/2020_12_05_Switching_Imaging",
     r"NRXN78.1D/2020_12_07_Switching_Imaging"],

    [r"NXAK14.1A/2021_05_21_Switching_Imaging",
     r"NXAK14.1A/2021_05_23_Switching_Imaging",
     r"NXAK14.1A/2021_06_11_Switching_Imaging"],

    [r"NXAK22.1A/2021_10_14_Switching_Imaging",
     r"NXAK22.1A/2021_10_20_Switching_Imaging",
     r"NXAK22.1A/2021_10_22_Switching_Imaging"],

    [r"NXAK4.1B/2021_03_02_Switching_Imaging",
     r"NXAK4.1B/2021_03_04_Switching_Imaging",
     r"NXAK4.1B/2021_03_06_Switching_Imaging"],

    [r"NXAK7.1B/2021_02_26_Switching_Imaging",
     r"NXAK7.1B/2021_02_28_Switching_Imaging",
     # r"NXAK7.1B/2021_03_02_Switching_Imaging"
     ],

]

for mouse in control_switching_only_sessions_nested:
    for session in mouse:
        check_running_traces(os.path.join("/media/matthew/Expansion/Control_Data", session))

