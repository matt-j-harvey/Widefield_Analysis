import os
import numpy as np
import matplotlib.pyplot as plt


from Widefield_Utils import widefield_utils


def check_widefield_frame_times(base_directory):


    # Load AI Data
    ai_data = widefield_utils.load_ai_recorder_file(base_directory)

    # Create Stimuli Dict
    stimuli_dict = widefield_utils.create_stimuli_dictionary()

    # Get Blue LED Trace
    widefield_trace = ai_data[stimuli_dict["LED 1"]]

    # Load Widefield Frame Times
    widefield_frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    widefield_frame_times = list(widefield_frame_times.keys())

    print(widefield_frame_times)
    plt.plot(widefield_trace)
    plt.scatter(widefield_frame_times, np.ones(len(widefield_frame_times)))
    plt.show()



def check_mousecam_frame_times(base_directory):


    # Load AI Data
    ai_data = widefield_utils.load_ai_recorder_file(base_directory)

    # Create Stimuli Dict
    stimuli_dict = widefield_utils.create_stimuli_dictionary()

    # Get Blue LED Trace
    widefield_trace = ai_data[stimuli_dict["Mousecam"]]

    # Load Widefield Frame Times
    widefield_frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Mousecam_Frame_Times.npy"), allow_pickle=True)[()]
    widefield_frame_times = list(widefield_frame_times.keys())

    print(widefield_frame_times)
    plt.plot(widefield_trace)
    plt.scatter(widefield_frame_times, np.ones(len(widefield_frame_times)))
    plt.show()





def view_mousecam_matching(base_directory):

    # Load AI Data
    ai_data = widefield_utils.load_ai_recorder_file(base_directory)

    # Create Stimuli Dict
    stimuli_dict = widefield_utils.create_stimuli_dictionary()

    # get Mousecam Trace
    mousecam_trace = ai_data[stimuli_dict["Mousecam"]]

    # Get Blue LED Trace
    widefield_trace = ai_data[stimuli_dict["LED 1"]]

    # Load Widefield Frame Times
    widefield_frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    widefield_frame_times = widefield_utils.invert_dictionary(widefield_frame_times)
    widefield_frame_times_key_list = list(widefield_frame_times.keys())
    print("widefield_frame_times_key_list", widefield_frame_times_key_list[0:5])

    # Load Mousecam Frame Dict
    mousecam_frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Mousecam_Frame_Times.npy"), allow_pickle=True)[()]
    mousecam_frame_times = widefield_utils.invert_dictionary(mousecam_frame_times)
    mousecam_frame_times_key_list = list(mousecam_frame_times.keys())[0:5]
    print("mousecam_frame_times_key_list", mousecam_frame_times_key_list)

    # Load Mousecam To Widefield Frame Dict
    widefield_to_mousecam_frame_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy"), allow_pickle=True)[()]

    print("Widefield To Mousecam Frame Keys", list(widefield_to_mousecam_frame_dict.keys())[0:5])
    print("Widefield To Mousecam Frame Values", list(widefield_to_mousecam_frame_dict.values())[0:5])

    matching_dict_list = list(widefield_to_mousecam_frame_dict.keys())


    number_of_widefield_frames = len(widefield_frame_times_key_list)
    for frame_index in range(number_of_widefield_frames):

        widefield_frame_time = widefield_frame_times[frame_index]
        matching_mousecam_frame = widefield_to_mousecam_frame_dict[frame_index]
        mousecam_frame_time = mousecam_frame_times[matching_mousecam_frame]

        print("Frame Index", frame_index, "Widefield Frame Time", widefield_frame_time, "Mousecam Frame Time", mousecam_frame_time)

        plt.plot([widefield_frame_time, mousecam_frame_time], [2,2])


    plt.plot(widefield_trace, c='b', alpha=0.5)
    plt.plot(mousecam_trace, c='m', alpha=0.5)
    plt.show()


base_directory = "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging"
#check_widefield_frame_times(base_directory)
#check_mousecam_frame_times(base_directory)
view_mousecam_matching(base_directory)