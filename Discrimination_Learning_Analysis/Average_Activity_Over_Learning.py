import numpy as np
import sklearn.svm
from sklearn.decomposition import NMF
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
import os
import matplotlib.pyplot as plt
import sys
from matplotlib import cm
import tables
import matplotlib.gridspec as gridspec

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")
import Widefield_General_Functions


def get_selected_widefield_frames(onsets, start_window, stop_window):

    selected_fames = []

    for onset in onsets:
        trial_start = onset + start_window
        trial_stop = onset + stop_window
        trial_frames = list(range(trial_start, trial_stop))
        selected_fames.append(trial_frames)

    return selected_fames


def get_selected_data(selected_onsets, data):

    selected_data = []

    for trial in selected_onsets:
        trial_data = []
        for frame in trial:
            frame_data = data[frame]
            trial_data.append(frame_data)

        selected_data.append(trial_data)

    return selected_data



def plot_mean_responses(vis_1_session_list, vis_2_session_list, base_directory, root_directory):

    save_directory = root_directory + "/Average_Comparison"
    Widefield_General_Functions.check_directory(save_directory)

    trial_length = np.shape(vis_1_session_list[0])[0]
    number_of_sessions = len(vis_1_session_list)
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    for timepoint in range(trial_length):
        figure_1 = plt.figure(constrained_layout=True)
        grid_specification = gridspec.GridSpec(ncols=number_of_sessions, nrows=3, figure=figure_1)

        for session_index in range(number_of_sessions):
            vis_1_axis = figure_1.add_subplot(grid_specification[0, session_index])
            vis_2_axis = figure_1.add_subplot(grid_specification[1, session_index])
            diff_axis = figure_1.add_subplot(grid_specification[2, session_index])

            vis_1_image = Widefield_General_Functions.create_image_from_data(vis_1_session_list[session_index][timepoint], indicies, image_height, image_width)
            vis_2_image = Widefield_General_Functions.create_image_from_data(vis_2_session_list[session_index][timepoint], indicies, image_height, image_width)
            diff_image = np.subtract(vis_1_image, vis_2_image)

            vmax = 1#np.max(np.concatenate([vis_1_image, vis_2_image]))
            vmin = 0
            diff_magnitude = 0.2#np.max(np.abs(diff_image))

            vis_1_axis.imshow(vis_1_image, vmin=vmin, vmax=vmax, cmap='jet')
            vis_2_axis.imshow(vis_2_image, vmin=vmin, vmax=vmax, cmap='jet')
            diff_axis.imshow(diff_image, vmin=-1*diff_magnitude, vmax=diff_magnitude, cmap='bwr')

            vis_2_axis.axis('off')
            vis_1_axis.axis('off')
            diff_axis.axis('off')

        plt.suptitle(str(timepoint))
        plt.savefig(save_directory + "/" + str(timepoint).zfill(3) + ".png")
        plt.close()
        #plt.show()





root_directory = "/media/matthew/Seagate Expansion Drive2/Longitudinal_Analysis/NXAK4.1B/"
session_list = ["2021_02_04_Discrimination_Imaging",
                "2021_02_06_Discrimination_Imaging",
                "2021_02_08_Discrimination_Imaging",
                "2021_02_10_Discrimination_Imaging",
                "2021_02_12_Discrimination_Imaging",
                "2021_02_14_Discrimination_Imaging",
                "2021_02_22_Discrimination_Imaging"]

start_window = -10
stop_window = 100

vis_1_session_list = []
vis_2_session_list = []

for session_index in range(len(session_list)):
    print("Session: ", session_index, " of ", len(session_list))

    # Set Base Directory
    base_directory = root_directory + session_list[session_index]

    # Create Output Directory
    output_directory = root_directory + "/Average_Activity_Over_Learning"
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # Load Frame Onsets and Frame Times
    vis_1_onsets_file = base_directory + "/Stimuli_Onsets/All_vis_1_frame_indexes.npy"
    vis_2_onsets_file = base_directory + "/Stimuli_Onsets/All_vis_2_frame_indexes.npy"
    vis_1_onsets = np.load(vis_1_onsets_file)
    vis_2_onsets = np.load(vis_2_onsets_file)

    # Extract Widefield Data
    widefield_file = base_directory + "/Delta_F.h5"
    widefield_data_file = tables.open_file(widefield_file, mode='r')
    widefield_data = widefield_data_file.root['Data']

    # Get All Selected Widefield Frames
    vis_1_widefield_frames = get_selected_widefield_frames(vis_1_onsets, start_window, stop_window)
    vis_2_widefield_frames = get_selected_widefield_frames(vis_2_onsets, start_window, stop_window)

    # Extract These Frames From The Delta_F.h5 File
    #vis_1_widefield_data = get_selected_data(vis_1_widefield_frames, widefield_data)
    #vis_1_widefield_data = np.mean(vis_1_widefield_data, axis=0)

    #vis_2_widefield_data = get_selected_data(vis_2_widefield_frames, widefield_data)
    #vis_2_widefield_data = np.mean(vis_2_widefield_data, axis=0)

    vis_1_widefield_data = np.load(base_directory + "/Stimuli_Evoked_Responses/All Vis 1/All Vis 1_Activity_Matrix_Average.npy")
    vis_2_widefield_data = np.load(base_directory + "/Stimuli_Evoked_Responses/All Vis 2/All Vis 2_Activity_Matrix_Average.npy")

    # Add These To Lists
    vis_1_session_list.append(vis_1_widefield_data)
    vis_2_session_list.append(vis_2_widefield_data)

plot_mean_responses(vis_1_session_list, vis_2_session_list, base_directory, root_directory)