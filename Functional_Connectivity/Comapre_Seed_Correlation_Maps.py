import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions



def get_seed_correlation_maps(session_list, trial_start, trial_stop):

    number_of_sessions = len(session_list)

    rows = number_of_sessions
    columns = 3
    figure_1 = plt.figure()

    for timepoint in range(trial_start, trial_stop):
        # Get Visual Maps
        count = 1
        for base_directory in session_list:


            # Load Correlation Maps
            visual_correlation_map = np.load(os.path.join(base_directory, "Correlation_Map_Systematic_Search", "RSC_Visual_Correlation_Map_" + str(timepoint).zfill(3) + ".npy"))
            odour_correlation_map = np.load(os.path.join(base_directory, "Correlation_Map_Systematic_Search", "RSC_Odour_Correlation_Map_" + str(timepoint).zfill(3) + ".npy"))
            difference_map = np.subtract(visual_correlation_map, odour_correlation_map)

            # Load Mask
            indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

            # Convert To Images
            visual_correlation_map_image = Widefield_General_Functions.create_image_from_data(visual_correlation_map, indicies, image_height, image_width)
            odour_correlation_map_image = Widefield_General_Functions.create_image_from_data(odour_correlation_map,   indicies, image_height, image_width)
            difference_map_image = Widefield_General_Functions.create_image_from_data(difference_map,                 indicies, image_height, image_width)

            # Create Axes
            visual_context_axis = figure_1.add_subplot(rows, columns, count)
            odour_context_axis = figure_1.add_subplot(rows, columns, count + 1)
            difference_axis = figure_1.add_subplot(rows, columns, count + 2)

            visual_context_axis.imshow(visual_correlation_map_image, cmap='bwr', vmin=-1, vmax=1)
            odour_context_axis.imshow(odour_correlation_map_image, cmap='bwr', vmin=-1, vmax=1)
            difference_axis.imshow(difference_map_image, cmap='bwr', vmin=-1, vmax=1)

            count += 3


        figure_1.suptitle(str(timepoint))
        plt.draw()
        plt.pause(0.1)
        plt.clf()




controls = [
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_02_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_08_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_23_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_31_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_15_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_10_29_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_11_05_Transition_Imaging"]

trial_start = -10
trial_stop = 50

get_seed_correlation_maps(controls, trial_start, trial_stop)