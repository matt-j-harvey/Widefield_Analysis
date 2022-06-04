import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions



def view_coef_maps(session_list):

    number_of_sessions = len(session_list)

    rows = number_of_sessions
    columns = 3
    figure_1 = plt.figure()

    different_map_list = []

    count = 1
    for base_directory in session_list:

        # Load Correlation Maps
        #visual_correlation_map = np.load(os.path.join(base_directory, "V1_Visual_Correlation_Map.npy"))
        #odour_correlation_map = np.load(os.path.join(base_directory, "V1_Odour_Correlation_Map.npy"))

        visual_correlation_map = np.load(os.path.join(base_directory, "Linear_Model_Context_1_Coef_V1.npy"))
        odour_correlation_map = np.load(os.path.join(base_directory, "Linear_Model_Context_2_Coef_V1.npy"))

        difference_map = np.subtract(visual_correlation_map, odour_correlation_map)
        different_map_list.append(difference_map)

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

        visual_context_axis.imshow(visual_correlation_map_image, cmap='jet', vmin=0)
        odour_context_axis.imshow(odour_correlation_map_image, cmap='jet', vmin=0)
        difference_axis.imshow(difference_map_image, cmap='bwr', vmin=-0.5, vmax=0.5)

        count += 3


    plt.show()
    #plt.draw()
    #plt.pause(0.1)
    #plt.clf()

    mean_diff_map = np.mean(different_map_list, axis=0)
    difference_map_image = Widefield_General_Functions.create_image_from_data(mean_diff_map, indicies, image_height, image_width)
    difference_magnitude = np.max(np.abs(difference_map_image))
    plt.imshow(difference_map_image, cmap='bwr', vmin=-1 * difference_magnitude, vmax=difference_magnitude)
    plt.show()


controls = [
            "/media/matthew/Seagate Expansion Drive1/Switching_Analysis/Wildtype/NRXN78.1A/2020_12_05_Switching_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Switching_Analysis/Wildtype/NRXN78.1D/2020_11_29_Switching",

            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_02_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_08_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_23_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_31_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_15_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_10_29_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_11_05_Transition_Imaging"]


mutants = [
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_10_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK24.1C/2021_11_10_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive1/Transition_2/NXAK20.1B/2021_11_22_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Transition_2/NXAK20.1B/2021_11_24_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Transition_2/NXAK20.1B/2021_11_26_Transition_Imaging",
]

view_coef_maps(controls)
view_coef_maps(mutants)