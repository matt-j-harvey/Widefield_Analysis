import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions



def view_coef_maps(mouse_list):

    context_1_map_list = []
    context_2_map_list = []
    difference_list = []

    for mouse in mouse_list:

        mouse_context_1_map_list = []
        mouse_context_2_map_list = []
        mouse_difference_list = []

        for base_directory in mouse:

            # Load Correlation Maps
            baseline_regression_map = np.load(os.path.join(base_directory, "Linear_Model_Baseline_Coef_V1.npy"))
            visual_correlation_map = np.load(os.path.join(base_directory, "Linear_Model_Context_1_Coef_V1.npy"))
            odour_correlation_map = np.load(os.path.join(base_directory, "Linear_Model_Context_2_Coef_V1.npy"))
            difference_map = np.subtract(visual_correlation_map, odour_correlation_map)

            mouse_context_1_map_list.append(visual_correlation_map)
            mouse_context_2_map_list.append(odour_correlation_map)
            mouse_difference_list.append(difference_map)

        # Get Mean
        mouse_context_1_map_list = np.array(mouse_context_1_map_list)
        mouse_context_2_map_list = np.array(mouse_context_2_map_list)
        mouse_difference_list = np.array(mouse_difference_list)

        mouse_average_context_1 = np.mean(mouse_context_1_map_list, axis=0)
        mouse_average_context_2 = np.mean(mouse_context_2_map_list, axis=0)
        mouse_average_difference = np.mean(mouse_difference_list, axis=0)

        context_1_map_list.append(mouse_average_context_1)
        context_2_map_list.append(mouse_average_context_2)
        difference_list.append(mouse_average_difference)

    number_of_mice = len(mouse_list)
    rows = number_of_mice
    columns = 3
    figure_1 = plt.figure()

    count = 1

    for mouse_index in range(number_of_mice):

        # Load Mask
        indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

        visual_correlation_map = context_1_map_list[mouse_index]
        odour_correlation_map = context_2_map_list[mouse_index]
        difference_map = difference_list[mouse_index]

        # Convert To Images
        visual_correlation_map_image = Widefield_General_Functions.create_image_from_data(visual_correlation_map, indicies, image_height, image_width)
        odour_correlation_map_image = Widefield_General_Functions.create_image_from_data(odour_correlation_map,   indicies, image_height, image_width)
        difference_map_image = Widefield_General_Functions.create_image_from_data(difference_map,                 indicies, image_height, image_width)

        # Create Axes
        visual_context_axis = figure_1.add_subplot(rows, columns, count + 0)
        odour_context_axis  = figure_1.add_subplot(rows, columns, count + 1)
        difference_axis     = figure_1.add_subplot(rows, columns, count + 2)

        # Show Images
        visual_context_axis.imshow(visual_correlation_map_image, cmap='jet')
        odour_context_axis.imshow(odour_correlation_map_image, cmap='jet')
        difference_axis.imshow(difference_map_image, cmap='bwr', vmin=-0.5, vmax=0.5)

        count += 3


    plt.show()

    mean_diff_map = np.mean(difference_list, axis=0)
    difference_map_image = Widefield_General_Functions.create_image_from_data(mean_diff_map, indicies, image_height, image_width)
    difference_magnitude = np.max(np.abs(difference_map_image))
    plt.imshow(difference_map_image, cmap='bwr', vmin=-1 * difference_magnitude, vmax=difference_magnitude)
    plt.show()


controls = [

            ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_02_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_08_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Switching_Analysis/Wildtype/NXAK4.1B/2021_03_04_Switching_Imaging"],

            ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_23_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_31_Transition_Imaging"],

            ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_15_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging"],

            ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_10_29_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_11_05_Transition_Imaging"],

            ["/media/matthew/Seagate Expansion Drive1/Switching_Analysis/Wildtype/NRXN78.1A/2020_12_05_Switching_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Switching_Analysis/Wildtype/NRXN78.1A/2020_12_09_Switching_Imaging"],

]

# "/media/matthew/Seagate Expansion Drive1/Switching_Analysis/Wildtype/NXAK7.1B/2021_03_02_Switching_Imaging",
# "/media/matthew/Seagate Expansion Drive1/Switching_Analysis/Wildtype/NXAK14.1A/2021_06_09_Switching_Imaging",


view_coef_maps(controls)
#view_coef_maps(mutants)