import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec
import numpy as np
import tables
import os
import pandas as pd
from scipy import ndimage
from skimage.transform import resize
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

import RT_Strat_Utils


def get_average_tensors(tensor_directory, save_directory):

    # Load Mask
    indicies, image_height, image_width = RT_Strat_Utils.load_tight_mask()


    file_list = ['0500.h5',
                 '0600.h5',
                 '0700.h5',
                 '0800.h5',
                 '0900.h5',
                 '1000.h5',
                 '1100.h5',
                 '1200.h5',
                 '1300.h5',
                 '1400.h5',
                 '1500.h5',
                 '1600.h5',
                 '1700.h5',
                 '1800.h5',
                 '1900.h5']


    print("File List", file_list)
    control_response_list = []
    mutant_response_list = []

    for file in tqdm(file_list):
        data_file = tables.open_file(os.path.join(tensor_directory, file), "r")

        # Iterate Through Control Group
        control_responses = []
        for control_file in data_file.list_nodes(where="/Controls"):
            average_response = np.mean(control_file, axis=0)
            control_responses.append(average_response)

        # Iterate Through Mutant Group
        mutant_responses = []
        for mutant_file in data_file.list_nodes(where="/Mutants"):
            average_response = np.mean(mutant_file, axis=0)
            mutant_responses.append(average_response)


        # Get Average Group Responses
        average_control_response = np.mean(control_responses, axis=0)
        average_mutant_response = np.mean(mutant_responses, axis=0)


        control_response_list.append(average_control_response)
        mutant_response_list.append(average_mutant_response)
        data_file.close()

    # View These
    # Create Figure
    number_of_bins = len(control_response_list)
    number_of_timepoints = np.shape(control_response_list[0])[0]
    print("Number of timepoints", number_of_timepoints)
    figure_1 = plt.figure(figsize=(20,60))
    gridspec_1 = GridSpec(nrows=3, ncols=number_of_bins)

    vmin = 0
    vmax = 35000
    diff_cmap = RT_Strat_Utils.get_mussal_cmap()

    # PLot Each Timepoint
    for timepoint in range(number_of_timepoints):

        for reaction_time_bin in range(number_of_bins):
            control_axis = figure_1.add_subplot(gridspec_1[0, reaction_time_bin])
            mutant_axis = figure_1.add_subplot(gridspec_1[1, reaction_time_bin])
            difference_axis = figure_1.add_subplot(gridspec_1[2, reaction_time_bin])

            control_data = control_response_list[reaction_time_bin][timepoint]
            mutant_data = mutant_response_list[reaction_time_bin][timepoint]
            difference = np.subtract(mutant_data, control_data)

            control_image = RT_Strat_Utils.create_image_from_data(control_data, indicies, image_height, image_width)
            mutant_image = RT_Strat_Utils.create_image_from_data(mutant_data, indicies, image_height, image_width)
            diff_image = RT_Strat_Utils.create_image_from_data(difference, indicies, image_height, image_width)

            control_axis.imshow(control_image, vmax=vmax, vmin=vmin)
            mutant_axis.imshow(mutant_image, vmax=vmax, vmin=vmin)
            difference_axis.imshow(diff_image, vmax=0.5 * vmax, vmin=-0.5 * vmax, cmap=diff_cmap)

            control_axis.axis('off')
            mutant_axis.axis('off')
            difference_axis.axis('off')

        plt.draw()
        plt.pause(0.1)
        plt.savefig(os.path.join(save_directory, str(timepoint).zfill(4) + ".png"))
        plt.clf()


tensor_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Early_Learning/RT_Stratified_Tensors"
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Reaction_Time_Distributions/Early_Learning/RT_Response_Images"
get_average_tensors(tensor_directory, save_directory)

