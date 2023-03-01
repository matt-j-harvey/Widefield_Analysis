import os
import h5py
from tqdm import tqdm
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import tables
from datetime import datetime

from Widefield_Utils import widefield_utils




def view_raw_difference(tensor_directory, analysis_name, vmin=-0.05, vmax=0.05):

    # Open Analysis Dataframe
    analysis_file = tables.open_file(os.path.join(tensor_directory, analysis_name + "_Trialwise_.h5"), mode="r")
    activity_dataset = analysis_file.root["Data"]
    metadata_dataset = analysis_file.root["Trial_Details"]

    number_of_trials, number_of_timepoints, number_of_pixels = np.shape(activity_dataset)

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further( indicies, image_height, image_width)

    print("metadata_dataset", np.shape(metadata_dataset))
    print("activity_dataset", np.shape(activity_dataset))

    # Load AS Array
    print("Starting opening", datetime.now())
    activity_dataset = np.array(activity_dataset)
    print("Finished opening", datetime.now())

    # Split By Condition
    condition_details = metadata_dataset[:, 3]
    condition_1_indicies = np.where(condition_details == 0)[0]
    condition_2_indicies = np.where(condition_details == 1)[0]

    condition_1_data = activity_dataset[condition_1_indicies]
    condition_2_data = activity_dataset[condition_2_indicies]
    print("Condition 1 data", np.shape(condition_1_data))
    print("condition 2 data", np.shape(condition_2_data))

    # Get MEans
    condition_1_data = np.mean(condition_1_data, axis=0)
    condition_2_data = np.mean(condition_2_data, axis=0)

    # Load Colourmap
    colourmap = widefield_utils.get_musall_cmap()

    for timepoint_index in tqdm(range(number_of_timepoints), position=0, desc="Timepoint"):
        figure_1 = plt.figure()

        condition_1_axis = figure_1.add_subplot(1,3,1)
        condition_2_axis = figure_1.add_subplot(1, 3, 2)
        diff_axis = figure_1.add_subplot(1, 3, 3)

        # Recreate Images
        condition_1_image = widefield_utils.create_image_from_data(condition_1_data[timepoint_index],  indicies, image_height, image_width)
        condition_2_image = widefield_utils.create_image_from_data(condition_2_data[timepoint_index], indicies, image_height, image_width)

        # Plot These
        condition_1_axis.imshow(condition_1_image, cmap=colourmap, vmin=vmin, vmax=vmax)
        condition_2_axis.imshow(condition_2_image, cmap=colourmap, vmin=vmin, vmax=vmax)
        diff_axis.imshow(np.subtract(condition_1_image, condition_2_image), cmap=colourmap, vmin=-0.02, vmax=0.02)


        plt.title(str(timepoint_index))
        plt.show()
