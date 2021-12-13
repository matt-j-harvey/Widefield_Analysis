import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn.linear_model import Ridge
import os
import math
import scipy
import tables
from bisect import bisect_left
import cv2
from sklearn.decomposition import TruncatedSVD, FastICA, NMF, SparseCoder, LatentDirichletAllocation
from pathlib import Path
import joblib
from scipy import signal, ndimage, stats
from skimage.transform import resize
from scipy.interpolate import interp1d
import sys
import matplotlib.gridspec as gridspec

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions


def perform_ica(base_directory):

    # Load Delta F Matrix
    delta_f_matrix_filepath = os.path.join(base_directory, "Delta_F.h5")
    delta_f_matrix_container = tables.open_file(delta_f_matrix_filepath, mode='r')
    delta_f_matrix = delta_f_matrix_container.root['Data']

    # Load Mask
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    # Get Delta F Shape
    number_of_frames = np.shape(delta_f_matrix)[0]
    number_of_pixels = np.shape(delta_f_matrix)[1]

    # Get Sample
    sample_size = 1000
    full_indicies = list(range(number_of_frames))
    sample_indicies = np.random.choice(full_indicies, size=sample_size)
    activity_sample = []
    for index in sample_indicies:
        activity_sample.append(delta_f_matrix[index])
    activity_sample = np.array(activity_sample)

    # Perform ICA
    number_of_components = 20
    model = LatentDirichletAllocation(n_components=number_of_components)
    model.fit(activity_sample)
    components = model.components_

    # View Components
    figure_1 = plt.figure()
    for x in range(20):
        component = components[x]
        image = Widefield_General_Functions.create_image_from_data(component, indicies, image_height, image_width)
        image_magnitude = np.max(np.abs(image))
        subplot = figure_1.add_subplot(4, 5, x + 1)
        subplot.imshow(image, cmap='bwr', vmax=image_magnitude, vmin=-1 * image_magnitude)

    plt.show()

    print("Delta F Shape", np.shape(delta_f_matrix))






controls = ["/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NRXN78.1A/2020_12_09_Switching_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NRXN78.1D/2020_11_29_Switching_Imaging"]

mutants = [ "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK12.1F/2021_09_22_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NRXN71.2A/2020_12_17_Switching_Imaging"]



perform_ica(controls[0])



