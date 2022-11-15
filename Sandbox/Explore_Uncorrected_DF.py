import numpy as np
import matplotlib.pyplot as plt
import h5py
import tables
from scipy import signal, ndimage, stats
import os
import cv2
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA, FactorAnalysis, TruncatedSVD
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import time

import Preprocessing_Utils



def explore_uncorrected_df(base_directory):

    # Get Filenames
    uncorrected_delta_f = os.path.join(base_directory, "Uncorrected_Delta_F.hdf5")
    uncorrected_delta_f = h5py.File(uncorrected_delta_f, mode='r')
    blue_traces = uncorrected_delta_f["Blue_DF"]
    violet_traces = uncorrected_delta_f["Violet_DF"]

    number_of_frames, number_of_pixels = np.shape(blue_traces)

    downsampled_coef_map = []

    for pixel_index in tqdm(range(number_of_pixels)):

        blue_trace = blue_traces[:, pixel_index]
        violet_trace = violet_traces[:, pixel_index]

        slope, intercept, r, p, stdev = stats.linregress(violet_trace, blue_trace)
        print("Slope", slope)

        # Perform Regression
        plt.plot(blue_trace, c='b')
        plt.plot(violet_trace, c='m')
        plt.show()

base_directory = r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging/Downsampled_Raw_Data"
explore_uncorrected_df(base_directory)