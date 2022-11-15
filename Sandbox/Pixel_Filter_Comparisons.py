
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
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import time

import Preprocessing_Utils



def get_lowcut_coefs(w=0.0033, fs=28.):
    b, a = signal.butter(2, w/(fs/2.), btype='highpass');


    return b, a


def perform_lowcut_filter(data, b, a):
    filtered_data = signal.filtfilt(b, a, data, padlen=10000)
    return filtered_data

def lowpass(X, w = 7.5, fs = 30.):
    from scipy.signal import butter, filtfilt
    b, a = butter(2,w/(fs/2.), btype='lowpass');
    return filtfilt(b, a, X, padlen=50)

def examine_pixel_regression(output_directory):

    # Load Mask

    # Load ROI
    roi_indicies = np.load(os.path.join(output_directory, "Selected_ROI.npy"))
    print("Roi Pixels", np.shape(roi_indicies))

    # Load BLue and Violet Files
    blue_df_file = os.path.join(output_directory, "Blue_DF.hdf5")
    blue_df_file_container = h5py.File(blue_df_file, 'r')
    blue_df_dataset = blue_df_file_container["Data"]
    print("Blue Shape", np.shape(blue_df_dataset))

    violet_df_file = os.path.join(output_directory, "violet_DF.hdf5")
    violet_df_file_container = h5py.File(violet_df_file, 'r')
    violet_df_dataset = violet_df_file_container["Data"]
    print("Blue Shape", np.shape(violet_df_dataset))

    print("Loading traces")
    violet_trace = violet_df_dataset[:, roi_indicies]
    blue_trace = blue_df_dataset[:, roi_indicies]
    violet_trace = np.mean(violet_trace, axis=1)
    blue_trace = np.mean(blue_trace, axis=1)

    slope, intercept, r, p, stdev = stats.linregress(violet_trace, blue_trace)
    print("Slope", slope)
    print("Intercept", intercept)

    # Scale Violet Trace
    scaled_violet_trace = np.multiply(violet_trace, slope)
    scaled_violet_trace = np.add(scaled_violet_trace, intercept)
    corrected_trace = np.subtract(blue_trace, scaled_violet_trace)

    plt.plot(violet_trace, c='m', alpha=0.8)
    plt.plot(blue_trace, c='b', alpha=0.8)
    plt.plot(corrected_trace, c='g', alpha=0.8)
    plt.show()

    blue_df_file_container.close()
    violet_df_file_container.close()


base_directory = r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging"
output_directory = r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging/Heamocorrection_Visualisation"
#visualise_heamocorrection_changes(base_directory, output_directory)
#create_comparison_movie(base_directory, output_directory)
#view_violet_movie(base_directory)

examine_pixel_regression(output_directory)