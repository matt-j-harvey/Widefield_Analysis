import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import Normalize
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
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pickle

import Regression_Utils


def load_downsampled_mask(base_directory):

    mask = np.load(os.path.join(base_directory, "Generous_Mask.npy"))

    # Transform Mask
    mask = resize(mask, (300, 304), preserve_range=True, order=0, anti_aliasing=True)

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask > 0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width


def load_smallest_mask(base_directory):

    indicies, image_height, image_width = load_downsampled_mask(base_directory)
    template = np.zeros(image_height * image_width)
    template[indicies] = 1
    template = np.reshape(template, (image_height, image_width))
    template = template[0:300, 0:300]
    template = resize(template, (100,100),preserve_range=True, order=0, anti_aliasing=True)
    template = np.reshape(template, 100 * 100)
    downsampled_indicies = np.nonzero(template)
    return downsampled_indicies, 100, 100



def denoise_data(data):
    model = PCA(n_components=250)
    transformed_data = model.fit_transform(data)
    data = model.inverse_transform(transformed_data)
    return data


def visualise_downsampled_data(base_directory, early_cutoff=3000):

    # Open Full Activity Matrix
    full_delta_f_file = os.path.join(base_directory, "Downsampled_Delta_F.h5")
    full_delta_f_container = tables.open_file(full_delta_f_file, "r")
    full_delta_f = full_delta_f_container.root["Data"]

    # Load Activity Matrix
    delta_f_matrix = np.load(os.path.join(base_directory, "100_By_100_Data.npy"))
    number_of_frames, number_of_pixels = np.shape(delta_f_matrix)

    # Get Sample
    video_length = (number_of_frames - early_cutoff)
    sample_size = int(video_length / 2)
    train_stop = early_cutoff + sample_size
    test_start = train_stop
    test_stop = test_start + sample_size

    # Denoise Preidicted Data
    test_y = delta_f_matrix[test_start:test_stop]
    test_y = denoise_data(test_y)


    # Load Mask
    full_indicies, full_height, full_width = load_downsampled_mask(base_directory)
    indicies, image_height, image_width = load_smallest_mask(base_directory)

    widefield_colourmap = Regression_Utils.get_musall_cmap()
    widefield_colourmap = plt.cm.ScalarMappable(norm=Normalize(vmin=-0.05, vmax=0.05), cmap=widefield_colourmap)

    # Create Video File
    video_name = os.path.join(base_directory, "downsampled_sample_video.avi")
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, video_codec, frameSize=(500, 500), fps=30)  # 0, 12

    figure_1 = plt.figure(figsize=(5, 5))
    canvas = FigureCanvasAgg(figure_1)

    plt.ion()
    number_of_timepoints = np.shape(test_y)[0]
    window_size = 3
    for timepoint_index in tqdm(range(window_size, number_of_timepoints)):
        real_timepoint = test_start + timepoint_index
        
        full_frame = np.mean(full_delta_f[real_timepoint-window_size:real_timepoint], axis=0)
        downsampled_frame = np.mean(test_y[timepoint_index-window_size:timepoint_index], axis=0)

        full_axis = figure_1.add_subplot(1, 2, 1)
        downsampled_axis = figure_1.add_subplot(1, 2, 2)

        downsampled_image = Regression_Utils.create_image_from_data(downsampled_frame, indicies, image_height, image_width)
        full_image = Regression_Utils.create_image_from_data(full_frame, full_indicies, full_height, full_width)

        downsampled_image = widefield_colourmap.to_rgba(downsampled_image)
        full_image = widefield_colourmap.to_rgba(full_image)

        full_axis.imshow(full_image)
        downsampled_axis.imshow(downsampled_image)

        full_axis.axis('off')
        downsampled_axis.axis('off')

        figure_1.canvas.draw()

        # Write To Video
        buf = canvas.buffer_rgba()
        image_from_plot = np.asarray(buf)
        image_from_plot = cv2.cvtColor(image_from_plot, cv2.COLOR_RGB2BGR)
        video.write(image_from_plot)

        plt.clf()

    cv2.destroyAllWindows()
    video.release()


base_directory = r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging"
visualise_downsampled_data(base_directory)