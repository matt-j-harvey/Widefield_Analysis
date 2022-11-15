import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
import os
import tables
from scipy import signal, ndimage, stats
from sklearn.neighbors import KernelDensity
import cv2
from matplotlib import gridspec, cm
from matplotlib.colors import Normalize

import Retinotopy_Utils


def create_activity_video(base_directory, stimulus_name):

    # Get Stimuli Folder
    data_folder = os.path.join(base_directory, "Stimuli_Evoked_Responses", stimulus_name)

    # Load Mask
    indicies, image_height, image_width = Retinotopy_Utils.load_downsampled_mask(base_directory)

    # Load Activity Matrix
    activity_matrix = np.load(os.path.join(data_folder, stimulus_name + "_Activity_Matrix_Average.npy"))
    activity_matrix = np.nan_to_num(activity_matrix)
    print("activity matrix", np.shape(activity_matrix))

    # Get Window Dimensions
    number_of_frames = np.shape(activity_matrix)[0]

    # Set Colourmap
    colourmap = Retinotopy_Utils.get_musall_cmap()
    image_max = 0.05
    image_min = -0.05
    colourmap = plt.cm.ScalarMappable(norm=Normalize(vmin=image_min, vmax=image_max), cmap=colourmap)

    # Create Video File
    video_name = os.path.join(base_directory, "Stimuli_Evoked_Responses", stimulus_name + "_Average_Response_Video.avi")
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, video_codec, frameSize=(image_width, image_height), fps=30)  # 0, 12

    # Draw Activity
    for frame in range(number_of_frames):

        # Reconstruct Image
        image = Retinotopy_Utils.create_image_from_data(activity_matrix[frame], indicies, image_height, image_width)

        # Apply Colour
        image = colourmap.to_rgba(image)
        image = np.multiply(image, 255)
        image = np.ndarray.astype(image, np.uint8)

        # Write To Video
        image_from_plot = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video.write(image_from_plot)

    cv2.destroyAllWindows()
    video.release()




