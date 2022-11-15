import numpy as np
import matplotlib.pyplot as plt
import tables
import os
from bisect import bisect_left
import cv2
from tqdm import tqdm
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.decomposition import IncrementalPCA, TruncatedSVD, FastICA, NMF, FactorAnalysis
from skimage.transform import resize
import pickle
from matplotlib import cm
from datetime import datetime

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph

pyqtgraph.setConfigOptions(imageAxisOrder='row-major')

cv2.setNumThreads(10)

import Regression_Utils


def get_video_name(base_directory):

    file_list = os.listdir(base_directory)
    for file in file_list:
        if "_cam_1" in file:
            return file


def compute_body_motion_svd(base_directory):

    # Load Bodycam Motion Energy
    bodycam_motion_energy_file = os.path.join(base_directory, "Mousecam_Analysis",  "Bodycam_Motion_Energy.h5")
    bodycam_motion_container = tables.open_file(bodycam_motion_energy_file, "r")
    bodycam_motion_data = bodycam_motion_container.root["blue"]
    number_of_frames, number_of_pixels = np.shape(bodycam_motion_data)
    print("Bodycam Motion Data", np.shape(bodycam_motion_data))


    preferred_chunk_size = 5000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = Regression_Utils.get_chunk_structure(preferred_chunk_size, number_of_frames)

    # Fit Data
    print("Fitting Model")
    model = IncrementalPCA(n_components=100)
    for chunk_index in tqdm(range(number_of_chunks)):
        chunk_start = chunk_starts[chunk_index]
        chunk_stop = chunk_stops[chunk_index]
        chunk_data = bodycam_motion_data[chunk_start:chunk_stop]
        model.fit(chunk_data)

    # Save Model
    model_file = os.path.join(base_directory, "Mousecam_Analysis",  "SVD Model.sav")
    pickle.dump(model, open(model_file, 'wb'))

    # Transform Data
    print("Transforming Data")
    transformed_data = []
    for chunk_index in tqdm(range(number_of_chunks)):
        chunk_start = chunk_starts[chunk_index]
        chunk_stop = chunk_stops[chunk_index]
        chunk_data = bodycam_motion_data[chunk_start:chunk_stop]
        transformed_chunk = model.transform(chunk_data)
        transformed_data.append(transformed_chunk)

    transformed_data = np.vstack(transformed_data)
    transformed_data = np.array(transformed_data)
    np.save(os.path.join(base_directory, "Mousecam_Analysis", "Transformed_Mousecam_Data.npy"), transformed_data)



def get_bodycam_motion_energy(base_directory):

    # Get Video Name
    video_name = get_video_name(base_directory)

    # Open Video File
    video_file = os.path.join(base_directory, video_name)
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Check Save Directory
    save_directory = os.path.join(base_directory, "Mousecam_Analysis")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Create Tables File
    bodycam_motion_energy_file = os.path.join(save_directory, "Bodycam_Motion_Energy.h5")
    bodycam_motion_energy_container = tables.open_file(bodycam_motion_energy_file, "w")
    bodycam_motion_array = bodycam_motion_energy_container.create_earray(bodycam_motion_energy_container.root, 'blue', tables.UInt16Atom(), shape=(0, 480 * 640),  expectedrows=frameCount-1)

    # Create File To Save This
    for frame_index in tqdm(range(1, frameCount)):

        # Set Current Frame Data To Preceeding
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index-2)
        ret, preceeding_frame_data = cap.read()

        # Read Current Frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index-1)
        ret, current_frame_data = cap.read()

        # Get Motion Energy
        current_frame_data = current_frame_data[:, :, 0]
        preceeding_frame_data = preceeding_frame_data[:, :, 0]

        current_frame_data = np.ndarray.astype(current_frame_data, int)
        preceeding_frame_data = np.ndarray.astype(preceeding_frame_data, int)

        motion_energy = np.subtract(current_frame_data, preceeding_frame_data)
        motion_energy = np.abs(motion_energy)
        motion_energy = np.ndarray.astype(motion_energy, int)

        # Flatten
        motion_energy = np.reshape(motion_energy, 480 * 640)

        # Write To Tables File
        bodycam_motion_array.append([motion_energy])

        # Flush eArray every 10 Frames
        if frame_index % 10 == 0:
            bodycam_motion_array.flush()

    bodycam_motion_energy_container.close()



session_list = [
    r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_23_Transition_Imaging",
    r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_31_Transition_Imaging",
    r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_04_02_Transition_Imaging",
]


for base_directory in session_list:
    get_bodycam_motion_energy(base_directory)
    compute_body_motion_svd(base_directory)