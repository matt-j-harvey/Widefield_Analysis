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


def get_video_name(base_directory):

    file_list = os.listdir(base_directory)
    for file in file_list:
        if "_cam_1" in file:
            return file


def load_video_chunk(cap, chunk_start, chunk_size):

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 2)

    # Create File To Save This
    for frame_index in range(frameCount):

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





def compute_body_motion_svd(base_directory):


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

    # Create Model
    model = IncrementalPCA(n_components=100)

    # Get Chunk Settings
    preferred_chunk_size = 5000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = Regression_Utils.get_chunk_structure(preferred_chunk_size, frameCount)

    # Fit Model
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