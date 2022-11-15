import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os
import tables
from bisect import bisect_left
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from skimage.transform import resize
from sklearn.metrics import r2_score
import cv2
from tqdm import tqdm

import Regression_Utils
import Match_Mousecam_Frames_To_Widefield_Frames


def downsample_data(base_directory, data):

    downsampled_data = []
    alignment_dictionary = np.load(os.path.join(base_directory, "Cluster_Alignment_Dictionary.npy"), allow_pickle=True)[()]

    indicies, image_height, image_width = Regression_Utils.load_generous_mask(base_directory)
    downsample_indicies, downsample_height, downsample_width = Regression_Utils.load_tight_mask_downsized()

    for frame in data:

        # Recreate Image
        frame = Regression_Utils.create_image_from_data(frame, indicies, image_height, image_width)

        # Align To Common Framework
        frame = Regression_Utils.transform_image(frame, alignment_dictionary)

        # Downsample
        frame = resize(frame, output_shape=(100, 100),preserve_range=True)

        frame = np.reshape(frame, 100 * 100)

        frame = frame[downsample_indicies]

        downsampled_data.append(frame)

    downsampled_data = np.array(downsampled_data)

    return downsampled_data



def incremental_regression(train_x, train_y, test_x, test_y):

    number_of_pixels = np.shape(train_y)[1]

    r2_list = []

    for pixel_index in tqdm(range(number_of_pixels)):

        # Create Model
        model = Ridge()

        # Fit Model
        model.fit(train_x, train_y[:, pixel_index])

        # Make Prediciton
        y_pred = model.predict(test_x)

        # Score Model
        model_score = r2_score(y_true=test_y[:, pixel_index], y_pred=y_pred)
        r2_list.append(model_score)


    return r2_list

    """
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = Regression_Utils.get_chunk_structure(chunk_size, number_of_pixels)
    model_list = []
    dependent_variable, chunk_size=1000
    for chunk_index in tqdm(range(number_of_chunks)):
        chunk_start = chunk_starts[chunk_index]
        chunk_stop = chunk_stops[chunk_index]
        
        chunk_dependent_variable = dependent_variable[:, chunk_start:chunk_stop]
    """

def run_regression(train_x, train_y, test_x, test_y):

    # Create Model
    model = Ridge()

    # Fit Model
    model.fit(train_x, train_y)

    # Make Prediciton
    y_pred = model.predict(test_x)

    # Score Model
    model_score = r2_score(y_true=test_y, y_pred=y_pred, multioutput='uniform_average')

    return model_score




def perform_face_motion_regression(base_directory, save_directory, early_cutoff=3000):

    # Load Face Motion Data
    face_motion_pcs = np.load(os.path.join(base_directory, "Mousecam_analysis", "Widefield_Matched_Face_Motion.npy"))
    number_of_face_motion_components = np.shape(face_motion_pcs)[1]

    # Load Downsampled Delta F Data
    delta_f_matrix = np.load(os.path.join(base_directory, "Downsampled_Aligned_Data.npy"))
    number_of_timepoints = np.shape(delta_f_matrix)[0]
    halfway_point = int((number_of_timepoints-early_cutoff) / 2)
    print("Delta F Matrix Shape", np.shape(delta_f_matrix))

    # Get Sample
    train_start = early_cutoff
    train_stop = early_cutoff + halfway_point
    test_start = train_stop
    test_stop = test_start + halfway_point

    train_x = face_motion_pcs[train_start:train_stop]
    test_x = face_motion_pcs[test_start:test_stop]
    train_y = delta_f_matrix[train_start:train_stop]
    test_y = delta_f_matrix[test_start:test_stop]

    print("Train X", np.shape(train_x))
    print("Train y", np.shape(train_y))
    print("Test X", np.shape(test_x))
    print("Test Y", np.shape(test_y))

    mean_r2_list = []
    for x in range(1, number_of_face_motion_components-1):

        train_motion_data = train_x[:, 0:x]
        test_motion_data = test_x[:, 0:x]

        r_squared = run_regression(train_motion_data, train_y, test_motion_data, test_y)
        print("Components: ", x, "R2", r_squared)
        mean_r2_list.append(r_squared)

    plt.plot(mean_r2_list)
    plt.show()

    session_name = Regression_Utils.get_session_name()
    np.save(os.path.join(save_directory, session_name + "Face_Motion_PCs_and_Regression_Score.npy"), mean_r2_list)


session_list = [r"//media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging"]
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Face_PC_Number_and_Regression_Performance"
perform_face_motion_regression(session_list[0], save_directory)
