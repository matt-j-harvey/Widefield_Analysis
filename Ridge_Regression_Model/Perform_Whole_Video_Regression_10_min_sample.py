import numpy as np
import matplotlib.pyplot as plt
import tables
import os
from bisect import bisect_left
import cv2
from tqdm import tqdm
from sklearn.linear_model import Ridge, Lasso, LinearRegression, SGDRegressor
from skimage.transform import resize
from sklearn.metrics import r2_score
from sklearn.decomposition import TruncatedSVD

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph

pyqtgraph.setConfigOptions(imageAxisOrder='row-major')

import Regression_Utils


def run_regression(train_x, train_y, test_x, test_y):


    # Create Model
    model = Lasso(alpha=2, max_iter=10000)

    # Fit Model
    model.fit(X=train_x, y=train_y)

    # Make Prediciton
    y_pred = model.predict(test_x)

    # Score Model
    model_score = r2_score(y_true=test_y, y_pred=y_pred, multioutput='raw_values')

    weights = model.coef_

    return model_score, weights



def incremental_regression(train_x, train_y, test_x, test_y):

    number_of_pixels = np.shape(train_y)[1]

    weights_list = []
    r2_list = []

    for pixel_index in tqdm(range(number_of_pixels)):

        # Get Pixel Data
        pixel_train_y = train_y[:, pixel_index]
        pixel_test_y = test_y[:, pixel_index]
        print("Pixel train y", np.shape(pixel_train_y))
        print("Pixel Test Y", np.shape(pixel_test_y))
        # Create Model
        #model = Ridge(solver='sag')
        #model = Lasso()
        chunk_size = 1000
        model = SGDRegressor(penalty='l1')

        # Fit Model
        model.fit(X=train_x, y=pixel_train_y)

        # Make Prediciton
        y_pred = model.predict(test_x)

        # Score Model
        model_score = r2_score(y_true=pixel_test_y, y_pred=y_pred)
        r2_list.append(model_score)
        print("Model Score", model_score)

        # Save Weights
        weights = model.coef_
        weights_list.append(weights)

    score_list = np.array(r2_list)
    weights_list = np.array(weights_list)
    return score_list, weights_list


def view_motion_energy_and_activity(motion_energy, activity):

    indicies, image_height, image_width = Regression_Utils.load_tight_mask_downsized()

    number_of_frames = np.shape(motion_energy)[0]

    figure_1 = plt.figure()
    rows = 1
    columns = 2


    cmap = Regression_Utils.get_musall_cmap()

    plt.ion()
    for frame_index in range(number_of_frames):
        print(frame_index)


        motion_frame = np.reshape(motion_energy[frame_index], (240, 320))
        activity_frame = Regression_Utils.create_image_from_data( activity[frame_index], indicies, image_height, image_width)

        axis_1 = figure_1.add_subplot(rows, columns, 1)
        axis_2 = figure_1.add_subplot(rows, columns, 2)

        #axis_1.imshow(motion_frame, vmin=0, vmax=50, cmap='inferno')
        #axis_2.imshow(activity_frame, vmin=-0.03, vmax=0.03, cmap=cmap)

        axis_1.imshow(motion_frame)
        axis_2.imshow(activity_frame)

        plt.draw()
        plt.pause(0.1)
        plt.clf()

def get_motion_energy_matched_sample(motion_energy_matrix, brain_frame_start, brain_frame_stop, widefield_frame_dict):

    matched_motion_energy = []
    print("Matching Mousecam Frames")
    for widefield_frame in tqdm(range(brain_frame_start, brain_frame_stop)):
        corresponding_mousecam_frame = widefield_frame_dict[widefield_frame]
        matched_motion_energy.append(motion_energy_matrix[corresponding_mousecam_frame])

    matched_motion_energy = np.array(matched_motion_energy)

    # Decompose
    #model = TruncatedSVD(n_components=100)
    #transformed_data = model.fit_transform(matched_motion_energy)

    return matched_motion_energy





def regress_activity_onto_whole_video(base_directory, early_cutoff=3000):

    # Load Bodycam Motion Energy
    bodycam_motion_energy_file = os.path.join(base_directory, "Mousecam_analysis", "Bodycam_Motion_Energy.h5")
    bodycam_motion_container = tables.open_file(bodycam_motion_energy_file, "r")
    bodycam_motion_data = bodycam_motion_container.root["blue"]
    print("Bodycam Motion Data", np.shape(bodycam_motion_data))

    # Load Activity Matrix
    delta_f_matrix = np.load(os.path.join(base_directory, "Downsampled_Aligned_Data.npy"))
    number_of_frames, number_of_pixels = np.shape(delta_f_matrix)

    # Get Sample
    # 10 mins = 16666 Frames
    sample_size = 1000 #16666
    train_start = early_cutoff
    train_stop = early_cutoff + sample_size
    test_start = train_stop
    test_stop = test_start + sample_size

    # Load Widefield To Mousecam Frame Dict
    widefield_to_mousecam_frame_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy"), allow_pickle=True)[()]

    train_x = get_motion_energy_matched_sample(bodycam_motion_data, train_start, train_stop, widefield_to_mousecam_frame_dict)
    test_x = get_motion_energy_matched_sample(bodycam_motion_data, test_start, test_stop, widefield_to_mousecam_frame_dict)
    train_y = delta_f_matrix[train_start:train_stop]
    test_y = delta_f_matrix[test_start:test_stop]

    #view_motion_energy_and_activity(train_x, train_y)
    print("Train x", np.shape(train_x))
    print("Test x", np.shape(test_x))
    print("Train Y", np.shape(train_y))
    print("Test y", np.shape(test_y))


    # Perform Regression
    #r_squared, regression_coefs = incremental_regression(train_x, train_y, test_x, test_y)
    r_squared, regression_coefs = run_regression(train_x, train_y, test_x, test_y)

    # View R2 Map
    indicies, image_height, image_width = Regression_Utils.load_tight_mask_downsized()
    r2_map = Regression_Utils.create_image_from_data(r_squared, indicies, image_height, image_width)

    plt.imshow(r2_map, vmin=0)
    plt.show()

    plt.imshow(r2_map, cmap='jet')
    plt.show()

    # Save These
    save_directory = os.path.join(base_directory, "Mousecam_analysis")
    np.save(os.path.join(save_directory, "Whole_Video_R2.npy"), r_squared)
    np.save(os.path.join(save_directory, "Whole_Video_Coefs.npy"), regression_coefs)



session = r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging"
regress_activity_onto_whole_video(session)