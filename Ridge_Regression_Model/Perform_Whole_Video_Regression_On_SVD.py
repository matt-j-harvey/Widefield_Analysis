import numpy as np
import matplotlib.pyplot as plt
import tables
import os
from bisect import bisect_left
import cv2
from tqdm import tqdm
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from skimage.transform import resize
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.decomposition import IncrementalPCA

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph

pyqtgraph.setConfigOptions(imageAxisOrder='row-major')

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


def run_regression(train_x, train_y, test_x, test_y):

    # Create Model
    #model = Lasso(normalize=True,  tol=1e-2)
    model = LinearRegression()

    # Fit Model
    model.fit(X=train_x, y=train_y)

    # Make Prediciton
    y_pred = model.predict(test_x)

    # Score Model
    model_score = explained_variance_score(y_true=test_y, y_pred=y_pred, multioutput='raw_values')

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
        model = Ridge(solver='sparse_cg')

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


def get_matched_motion_energy_frame(base_directory, sample_start, sample_end):

    # Load Widefield Frame Dict
    widefield_frame_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy"), allow_pickle=True)[()]

    # Get Mousecam Frames
    mousecam_frames = []
    for widefield_frame in range(sample_start, sample_end):
        corresponding_mousecam_frame = widefield_frame_dict[widefield_frame]
        mousecam_frames.append(corresponding_mousecam_frame)

    return mousecam_frames

def get_bodycam_sample(bodycam_matrix, selected_frames):
    extracted_data = []
    for frame in selected_frames:
        extracted_data.append(bodycam_matrix[frame])
    extracted_data = np.array(extracted_data)
    return extracted_data


def get_coef_partial_det_map(train_x, train_y, test_x, test_y):

    number_of_components = np.shape(train_x)[1]

    # Create Model
    model = LinearRegression()

    # Fit Model
    model.fit(X=train_x, y=train_y)

    partial_coef_matrix = []
    for componen_index in tqdm(range(number_of_components)):

        # Make Prediciton
        component_test = np.copy(test_x)
        component_test[:, componen_index] = 0

        y_pred = model.predict(component_test)

        # Score Model
        model_score = explained_variance_score(y_true=test_y, y_pred=y_pred, multioutput='raw_values')

        partial_coef_matrix.append(model_score)

    partial_coef_matrix = np.array(partial_coef_matrix)
    return partial_coef_matrix



def regress_activity_onto_whole_video(base_directory, early_cutoff=3000):

    # Load Bodycam Motion Energy
    bodycam_motion_data = np.load(os.path.join(base_directory, "Mousecam_Analysis", "Transformed_Mousecam_Data.npy"))
    print("Bodycam Motion Data", np.shape(bodycam_motion_data))

    # Load Activity Matrix
    delta_f_matrix = np.load(os.path.join(base_directory, "100_By_100_Data.npy"))
    number_of_frames, number_of_pixels = np.shape(delta_f_matrix)
    print("Delta F Matrix", np.shape(delta_f_matrix))

    # Get Sample
    video_length = (number_of_frames - early_cutoff)
    sample_size = int(video_length/2)
    print("Sample Size", sample_size)
    train_start = early_cutoff
    train_stop = early_cutoff + sample_size
    test_start = train_stop
    test_stop = test_start + sample_size

    train_mousecam_frames = get_matched_motion_energy_frame(base_directory, train_start, train_stop)
    test_mousecam_frames = get_matched_motion_energy_frame(base_directory, test_start, test_stop)

    train_x = bodycam_motion_data[train_mousecam_frames]
    test_x = bodycam_motion_data[test_mousecam_frames]
    train_y = delta_f_matrix[train_start:train_stop]
    test_y = delta_f_matrix[test_start:test_stop]

    print("Train x", np.shape(train_x))
    print("Test x", np.shape(test_x))
    print("Train Y", np.shape(train_y))
    print("Test y", np.shape(test_y))

    # Perform Regression
    #r_squared, regression_coefs = incremental_regression(train_x, train_y, test_x, test_y)
    r_squared, regression_coefs = run_regression(train_x, train_y, test_x, test_y)

    # View R2 Map
    indicies, image_height, image_width = load_smallest_mask(base_directory)
    r2_map = Regression_Utils.create_image_from_data(r_squared, indicies, image_height, image_width)
    plt.imshow(r2_map)
    plt.show()

    plt.imshow(r2_map, vmin=0)
    plt.show()

    #partial_determinant_matrix = get_coef_partial_det_map(train_x, train_y, test_x, test_y)

    # Save These
    save_directory = os.path.join(base_directory, "Mousecam_Analysis")
    np.save(os.path.join(save_directory, "Whole_Video_R2.npy"), r_squared)
    np.save(os.path.join(save_directory, "Whole_Video_Coefs.npy"), regression_coefs)
    #np.save(os.path.join(save_directory, "Coefs_Partial_Det.npy"), partial_determinant_matrix)

session = r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging"
regress_activity_onto_whole_video(session)
#regress_activity_onto_whole_video(session)