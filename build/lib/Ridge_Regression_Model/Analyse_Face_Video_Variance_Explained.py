import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os
import tables
from bisect import bisect_left
from sklearn.linear_model import Ridge, LinearRegression
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
        model = Lin()

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
    model = LinearRegression()

    # Fit Model
    model.fit(X=train_x, y=train_y)

    # Make Prediciton
    y_pred = model.predict(test_x)

    # Score Model
    model_score = r2_score(y_true=test_y, y_pred=y_pred, multioutput='raw_values')

    weights = model.coef_

    return model_score, weights


def get_video_name(base_directory):

    file_list = os.listdir(base_directory)
    for file in file_list:
        if "_cam_1" in file:
            return file


def get_video_details(base_directory, video_name):

    # Open Video File
    cap = cv2.VideoCapture(os.path.join(base_directory, video_name))

    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return frameCount, frameHeight, frameWidth


def view_face_weights(base_directory, face_pixels, regression_coefs):

    # Get Video Name
    video_name = get_video_name(base_directory)

    # Get Video Details
    frameCount, image_height, image_width = get_video_details(base_directory, video_name)

    # Get Face Details
    number_of_face_pixels = np.shape(face_pixels)[0]
    face_y_min = np.min(face_pixels[:, 0])
    face_y_max = np.max(face_pixels[:, 0])
    face_x_min = np.min(face_pixels[:, 1])
    face_x_max = np.max(face_pixels[:, 1])

    figure_1 = plt.figure(figsize=(15, 10))

    template = np.zeros((image_height, image_width))

    for face_pixel_index in range(number_of_face_pixels):
        pixel_data = regression_coefs[face_pixel_index]
        pixel_position = face_pixels[face_pixel_index]
        template[pixel_position[0], pixel_position[1]] = pixel_data

    template = template[face_y_min:face_y_max, face_x_min:face_x_max]
    template_magnitude = np.percentile(np.abs(template), q=99)

    axis = figure_1.add_subplot(1, 1, 1)
    axis.axis('off')
    axis.imshow(template, vmax=template_magnitude, vmin=-template_magnitude, cmap='bwr')

    plt.show()


def view_face_motion(base_directory, face_pixels, face_motion_data):

    # Get Video Name
    video_name = get_video_name(base_directory)

    # Get Video Details
    frameCount, image_height, image_width = get_video_details(base_directory, video_name)
    print("Image Height", image_height)
    print("image width", image_width)
    number_of_face_pixels = np.shape(face_pixels)[0]
    face_y_min = np.min(face_pixels[:, 0])
    face_y_max = np.max(face_pixels[:, 0])
    face_x_min = np.min(face_pixels[:, 1])
    face_x_max = np.max(face_pixels[:, 1])

    motion_magnitude = np.percentile(face_motion_data, q=99)

    plt.ion()
    count =0
    for frame in face_motion_data:
        template = np.zeros((image_height, image_width))
        for face_pixel_index in range(number_of_face_pixels):
            pixel_data = frame[face_pixel_index]
            pixel_position = face_pixels[face_pixel_index]
            template[pixel_position[0], pixel_position[1]] = pixel_data

        template = template[face_y_min:face_y_max, face_x_min:face_x_max]

        plt.imshow(template, cmap='hot', vmin=0, vmax=motion_magnitude)
        plt.draw()
        plt.title(str(count))
        plt.pause(0.1)
        plt.clf()
        count += 1


def perform_face_motion_regression(base_directory, save_directory, early_cutoff=3000):

    # Load Face Motion Data
    face_motion_data = np.load(os.path.join(base_directory, "Mousecam_analysis", "Face_Motion.npy"))
    print("Face Motion Data", np.shape(face_motion_data))

    # Load Face Pixels
    face_pixels = np.load(os.path.join(base_directory, "Mousecam_analysis", "Whisker_Pixels.npy"))
    face_pixels = np.transpose(face_pixels)

    print("Face Pxiels", np.shape(face_pixels))
    # View Face motion
    #view_face_motion(base_directory, face_pixels, face_motion_data)

    # Load Downsampled Delta F Data
    delta_f_matrix = np.load(os.path.join(base_directory, "Downsampled_Aligned_Data.npy"))
    number_of_timepoints = np.shape(delta_f_matrix)[0]
    print("Delta F Matrix Shape", np.shape(delta_f_matrix))

    # Get Sample
    halfway_point = int((number_of_timepoints-early_cutoff) / 2)
    train_start = early_cutoff
    train_stop = early_cutoff + halfway_point
    test_start = train_stop
    test_stop = test_start + halfway_point

    train_x = face_motion_data[train_start:train_stop]
    test_x = face_motion_data[test_start:test_stop]
    train_y = delta_f_matrix[train_start:train_stop]
    test_y = delta_f_matrix[test_start:test_stop]

    print("Train X", np.shape(train_x))
    print("Train y", np.shape(train_y))
    print("Test X", np.shape(test_x))
    print("Test Y", np.shape(test_y))

    r_squared, regression_coefs = run_regression(train_x, train_y, test_x, test_y)
    print("R2", np.mean(r_squared))
    print("Coefs", np.shape(regression_coefs))

    indicies, image_height, image_width = Regression_Utils.load_tight_mask_downsized()
    r2_map = Regression_Utils.create_image_from_data(r_squared, indicies, image_height, image_width)
    plt.imshow(r2_map, cmap='inferno', vmin=0)
    plt.show()

    # Visualise Coefs
    plt.imshow(regression_coefs)
    plt.show()

    model = PCA(n_components=10)
    model.fit(np.transpose(regression_coefs))
    components = model.components_
    print("Component Shape", np.shape(components))

    for component in components:
        coef_map = Regression_Utils.create_image_from_data(component, indicies, image_height, image_width)
        plt.imshow(coef_map)
        plt.show()

    model = PCA(n_components=10)
    model.fit(regression_coefs)
    components = model.components_
    print("Component Shape", np.shape(components))
    for component in components:
        view_face_weights(base_directory, face_pixels, component)
    #session_name = Regression_Utils.get_session_name()
    #np.save(os.path.join(save_directory, session_name + "Face_Motion_PCs_and_Regression_Score.npy"), mean_r2_list)


session_list = [r"//media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging"]
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Face_PC_Number_and_Regression_Performance"
perform_face_motion_regression(session_list[0], save_directory)
