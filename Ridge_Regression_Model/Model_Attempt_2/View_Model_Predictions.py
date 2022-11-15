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


def get_matched_motion_energy_frame(base_directory, sample_start, sample_end):

    # Load Widefield Frame Dict
    widefield_frame_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy"), allow_pickle=True)[()]

    # Get Mousecam Frames
    mousecam_frames = []
    for widefield_frame in range(sample_start, sample_end):
        corresponding_mousecam_frame = widefield_frame_dict[widefield_frame]
        mousecam_frames.append(corresponding_mousecam_frame)

    return mousecam_frames

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


def denoise_data(data):
    model = PCA(n_components=250)
    transformed_data = model.fit_transform(data)
    data = model.inverse_transform(transformed_data)
    return data


def compare_prediction(base_directory, actual_data, prediction, mouseface_motion):
    figure_1 = plt.figure()
    rows = 1
    columns = 4

    number_of_timepoints = np.shape(actual_data)[0]

    # Get Model Residuals
    residuals = np.subtract(actual_data, prediction)

    # Load Mask
    indicies, image_height, image_width = load_smallest_mask(base_directory)

    # Create Colourmap
    widefield_colourmap = Regression_Utils.get_musall_cmap()
    widefield_colourmap = plt.cm.ScalarMappable(norm=Normalize(vmin=-0.05, vmax=0.05), cmap=widefield_colourmap)
    mousecam_colourmap = plt.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=255), cmap=cm.get_cmap('viridis'))

    # Create Video File
    video_name = os.path.join(base_directory, "Model_and_Residuals.avi")
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, video_codec, frameSize=(1500, 500), fps=30)  # 0, 12

    figure_1 = plt.figure(figsize=(15, 5))
    canvas = FigureCanvasAgg(figure_1)

    plt.ion()
    window_size=3
    for timepoint_index in tqdm(range(window_size, number_of_timepoints)):

        # Create Axes
        #face_axis = figure_1.add_subplot(rows, columns, 1)
        real_axis = figure_1.add_subplot(rows, columns, 1)
        predicted_axis = figure_1.add_subplot(rows, columns, 3)
        residual_axis = figure_1.add_subplot(rows, columns, 2)

        # Load Frames
        real_image = np.mean(actual_data[timepoint_index-window_size:timepoint_index], axis=0)
        predicted_image = np.mean(prediction[timepoint_index-window_size:timepoint_index], axis=0)
        residual_image = np.mean(residuals[timepoint_index-window_size:timepoint_index], axis=0)

        # Create Image
        real_image = Regression_Utils.create_image_from_data(real_image, indicies, image_height, image_width)
        predicited_image = Regression_Utils.create_image_from_data(predicted_image, indicies, image_height, image_width)
        residual_image = Regression_Utils.create_image_from_data(residual_image, indicies, image_height, image_width)

        # Add Colour
        real_image = widefield_colourmap.to_rgba(real_image)
        predicited_image = widefield_colourmap.to_rgba(predicited_image)
        residual_image = widefield_colourmap.to_rgba(residual_image)

        # Display Images
        real_axis.imshow(real_image)
        predicted_axis.imshow(predicited_image)
        residual_axis.imshow(residual_image)

        # Remove Axis
        real_axis.axis('off')
        predicted_axis.axis('off')
        residual_axis.axis('off')

        figure_1.canvas.draw()

        # Write To Video
        canvas.draw()
        buf = canvas.buffer_rgba()
        image_from_plot = np.asarray(buf)
        image_from_plot = cv2.cvtColor(image_from_plot, cv2.COLOR_RGB2BGR)
        video.write(image_from_plot)

        plt.clf()

    cv2.destroyAllWindows()
    video.release()

def view_model_predictions(base_directory, early_cutoff=3000):

    # Load Face Motion Data
    bodycam_motion_data = np.load(os.path.join(base_directory, "Mousecam_Analysis", "Transformed_Face_Data.npy"))

    # Load Activity Matrix
    delta_f_matrix = np.load(os.path.join(base_directory, "100_By_100_Data.npy"))
    number_of_frames, number_of_pixels = np.shape(delta_f_matrix)

    # Get Sample
    video_length = (number_of_frames - early_cutoff)
    sample_size = int(video_length / 2)
    train_stop = early_cutoff + sample_size
    test_start = train_stop
    test_stop = test_start + sample_size

    # Load Running and Licking
    ai_data = np.load(os.path.join(base_directory, "Downsampled_AI_Matrix_Framewise.npy"))
    stimuli_dict = Regression_Utils.create_stimuli_dictionary()
    lick_trace = ai_data[stimuli_dict["Lick"]]
    running_trace = ai_data[stimuli_dict["Running"]]

    # Get Matched Mousecam Frames
    test_mousecam_frames = get_matched_motion_energy_frame(base_directory, test_start, test_stop)
    bodycam_test = bodycam_motion_data[test_mousecam_frames]
    lick_test = np.expand_dims(lick_trace[test_start:test_stop], 1)
    running_test = np.expand_dims(running_trace[test_start:test_stop], 1)
    test_y = delta_f_matrix[test_start:test_stop]
    test_x = np.hstack([bodycam_test, lick_test, running_test])

    # Denoise Preidicted Data
    test_y = denoise_data(test_y)

    # Load Model
    save_directory = os.path.join(base_directory, "Mousecam_Analysis")
    model = pickle.load(open(os.path.join(save_directory, 'Face_Model.sav'), 'rb'))

    # Get Model Prediction
    print("Predicting")
    model_prediction = model.predict(test_x)
    print()

    # View This
    compare_prediction(base_directory, test_y, model_prediction, test_x)


session = r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging"
view_model_predictions(session)
# regress_activity_onto_whole_video(session)