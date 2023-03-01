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

import Preprocessing_Utils



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


def get_delta_f_sample(base_directory, sample_start, sample_end, window_size=3):

    # Extract Raw Delta F
    delta_f_file = os.path.join(base_directory, "Downsampled_Delta_F.h5")
    delta_f_file_container = tables.open_file(delta_f_file, mode="r")
    delta_f_matrix  = delta_f_file_container.root["Data"]
    delta_f_sample = delta_f_matrix[sample_start-window_size:sample_end]
    delta_f_sample = np.nan_to_num(delta_f_sample)

    # Denoise with dimensionality reduction
    model = PCA(n_components=150)
    transformed_data = model.fit_transform(delta_f_sample)
    delta_f_sample = model.inverse_transform(transformed_data)

    # Load Mask
    indicies, image_height, image_width = load_downsampled_mask(base_directory)

    # Reconstruct Data
    reconstructed_delta_f = []
    number_of_frames = (sample_end - sample_start) + window_size
    for frame_index in range(number_of_frames):
        frame_data = delta_f_sample[frame_index :frame_index + window_size]
        frame_data = np.mean(frame_data, axis=0)
        template = np.zeros(image_height * image_width)
        template[indicies] = frame_data
        template = np.reshape(template, (image_height, image_width))
        template = ndimage.gaussian_filter(template, sigma=1)

        reconstructed_delta_f.append(template)

    reconstructed_delta_f = np.array(reconstructed_delta_f)

    delta_f_file_container.close()
    return reconstructed_delta_f


def extract_mousecam_data(video_file, frame_list):

    # Open Video File
    cap = cv2.VideoCapture(video_file)

    # Extract Selected Frames
    extracted_data = []
    for frame in frame_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame-1)
        ret, frame = cap.read()
        frame = frame[:, :, 0]
        extracted_data.append(frame)

    cap.release()
    extracted_data = np.array(extracted_data)

    return extracted_data





def get_mousecam_sample(base_directory, mousecam_filename, sample_start, sample_end):

    # Load Widefield Frame Dict
    widefield_frame_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy"), allow_pickle=True)[()]

    # Get Mousecam Frames
    mousecam_frames = []
    for widefield_frame in range(sample_start, sample_end):
        corresponding_mousecam_frame = widefield_frame_dict[widefield_frame]
        mousecam_frames.append(corresponding_mousecam_frame)

    # Extract Mousecam Data
    mousecam_data = extract_mousecam_data(os.path.join(base_directory, mousecam_filename), mousecam_frames)

    return mousecam_data





def create_sample_video_with_mousecam(delta_f_directory, bodycam_directory, output_directory):

    sample_start = 10000
    sample_length = 5000
    sample_end = sample_start + sample_length

    # Get Delta F Sample
    print("Getting Delta F Sample", datetime.now())
    delta_f_sample = get_delta_f_sample(base_directory, sample_start, sample_end)
    print("Finished Getting Delta F Sample", datetime.now())

    # Get Mousecam Sample
    print("Getting Mousecam Sample", datetime.now())
    bodycam_filename = Preprocessing_Utils.get_bodycam_filename(base_directory)
    eyecam_filename = Preprocessing_Utils.get_eyecam_filename(base_directory)
    bodycam_sample = get_mousecam_sample(base_directory, bodycam_filename, sample_start, sample_end)
    eyecam_sample = get_mousecam_sample(base_directory, eyecam_filename, sample_start, sample_end)
    print("Finished Getting Mousecam Sample", datetime.now())

    # Create Colourmaps
    widefield_colourmap = Preprocessing_Utils.get_musall_cmap()
    widefield_colourmap = plt.cm.ScalarMappable(norm=Normalize(vmin=-0.25, vmax=0.25), cmap=widefield_colourmap)
    mousecam_colourmap = plt.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=255), cmap=cm.get_cmap('Greys_r'))

    # Load Mask
    indicies, image_height, image_width = load_downsampled_mask(base_directory)
    background_pixels = Preprocessing_Utils.get_background_pixels(indicies, image_height, image_width)

    # Create Video File
    video_name = os.path.join(base_directory, "Brain_Behaviour_Video.avi")
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, video_codec, frameSize=(1500, 500), fps=30)  # 0, 12

    figure_1 = plt.figure(figsize=(15, 5))
    canvas = FigureCanvasAgg(figure_1)
    for frame_index in range(sample_length):

        rows = 1
        columns = 3
        brain_axis = figure_1.add_subplot(rows, columns, 1)
        body_axis = figure_1.add_subplot(rows, columns, 2)
        eye_axis = figure_1.add_subplot(rows, columns, 3)

        # Extract Frames
        brain_frame = delta_f_sample[frame_index]
        body_frame = bodycam_sample[frame_index]
        eye_frame = eyecam_sample[frame_index]

        # Set Colours
        brain_frame = widefield_colourmap.to_rgba(brain_frame)
        body_frame = mousecam_colourmap.to_rgba(body_frame)
        eye_frame = mousecam_colourmap.to_rgba(eye_frame)
        brain_frame[background_pixels] = (1,1,1,1)

        # Display Images
        brain_axis.imshow(brain_frame)
        body_axis.imshow(body_frame)
        eye_axis.imshow(eye_frame)

        # Remove Axis
        brain_axis.axis('off')
        body_axis.axis('off')
        eye_axis.axis('off')

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

base_directory = r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Mutant_Data/NXAK10.1A/2021_05_06_Discrimination_Imaging"
create_sample_video_with_mousecam(base_directory, base_directory)