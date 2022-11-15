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

import Regression_Utils



def get_video_name(base_directory):

    file_list = os.listdir(base_directory)
    for file in file_list:
        if "_cam_1" in file:
            return file


def view_bodycam_motion_energy(base_directory):

    bodycam_motion_energy_file = os.path.join(base_directory, "Mousecam_Analysis", "Bodycam_Motion_Energy.h5")
    bodycam_motion_energy_container = tables.open_file(bodycam_motion_energy_file, "r")
    bodycam_motion_energy = bodycam_motion_energy_container.root["blue"]

    figure_1 = plt.figure()
    plt.ion()
    for frame in bodycam_motion_energy:

        frame = np.reshape(frame, (480, 640))
        axis_1 = figure_1.add_subplot(1,1,1)
        axis_1.imshow(frame, vmin=0, vmax=100)
        plt.draw()
        plt.pause(0.1)
        plt.clf()


    bodycam_motion_energy_container.close()



def compute_body_motion_svd(base_directory):

    # Load Bodycam Motion Energy
    bodycam_motion_energy_file = os.path.join(base_directory, "Mousecam_Analysis",  "Bodycam_Motion_Energy.h5")
    bodycam_motion_container = tables.open_file(bodycam_motion_energy_file, "r")
    bodycam_motion_data = bodycam_motion_container.root["blue"]
    number_of_frames, number_of_pixels = np.shape(bodycam_motion_data)
    print("Bodycam Motion Data", np.shape(bodycam_motion_data))


    preferred_chunk_size = 1000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = Regression_Utils.get_chunk_structure(preferred_chunk_size, number_of_frames)

    # Fit Data
    print("Fitting Model")
    model = IncrementalPCA(n_components=100)
    for chunk_index in tqdm(range(number_of_chunks)):
        chunk_start = chunk_starts[chunk_index]
        chunk_stop = chunk_stops[chunk_index]
        chunk_data = bodycam_motion_data[chunk_start:chunk_stop]
        if chunk_stop - chunk_start > 100:
            model.partial_fit(chunk_data)

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


def visualise_svd_decomposition(base_directory):

    # Load Model
    model = pickle.load(open(os.path.join(base_directory, "Mousecam_analysis",  "SVD Model.sav"), 'rb'))

    components = model.components_

    for component in components:
        component = np.reshape(component, (480, 640))

        plt.imshow(component)
        plt.show()


def view_face_image(face_data, face_pixels):

    template = np.zeros((480, 640))
    face_pixels = np.transpose(face_pixels)

    count = 0
    for pixel in face_pixels:
        template[pixel[0], pixel[1]] = face_data[count]
        count += 1

    plt.imshow(template)
    plt.show()

def get_face_pixel_indicies(face_pixels):

    template = np.zeros((480, 640))
    face_pixels = np.transpose(face_pixels)
    for pixel in face_pixels:
        template[pixel[0], pixel[1]] = 1

    template = np.reshape(template, 480 * 640)
    indicies = np.nonzero(template)
    return indicies


def extract_face_motion_only(base_directory):

     # Load Bodycam Motion Energy
     bodycam_motion_energy_file = os.path.join(base_directory, "Mousecam_Analysis", "Bodycam_Motion_Energy.h5")
     bodycam_motion_container = tables.open_file(bodycam_motion_energy_file, "r")
     bodycam_motion_data = bodycam_motion_container.root["blue"]
     number_of_frames, number_of_pixels = np.shape(bodycam_motion_data)

     # Load Face Pixels
     face_pixels = np.load(os.path.join(base_directory, "Mousecam_Analysis", "Whisker_Pixels.npy"))
     print("Face Pixel Shape", np.shape(face_pixels))
     # Get Face Pixel Indicies
     face_indicies = get_face_pixel_indicies(face_pixels)

     face_motion_data = []
     for frame in tqdm(bodycam_motion_data):
         face_data = frame[face_indicies]
         face_motion_data.append(face_data)

     face_motion_data = np.array(face_motion_data)
     np.save(os.path.join(base_directory, "Mousecam_Analysis", "Face_Motion_Data.npy"), face_motion_data)


def create_face_motion_video(base_directory):

    # Load Face Pixels
    face_pixels = np.load(os.path.join(base_directory, "Mousecam_Analysis", "Whisker_Pixels.npy"))

    # Get Face Extent
    face_y_min = np.min(face_pixels[0])
    face_y_max = np.max(face_pixels[0])
    face_x_min = np.min(face_pixels[1])
    face_x_max = np.max(face_pixels[1])

    face_height = face_y_max - face_y_min
    face_width = face_x_max - face_x_min

    face_pixels = np.transpose(face_pixels)


    # Load Face Motion Data
    face_motion_data = np.load(os.path.join(base_directory,"Mousecam_Analysis", "Face_Motion_Data.npy"))

    # Create Video File
    reconstructed_video_file = os.path.join(os.path.join(base_directory,"Mousecam_Analysis", "Face_Motion_Data.avi"))
    video_name = reconstructed_video_file
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, video_codec, frameSize=(face_width, face_height), fps=30)  # 0, 12

    # Create Colourmap
    colourmap = plt.cm.ScalarMappable(norm=None, cmap='viridis')
    colour_max = 50
    colour_min = 0
    colourmap.set_clim(vmin=colour_min, vmax=colour_max)

    plt.ion()
    for frame in tqdm(face_motion_data):
        template = np.zeros((480, 640))

        count = 0
        for pixel in face_pixels:
            template[pixel[0], pixel[1]] = frame[count]
            count += 1

        template = template[face_y_min:face_y_max, face_x_min:face_x_max]
        template = colourmap.to_rgba(template)
        template = np.multiply(template, 255)
        template = np.ndarray.astype(template, np.uint8)
        template = cv2.cvtColor(template, cv2.COLOR_RGB2BGR)
        video.write(template)

    cv2.destroyAllWindows()
    video.release()



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

def get_face_svd(base_directory):


    # Load Face Motion Data
    face_motion_data = np.load(os.path.join(base_directory,"Mousecam_Analysis", "Face_Motion_Data.npy"))

    print("Performing NMF", datetime.now())
    model = NMF(n_components=10)
    transformed_data = model.fit_transform(face_motion_data)

    components = model.components_

    face_pixels = np.load(os.path.join(base_directory, "Mousecam_Analysis", "Whisker_Pixels.npy"))
    view_face_componnents(components, face_pixels)

    np.save(os.path.join(base_directory, "Mousecam_Analysis", "Transformed_Face_Data.npy"), transformed_data)
    np.save(os.path.join(base_directory, "Mousecam_Analysis", "Face_SVD_Components.npy"), components)



def view_face_componnents(regression_coefs, face_pixels):

    # Get Face Extent
    face_y_min = np.min(face_pixels[0])
    face_y_max = np.max(face_pixels[0])
    face_x_min = np.min(face_pixels[1])
    face_x_max = np.max(face_pixels[1])

    face_pixels = np.transpose(face_pixels)

    reconstructed_coefs = []
    for coef in tqdm(regression_coefs):
        template = np.zeros((480, 640))

        count = 0
        for pixel in face_pixels:
            template[pixel[0], pixel[1]] = coef[count]
            count += 1

        template = template[face_y_min:face_y_max, face_x_min:face_x_max]
        reconstructed_coefs.append(template)

        plt.imshow(template)
        plt.show()

    reconstructed_coefs = np.array(reconstructed_coefs)
    return reconstructed_coefs


session_list = [
    r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_23_Transition_Imaging",
    r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_31_Transition_Imaging",
    r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_04_02_Transition_Imaging",
]


for base_directory in session_list:
    #get_bodycam_motion_energy(base_directory)
    compute_body_motion_svd(base_directory)