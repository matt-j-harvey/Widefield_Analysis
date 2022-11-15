import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import TruncatedSVD
from matplotlib.pyplot import Normalize

from tqdm import tqdm


def get_video_name(base_directory):

    file_list = os.listdir(base_directory)
    for file in file_list:
        if "_cam_1" in file:
            return file


def extract_video_sample(video_file, sample_size=100):

    # Open Video File
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Extract Selected Frames
    face_data = []
    print("Sample Size", sample_size)
    for frame_index in tqdm(range(sample_size)):
        ret, frame = cap.read()
        frame = frame[:, :, 0]
        face_data.append(frame)
        frame_index += 1

    cap.release()
    face_data = np.array(face_data)

    return face_data


def extract_face_video_with_full_video(video_file, face_pixels):

    # Open Video File
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Extract Selected Frames
    frame_index = 0
    ret = True
    face_data = []
    full_video = []
    frame_count = 5000
    while (frame_index < frameCount and ret):
        ret, frame = cap.read()
        frame = frame[:, :, 0]

        face_frame = []
        for pixel in face_pixels:
            face_frame.append(frame[pixel[0], pixel[1]])

        full_video.append(frame)
        face_data.append(face_frame)
        frame_index += 1

    cap.release()
    face_data = np.array(face_data)
    full_video = np.array(full_video)

    return frameHeight, frameWidth, face_data, full_video




def view_face_video(face_pixels, full_video):

    greyscale_cmap = cm.get_cmap('Greys_r')
    motion_cmap = cm.get_cmap("inferno")
    greyscale_norm = Normalize(vmin=0, vmax=255)
    motion_norm = Normalize(vmin=0, vmax=255)
    greyscale_cmap_mappable = cm.ScalarMappable(cmap=greyscale_cmap, norm=greyscale_norm)
    motion_cmap_mappable = cm.ScalarMappable(cmap=motion_cmap, norm=motion_norm)

    number_of_frames, height, width = np.shape(full_video)

    plt.ion()
    for x in range(number_of_frames):

        raw_frame = full_video[x]
        greyscale_frame = greyscale_cmap_mappable.to_rgba(raw_frame)

        for pixel in face_pixels:
            greyscale_frame[pixel[0], pixel[1]] = motion_cmap_mappable.to_rgba(raw_frame[pixel[0], pixel[1]])

        plt.clf()
        plt.imshow(greyscale_frame)
        plt.draw()
        plt.pause(0.1)


def get_motion_energy(face_data):

    motion = np.diff(face_data, axis=0)
    motion = np.abs(motion)
    return motion


def reconstruct_video(motion_energy, n_components=20):

    model = TruncatedSVD(n_components=n_components)

    transformed_data = model.fit_transform(motion_energy)

    reconstructed_video = model.inverse_transform(transformed_data)

    return reconstructed_video



def decompose_motion_energy(motion_energy, n_components=20):
    model = TruncatedSVD(n_components=n_components)

    transformed_data = model.fit_transform(motion_energy)
    components = model.components_

    return transformed_data, components


session_list = [r"//media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging"]

for session in session_list:

    # Load Facepoly
    face_pixels = np.load(os.path.join(session, "Mousecam_analysis", "Whisker_Pixels.npy"))
    face_pixels = np.transpose(face_pixels)

    # Get Face Video
    video_name = get_video_name(session)

    # Extract Face Video
    sample_video = extract_video_sample(os.path.join(session, video_name))

    # View Video
    view_face_video(face_pixels, sample_video)

