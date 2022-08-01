import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import TruncatedSVD
from matplotlib.pyplot import Normalize


def get_video_name(base_directory):

    file_list = os.listdir(base_directory)
    for file in file_list:
        if "_cam_1" in file:
            return file


def extract_face_video(video_file, face_pixels):

    # Open Video File
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Extract Selected Frames
    frame_index = 0
    ret = True
    face_data = []
    while (frame_index < frameCount and ret):
        ret, frame = cap.read()
        frame = frame[:, :, 0]

        face_frame = []
        for pixel in face_pixels:
            face_frame.append(frame[pixel[0], pixel[1]])

        face_data.append(face_frame)
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
    while (frame_index < 1000 and ret):
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




def view_face_video(face_pixels, face_data, full_video):

    greyscale_cmap = cm.get_cmap('Greys_r')
    motion_cmap = cm.get_cmap("inferno")
    greyscale_norm = Normalize(vmin=0, vmax=255)
    motion_norm = Normalize(vmin=0, vmax=255)
    greyscale_cmap_mappable = cm.ScalarMappable(cmap=greyscale_cmap, norm=greyscale_norm)
    motion_cmap_mappable = cm.ScalarMappable(cmap=motion_cmap, norm=motion_norm)

    number_of_frames, number_of_face_pixels = np.shape(face_data)

    plt.ion()
    for x in range(number_of_frames):

        raw_frame = full_video[x]
        raw_frame = greyscale_cmap_mappable.to_rgba(raw_frame)

        for face_pixel_index in range(number_of_face_pixels):
            pixel = face_pixels[face_pixel_index]
            raw_frame[pixel[0], pixel[1]] = motion_cmap_mappable.to_rgba(face_data[x, face_pixel_index])

        plt.clf()
        plt.imshow(raw_frame)
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


session_list = [r"/media/matthew/External_Harddrive_1/Processed_Widefield_Data/NRXN78.1D/2020_12_07_Switching_Imaging"]

for session in session_list:

    # Load Facepoly
    face_pixels = np.load(os.path.join(session, "Mousecam_analysis", "Whisker_Pixels.npy"))
    face_pixels = np.transpose(face_pixels)
    print("Face Pixels", np.shape(face_pixels))
    print("Face Pixels 0", np.max(face_pixels[0]))
    print("Face Pixels 1", np.max(face_pixels[1]))

    # Get Face Video
    video_name = get_video_name(session)

    # Extract Face Video
    frame_height, frame_width, face_data, full_video = extract_face_video_with_full_video(os.path.join(session, video_name), face_pixels)
    print("Face Data", np.shape(face_data))

    # Convert To Motion Energy
    face_data = get_motion_energy(face_data)

    # Decompose Into 20 Components
    face_data = reconstruct_video(face_data)

    # View Video
    #view_face_video(face_pixels, face_data, full_video)


    face_data = extract_face_video(os.path.join(session, video_name), face_pixels)

    # Convert To Motion Energy

    face_data = get_motion_energy(face_data)

    # Decompose Into 20 Components
    transformed_data, components = decompose_motion_energy(face_data)

    np.save(os.path.join(session, "Mousecam_analysis", "Face_Motion_Temporal_Components.npy"), transformed_data)
    np.save(os.path.join(session, "Mousecam_analysis", "Face_Motion_Spatial_Components.npy"), components)