import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os
import tables
from bisect import bisect_left
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
import cv2
from tqdm import tqdm
from datetime import datetime

import Regression_Utils
#import Match_Mousecam_Frames_To_Widefield_Frames


def match_mousecam_to_widefield_frames(base_directory):

    # Load Frame Times
    widefield_frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    mousecam_frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Mousecam_Frame_Times.npy"), allow_pickle=True)[()]

    widefield_frame_times = invert_dictionary(widefield_frame_times)
    widefield_frame_time_keys = list(widefield_frame_times.keys())
    mousecam_frame_times_keys = list(mousecam_frame_times.keys())
    mousecam_frame_times_keys.sort()

    # Get Number of Frames
    number_of_widefield_frames = len(widefield_frame_time_keys)

    # Dictionary - Keys are Widefield Frame Indexes, Values are Closest Mousecam Frame Indexes
    widfield_to_mousecam_frame_dict = {}

    for widefield_frame in range(number_of_widefield_frames):
        frame_time = widefield_frame_times[widefield_frame]
        closest_mousecam_time = take_closest(mousecam_frame_times_keys, frame_time)
        closest_mousecam_frame = mousecam_frame_times[closest_mousecam_time]
        widfield_to_mousecam_frame_dict[widefield_frame] = closest_mousecam_frame
        #print("Widefield Frame: ", widefield_frame, " Closest Mousecam Frame: ", closest_mousecam_frame)

    # Save Directory
    save_directoy = os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy")
    np.save(save_directoy, widfield_to_mousecam_frame_dict)


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


def get_face_data(video_file, face_pixels):

    # Open Video File
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Extract Selected Frames
    face_data = []

    print("Extracting Face Video Data")
    for frame_index in tqdm(range(frameCount)):
    #while (frame_index < frameCount and ret):
        ret, frame = cap.read()
        frame = frame[:, :, 0]

        face_frame = []
        for pixel in face_pixels:
            face_frame.append(frame[pixel[0], pixel[1]])

        face_data.append(face_frame)
        frame_index += 1

    cap.release()
    face_data = np.array(face_data)
    return face_data, frameHeight, frameWidth


def get_face_eigenspectrum(face_data, n_components=150):

    model = PCA(n_components=n_components)
    model.fit(face_data)
    explained_variance_ratio = model.explained_variance_ratio_

    cumulative_variance_explained_list = []

    for x in range(n_components-1):
        cumulative_variance_explained = np.sum(explained_variance_ratio[0:x])
        cumulative_variance_explained_list.append(cumulative_variance_explained)

    print("Explained Variance Ratio", explained_variance_ratio)
    plt.plot(cumulative_variance_explained_list)
    plt.show()

    return explained_variance_ratio


def decompose_face_data(face_data, n_components=150):
    model = PCA(n_components=n_components)
    transformed_data = model.fit_transform(face_data)
    components = model.components_
    return transformed_data, components


def view_face_motion_components(base_directory, components, face_pixels, image_height, image_width):

    number_of_face_pixels = np.shape(face_pixels)[0]
    face_y_min = np.min(face_pixels[:, 0])
    face_y_max = np.max(face_pixels[:, 0])
    face_x_min = np.min(face_pixels[:, 1])
    face_x_max = np.max(face_pixels[:, 1])


    figure_1 = plt.figure(figsize=(15, 10))

    count = 1
    for component in components:

        template = np.zeros((image_height, image_width))

        for face_pixel_index in range(number_of_face_pixels):
            pixel_data = component[face_pixel_index]
            pixel_position = face_pixels[face_pixel_index]
            template[pixel_position[0], pixel_position[1]] = pixel_data

        template = template[face_y_min:face_y_max, face_x_min:face_x_max]
        template_magnitude = np.max(np.abs(template))

        axis = figure_1.add_subplot(10, 15, count)
        #axis.set_title(count)
        axis.axis('off')
        axis.imshow(template, vmax=template_magnitude, vmin=-template_magnitude, cmap='bwr')

        count += 1

    plt.savefig(os.path.join(base_directory, "Mousecam_analysis", "Face_Motion_Components.png"))
    plt.close()





def match_face_motion_to_widefield_frames(base_directory, transformed_data):

    # Load Widefield Frame Dict
    widefield_frame_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy"), allow_pickle=True)[()]

    print("Widefield Frame Dict Keys", list(widefield_frame_dict.keys())[0:1000])
    print("Widefield Frame Dict Values", list(widefield_frame_dict.values())[0:1000])

    # Visualise This
    """
    ai_data = Regression_Utils.load_ai_recorder_file(base_directory)
    stimuli_dictionary = Regression_Utils.create_stimuli_dictionary()
    blue_led_trace = ai_data[stimuli_dictionary["LED 1"]]
    mousecam_trace = ai_data[stimuli_dictionary["Mousecam"]]

    mousecam_frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Mousecam_Frame_Times.npy"), allow_pickle=True)[()]
    mousecam_frame_times = list(mousecam_frame_times.keys())
    print("mousecam frame times", mousecam_frame_times[0:100])

    plt.plot(blue_led_trace, c='b')
    plt.plot(mousecam_trace, c='m')
    plt.scatter(mousecam_frame_times, np.ones(len(mousecam_frame_times)))
    plt.show()
    """

    widefield_frame_matched_motion = []
    for widefield_frame in widefield_frame_dict.keys():
        mousecam_frame = widefield_frame_dict[widefield_frame]
        widefield_frame_matched_motion.append(transformed_data[mousecam_frame])

    widefield_frame_matched_motion = np.array(widefield_frame_matched_motion)
    return widefield_frame_matched_motion




def view_face_motion_differences(base_directory, face_pixels, face_motion_data):

    # Get Video Name
    video_name = get_video_name(base_directory)

    # Get Video Details
    frameCount, image_height, image_width = get_video_details(base_directory, video_name)
    number_of_face_pixels = np.shape(face_pixels)[0]
    face_y_min = np.min(face_pixels[:, 0])
    face_y_max = np.max(face_pixels[:, 0])
    face_x_min = np.min(face_pixels[:, 1])
    face_x_max = np.max(face_pixels[:, 1])

    for frame_index in range(1, frameCount):

        frame_1 = face_motion_data[frame_index]
        template_1 = np.zeros((image_height, image_width))
        for face_pixel_index in range(number_of_face_pixels):
            pixel_data = frame_1[face_pixel_index]
            pixel_position = face_pixels[face_pixel_index]
            template_1[pixel_position[0], pixel_position[1]] = pixel_data

        frame_2 = face_motion_data[frame_index-1]
        template_2 = np.zeros((image_height, image_width))
        for face_pixel_index in range(number_of_face_pixels):
            pixel_data = frame_2[face_pixel_index]
            pixel_position = face_pixels[face_pixel_index]
            template_2[pixel_position[0], pixel_position[1]] = pixel_data

        template_1 = template_1[face_y_min:face_y_max, face_x_min:face_x_max]
        template_2 = template_2[face_y_min:face_y_max, face_x_min:face_x_max]
        motion_energy = np.abs(np.subtract(template_2, template_1))

        figure_1 = plt.figure()
        rows = 1
        columns = 3
        frame_1_axis = figure_1.add_subplot(rows, columns, 1)
        frame_2_axis = figure_1.add_subplot(rows, columns, 2)
        motion_energy_axis = figure_1.add_subplot(rows, columns, 3)

        frame_1_axis.imshow(template_1, cmap='inferno', vmin=0, vmax=255)
        frame_2_axis.imshow(template_2, cmap='inferno', vmin=0, vmax=255)
        motion_energy_axis.imshow(motion_energy, cmap='inferno', vmin=0, vmax=50)


        plt.show()


def view_face_motion(base_directory, face_pixels, face_motion_data):

    # Get Video Name
    video_name = get_video_name(base_directory)

    # Get Video Details
    frameCount, image_height, image_width = get_video_details(base_directory, video_name)
    number_of_face_pixels = np.shape(face_pixels)[0]
    face_y_min = np.min(face_pixels[:, 0])
    face_y_max = np.max(face_pixels[:, 0])
    face_x_min = np.min(face_pixels[:, 1])
    face_x_max = np.max(face_pixels[:, 1])

    motion_magnitude = np.percentile(np.abs(face_motion_data), q=99)

    plt.ion()
    count =0
    for frame in face_motion_data:
        template = np.zeros((image_height, image_width))
        for face_pixel_index in range(number_of_face_pixels):
            pixel_data = frame[face_pixel_index]
            pixel_position = face_pixels[face_pixel_index]
            template[pixel_position[0], pixel_position[1]] = pixel_data

        template = template[face_y_min:face_y_max, face_x_min:face_x_max]

        plt.imshow(template, cmap='inferno', vmin=0, vmax=motion_magnitude)
        plt.draw()
        plt.title(str(count))
        plt.pause(0.1)
        plt.clf()

        count += 1


def extract_face_motion(base_directory):

    # Match Mousecam To Widefield frames
    Match_Mousecam_Frames_To_Widefield_Frames.match_mousecam_to_widefield_frames(base_directory)

    # Load Facepoly
    face_pixels = np.load(os.path.join(base_directory, "Mousecam_analysis", "Whisker_Pixels.npy"))
    face_pixels = np.transpose(face_pixels)

    # Get Video Name
    video_name = get_video_name(base_directory)

    # Get Face Data
    face_data, image_height, image_width = get_face_data(os.path.join(base_directory, video_name), face_pixels)

    # Get Face Motion
    face_data = np.ndarray.astype(face_data, int) # Must be converted to signed int or negative values with become extremely high positive values
    face_data = np.diff(face_data, axis=0)
    face_data = np.abs(face_data)

    # Match Mousecam Motion To Widefield Frames
    widefield_frame_matched_motion = match_face_motion_to_widefield_frames(base_directory, face_data)

    # Save This
    np.save(os.path.join(base_directory, "Mousecam_analysis", "Face_Motion.npy"), widefield_frame_matched_motion)


def convert_face_pixels_to_indicies(face_pixel_list):

    index_map = np.zeros(480*640)
    index_map[:] = list(range(480*640))

    face_pixel_index_list = []
    index_map = np.reshape(index_map, (480, 640))
    for face_pixel in face_pixel_list:
        face_pixel_index = int(index_map[face_pixel[0], face_pixel[1]])
        face_pixel_index_list.append(face_pixel_index)

    return face_pixel_index_list


def extract_face_data_from_motion_data(base_directory):
    print("Session", base_directory, " at ", datetime.now())

    # Load Facepoly
    face_pixels = np.load(os.path.join(base_directory, "Mousecam_analysis", "Whisker_Pixels.npy"))
    face_pixels = np.transpose(face_pixels)
    face_pixel_indicies = convert_face_pixels_to_indicies(face_pixels)
    print("Face Pixel Index Shape", len(face_pixel_indicies))

    # Load Motion Energy
    motion_energy_file = os.path.join(base_directory, "Mousecam_Analysis", "Bodycam_Motion_Energy.h5")
    motion_energy_container = tables.open_file(motion_energy_file, "r")
    motion_energy = motion_energy_container.root["blue"]

    # Get Face Motion Energy
    face_motion_energy = motion_energy[:, face_pixel_indicies]
    np.save(os.path.join(base_directory, "Mousecam_Analysis", "Face_Motion_Energy.npy"), face_motion_energy)

    # Close FIle
    motion_energy_container.close()


session_list = [
        #r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_23_Transition_Imaging",
        #r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_31_Transition_Imaging",
        #r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_04_02_Transition_Imaging",

        r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_02_Transition_Imaging",
        #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_08_Transition_Imaging",
        r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_10_Transition_Imaging",

        r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_13_Transition_Imaging",
        r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_15_Transition_Imaging",
        r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_17_Transition_Imaging",

        #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_10_29_Transition_Imaging",
        #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_11_03_Transition_Imaging",
        #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_11_05_Transition_Imaging"

    ]

for session in session_list:
    extract_face_data_from_motion_data(session)