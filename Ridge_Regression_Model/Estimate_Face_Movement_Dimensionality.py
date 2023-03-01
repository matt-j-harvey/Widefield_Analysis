import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD

def get_face_data(video_file, face_pixels):

    # Open Video File
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Extract Selected Frames
    face_data = []
    for frame_index in range(frameCount):
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


def get_bodycam_filename(base_directory):
    file_list = os.listdir(base_directory)
    for file_name in file_list:
        if "_cam_1.mp4" in file_name:
            return file_name


def view_whisker_activity(whisker_data, whisker_pixels, frame_height, frame_width):

    number_of_frames = len(whisker_data)
    number_of_whisker_pixels = np.shape(whisker_pixels)[0]

    whisker_y_min = np.min(whisker_pixels[:, 0])
    whisker_y_max = np.max(whisker_pixels[:, 0])
    whisker_x_min = np.min(whisker_pixels[:, 1])
    whisker_x_max = np.max(whisker_pixels[:, 1])

    plt.ion()
    for frame_index in range(number_of_frames):
        template = np.zeros((frame_height, frame_width))
        for pixel_index in range(number_of_whisker_pixels):
            pixel_value = whisker_data[frame_index, pixel_index]
            template[whisker_pixels[pixel_index, 0], whisker_pixels[pixel_index, 1]] = pixel_value

        plt.imshow(template[whisker_y_min:whisker_y_max, whisker_x_min:whisker_x_max], vmin=0, vmax=50)
        plt.draw()
        plt.pause(0.1)
        plt.clf()

def match_whisker_motion_to_widefield_motion(base_directory, transformed_whisker_data):

    print("Matching")

    # Load Widefield To Mousecam Frame Dict
    widefield_to_mousecam_frame_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy"), allow_pickle=True)[()]
    widefield_frame_list = list(widefield_to_mousecam_frame_dict.keys())
    number_of_mousecam_frames = np.shape(transformed_whisker_data)[0]

    print("Widefield Frames", len(widefield_frame_list))
    print("Mousecam Frames", number_of_mousecam_frames)
    print("Minimum Matched Mousecam Frame", np.min(list(widefield_to_mousecam_frame_dict.values())))
    print("Maximum Matched Mousecam Frame", np.max(list(widefield_to_mousecam_frame_dict.values())))
    print("Transformed Whisker Data Shape", np.shape(transformed_whisker_data))

    # Match Whisker Activity To Widefield Frames
    matched_whisker_data = []
    for widefield_frame in widefield_frame_list:
        corresponding_mousecam_frame = widefield_to_mousecam_frame_dict[widefield_frame]
        if corresponding_mousecam_frame < number_of_mousecam_frames:
            matched_whisker_data.append(transformed_whisker_data[corresponding_mousecam_frame])
        else:
            print("unmatched, mousecam frame: ", corresponding_mousecam_frame)
    matched_whisker_data = np.array(matched_whisker_data)
    return matched_whisker_data


def plot_cumulative_explained_variance(explained_variance, save_directory):
    cumulative_variance = np.cumsum(explained_variance)
    x_values = list(range(1, len(cumulative_variance)+1))
    plt.title("Cumulative Explained Variance, Face Movement PCA")
    plt.plot(x_values, cumulative_variance)
    plt.ylim([0, 1.1])
    plt.savefig(os.path.join(save_directory, "Face_Cumulative_Explained_Variance.png"))
    plt.close()


def decompose_face_motion(base_directory):

    # Load Whisker Pixels
    face_pixels = np.load(os.path.join(base_directory, "Mousecam_Analysis", "Face_Pixels.npy"))
    face_pixels = np.transpose(face_pixels)

    # Get Bodycam Filename
    bodycam_filename = get_bodycam_filename(base_directory)
    bodycam_file = os.path.join(base_directory, bodycam_filename)

    # Get Whisker Data
    face_data, frame_height, frame_width = get_face_data(bodycam_file, face_pixels)
    face_data = np.ndarray.astype(face_data, float)

    # Get Whisker Motion Energy
    face_motion_energy = np.diff(face_data, axis=0)
    face_motion_energy = np.abs(face_motion_energy)

    # Peform SVD on this
    model = TruncatedSVD(n_components=500)
    model.fit(face_motion_energy)

    # Get Explained Variance
    explained_variance = model.explained_variance_ratio_

    # Save This
    np.save(os.path.join(base_directory, "Mousecam_Analysis", "Explained_Variance_200.npy"), explained_variance)
    return explained_variance



def calculate_dimensionality(session_list):

    # Create Figure
    figure_1 = plt.figure()
    individual_axis = figure_1.add_subplot(1,2,1)
    cumulative_axis = figure_1.add_subplot(1,2,2)

    for base_directory in tqdm(session_list):
        explained_variance = decompose_face_motion(os.path.join("/media/matthew/Expansion/Control_Data", base_directory))

        individual_axis.plot(explained_variance)
        cumulative_axis.plot(np.cumsum(explained_variance))

    plt.show()


def estimate_average_dimensionality(session_list):

    dimensionality_list = []
    for base_directory in tqdm(session_list):
        explained_variance = np.load(os.path.join("/media/matthew/Expansion/Control_Data",base_directory, "Mousecam_Analysis", "Explained_Variance_200.npy"))
        dimensionality_list.append(explained_variance)

    dimensionality_list = np.array(dimensionality_list)
    dimensionality_list = np.cumsum(dimensionality_list, axis=1)
    mean_dimensionality = np.mean(dimensionality_list, axis=0)
    dimensionality_sd = np.std(mean_dimensionality, axis=0)


    eighty_counter = 1
    for x in range(500):
        if mean_dimensionality[x] > 0.8:
            print("80% Of Variance Explained By: ", eighty_counter, "Components")
        else:
            eighty_counter += 1

    ninety_counter = 1
    for x in range(500):
        if mean_dimensionality[x] > 0.9:
            print("90% Of Variance Explained By: ", ninety_counter, "Components")
        else:
            ninety_counter += 1

    print("next 100 components", mean_dimensionality[eighty_counter-1 + 100])

    x_values = list(range(len(mean_dimensionality)))
    x_values = np.add(x_values, 1)
    plt.plot(x_values, mean_dimensionality)

    sd_upper_bound = np.add(mean_dimensionality, dimensionality_sd)
    sd_lower_bound = np.subtract(mean_dimensionality, dimensionality_sd)
    plt.fill_between(x_values, sd_lower_bound, sd_upper_bound, alpha=0.1)

    plt.show()


# Select Sessions
selected_session_list = [

    r"NRXN78.1A/2020_12_05_Switching_Imaging",
    r"NRXN78.1A/2020_12_09_Switching_Imaging",

    r"NRXN78.1D/2020_11_29_Switching_Imaging",
    r"NRXN78.1D/2020_12_05_Switching_Imaging",

    r"NXAK14.1A/2021_05_21_Switching_Imaging",
    r"NXAK14.1A/2021_05_23_Switching_Imaging",
    r"NXAK14.1A/2021_06_11_Switching_Imaging",
    r"NXAK14.1A/2021_06_13_Transition_Imaging",
    r"NXAK14.1A/2021_06_15_Transition_Imaging",
    r"NXAK14.1A/2021_06_17_Transition_Imaging",

    r"NXAK22.1A/2021_10_14_Switching_Imaging",
    r"NXAK22.1A/2021_10_20_Switching_Imaging",
    r"NXAK22.1A/2021_10_22_Switching_Imaging",
    r"NXAK22.1A/2021_10_29_Transition_Imaging",
    r"NXAK22.1A/2021_11_03_Transition_Imaging",
    r"NXAK22.1A/2021_11_05_Transition_Imaging",

    r"NXAK4.1B/2021_03_02_Switching_Imaging",
    r"NXAK4.1B/2021_03_04_Switching_Imaging",
    r"NXAK4.1B/2021_03_06_Switching_Imaging",
    r"NXAK4.1B/2021_04_02_Transition_Imaging",
    r"NXAK4.1B/2021_04_08_Transition_Imaging",
    r"NXAK4.1B/2021_04_10_Transition_Imaging",

    r"NXAK7.1B/2021_02_26_Switching_Imaging",
    r"NXAK7.1B/2021_02_28_Switching_Imaging",
    r"NXAK7.1B/2021_03_02_Switching_Imaging",
    r"NXAK7.1B/2021_03_23_Transition_Imaging",
    r"NXAK7.1B/2021_03_31_Transition_Imaging",
    r"NXAK7.1B/2021_04_02_Transition_Imaging",

]

#calculate_dimensionality(selected_session_list)
estimate_average_dimensionality(selected_session_list)
