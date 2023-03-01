import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD

def get_whisker_data(video_file, face_pixels):

    # Open Video File
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Extract Selected Frames
    face_data = []

    print("Extracting Whisker Video Data")
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

    # Load Widefield To Mousecam Frame Dict
    widefield_to_mousecam_frame_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy"), allow_pickle=True)[()]
    widefield_frame_list = list(widefield_to_mousecam_frame_dict.keys())

    number_of_mousecam_frames = np.shape(transformed_whisker_data)[0]

    # Match Whisker Activity To Widefield Frames
    matched_whisker_data = []
    for widefield_frame in widefield_frame_list:
        corresponding_mousecam_frame = widefield_to_mousecam_frame_dict[widefield_frame]

        if corresponding_mousecam_frame < number_of_mousecam_frames:
            matched_whisker_data.append(transformed_whisker_data[corresponding_mousecam_frame])

    matched_whisker_data = np.array(matched_whisker_data)
    return matched_whisker_data




def plot_cumulative_explained_variance(explained_variance, save_directory):
    cumulative_variance = np.cumsum(explained_variance)
    x_values = list(range(1, len(cumulative_variance)+1))
    plt.title("Cumulative Explained Variance, Whisker Movement PCA")
    plt.plot(x_values, cumulative_variance)
    plt.ylim([0, 1.1])
    plt.savefig(os.path.join(save_directory, "Whisker_Cumulative_Explained_Variance.png"))
    plt.close()


def extract_whisks(base_directory):

    # Get Save Directory
    save_directory = os.path.join(base_directory, "Mousecam_Analysis")

    # Load Whisker Pixels
    whisker_pixels = np.load(os.path.join(save_directory, "Whisker_Pixels.npy"))
    whisker_pixels = np.transpose(whisker_pixels)
    print("Whisker Pixel Shape", np.shape(whisker_pixels))

    # Get Bodycam Filename
    bodycam_filename = get_bodycam_filename(base_directory)
    bodycam_file = os.path.join(base_directory, bodycam_filename)

    # Get Whisker Data
    whisker_data, frame_height, frame_width = get_whisker_data(bodycam_file, whisker_pixels)
    print("Whisker Data Shape", np.shape(whisker_data))
    whisker_data = np.ndarray.astype(whisker_data, float)

    # Get Whisker Motion Energy
    whisker_motion_energy = np.diff(whisker_data, axis=0)
    whisker_motion_energy = np.abs(whisker_motion_energy)

    #view_whisker_activity(whisker_motion_energy, whisker_pixels, frame_height, frame_width)

    # Peform SVD on this
    model = TruncatedSVD(n_components=20)
    transformed_data = model.fit_transform(whisker_motion_energy)

    # Match This TO Widefield Frames
    matched_whisker_data = match_whisker_motion_to_widefield_motion(base_directory, transformed_data)

    # Save This
    np.save(os.path.join(base_directory, "Mousecam_Analysis", "matched_whisker_data.npy"), matched_whisker_data)

    # Calculate Eigenspectrum
    explained_variance_ratio = model.explained_variance_ratio_
    np.save(os.path.join(base_directory, "Mousecam_Analysis", "whisker_explained_variance_ratio.npy"), explained_variance_ratio)

    # Plot Eigenspetrum
    plot_cumulative_explained_variance(explained_variance_ratio, save_directory)

"""
session_list = [

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_13_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_15_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_17_Switching_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_31_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_02_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_04_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_08_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_10_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_12_Transition_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_20_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_22_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_24_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_14_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_16_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_18_Transition_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_17_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_19_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_23_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_30_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_06_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_08_Transition_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_15_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_17_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_19_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_22_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_24_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_26_Transition_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_14_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_20_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_26_Switching_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_05_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_08_Transition_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_10_Transition_Imaging",

        r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_28_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_05_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_09_Switching_Imaging",

        r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_29_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_05_Switching_Imaging",

        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_02_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_04_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_06_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_02_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_08_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_10_Transition_Imaging",

        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_26_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_28_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_02_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_23_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_31_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_04_02_Transition_Imaging",

        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_21_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_23_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_11_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_13_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_15_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_17_Transition_Imaging",

        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_14_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_20_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_22_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_29_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_03_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_05_Transition_Imaging",
    ]

session_list = ["/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Switching_Opto/KPGC2.2G/2022_12_02_Switching_Opto"]

for base_directory in tqdm(session_list):

    if not os.path.exists(os.path.join(base_directory, "Mousecam_Analysis", "matched_whisker_data.npy")):
        print(base_directory)
        extract_whisks(base_directory)
"""