import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from matplotlib.gridspec import GridSpec

from Widefield_Utils import widefield_utils
from Files import Session_List


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


def view_bodycam_and_led_frames(base_directory):

    # Load Downsampled AI
    downsampled_ai = np.load(os.path.join(base_directory, "Downsampled_AI_Matrix_Framewise.npy"))

    stimuli_dict = widefield_utils.create_stimuli_dictionary()

    mousecam_trace = downsampled_ai[stimuli_dict["Mousecam"]]

    plt.plot(mousecam_trace)
    plt.show()


def view_face_components(base_directory):

    image_width = 640
    image_height = 480

    # Get Mousecam Directory
    mousecam_directory = os.path.join(base_directory, "Mousecam_Analysis")

    # Get Mousecam Components
    mousecam_components = np.load(os.path.join(mousecam_directory, "face_motion_components.npy"))
    print("Mousecam Components", np.shape(mousecam_components))
    number_of_components = np.shape(mousecam_components)[0]

    # Load Face Pixels
    face_pixels = np.load(os.path.join(mousecam_directory, "Face_Pixels.npy"))
    print("Face Pixel Shape", np.shape(face_pixels))
    face_pixels = np.transpose(face_pixels)

    # Get Face Pixel Extent
    face_y_min = np.min(face_pixels[:, 0])
    face_y_max = np.max(face_pixels[:, 0])
    face_x_min = np.min(face_pixels[:, 1])
    face_x_max = np.max(face_pixels[:, 1])

    # Load Matched Mousecam Data
    mousecam_data = np.load(os.path.join(mousecam_directory, "matched_face_data.npy"))
    print("mousecam data shape", np.shape(mousecam_data))

    # Create Figure
    figure_1 = plt.figure(figsize=(10, 80))
    n_rows = number_of_components
    n_cols = 4
    gridspec_1 = GridSpec(ncols=n_cols, nrows=n_rows, figure=figure_1)

    for component_index in range(number_of_components):

        # Create Axis
        component_axis = figure_1.add_subplot(gridspec_1[component_index, 0])
        time_loading_axis = figure_1.add_subplot(gridspec_1[component_index, 1:4])

        # Create Component Image
        component = mousecam_components[component_index]
        component_magnitude = np.max(np.abs(component))
        component_image = place_roi_into_mousecam(face_pixels, image_height, image_width, component)
        component_image = component_image[face_y_min:face_y_max, face_x_min:face_x_max]

        # Plot Component Image
        component_axis.set_title("Component " + str(component_index+1).zfill(3))
        component_axis.axis('off')
        component_axis.imshow(component_image, cmap='seismic', vmin=-component_magnitude, vmax=component_magnitude)

        # Plot Time Trace
        time_loading_axis.plot(mousecam_data[:, component_index])

    plt.savefig(os.path.join(mousecam_directory, "Mousecam_Components.png"))
    plt.close()


def place_roi_into_mousecam(roi_pixels, image_height, image_width, vector):
    
    number_pixels = np.shape(roi_pixels)[0]
    template = np.zeros((image_height, image_width))

    for pixel_index in range(number_pixels):
        pixel_value = vector[pixel_index]
        pixel_x = roi_pixels[pixel_index, 1]
        pixel_y = roi_pixels[pixel_index, 0]
        template[pixel_y, pixel_x] = pixel_value

    return template



def view_whisker_face_movie(mousecam_directory):

    # Load Face Pixels
    face_pixels = np.load(os.path.join(mousecam_directory, "Face_Pixels.npy"))



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


def extract_face_motion(base_directory):

    # Get Save Directory
    save_directory = os.path.join(base_directory, "Mousecam_Analysis")

    # Load Whisker Pixels
    face_pixels = np.load(os.path.join(base_directory, "Mousecam_Analysis", "Face_Pixels.npy"))
    face_pixels = np.transpose(face_pixels)

    # Get Bodycam Filename
    bodycam_filename = get_bodycam_filename(base_directory)
    bodycam_file = os.path.join(base_directory, bodycam_filename)

    # Get Whisker Data
    face_data, frame_height, frame_width = get_face_data(bodycam_file, face_pixels)
    face_data = np.ndarray.astype(face_data, float)
    print("Face Data Shape", np.shape(face_data))

    # Get Whisker Motion Energy
    face_motion_energy = np.diff(face_data, axis=0)
    face_motion_energy = np.abs(face_motion_energy)
    print("Motion Energy Shape", np.shape(face_motion_energy))

    # Peform SVD on this
    model = TruncatedSVD(n_components=20)
    transformed_data = model.fit_transform(face_motion_energy)
    print("Transformed Data Shape", np.shape(transformed_data))

    # Get Explained Variance
    explained_variance = model.explained_variance_ratio_
    plot_cumulative_explained_variance(explained_variance, save_directory)

    # Get Components
    face_components = model.components_

    # Match This TO Widefield Frames
    matched_face_data = match_whisker_motion_to_widefield_motion(base_directory, transformed_data)
    print("Matched Face Data Shape", np.shape(matched_face_data))

    # Save This
    np.save(os.path.join(save_directory, "matched_face_data.npy"), matched_face_data)
    np.save(os.path.join(save_directory, "face_explained_variance_ratio.npy"), explained_variance)
    np.save(os.path.join(save_directory, "face_motion_components.npy"), face_components)



def get_video_height_width(video_filepath):
    video_capture = cv2.VideoCapture(video_filepath)

    image_height = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    image_width = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_capture.release()
    return image_height, image_width


selected_session_list = Session_List.control_session_tuples
selected_session_list = [["/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_29_Discrimination_Imaging"]]

session_list = [
    r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_14_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_21_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_14_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_23_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_01_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_22_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_04_29_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_07_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_25_Discrimination_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_07_Discrimination_Imaging"
]

"""
session_list = [

    "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_14_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_09_Discrimination_Imaging",

    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_04_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging",

    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_02_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_14_Discrimination_Imaging",

    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_02_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_15_Discrimination_Imaging",

    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_30_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging",

    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_22_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_08_Discrimination_Imaging",
]
"""

session_list = ["/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_29_Discrimination_Imaging"]

session_list = [
    r"NRXN78.1A/2020_11_28_Switching_Imaging",
    r"NRXN78.1A/2020_12_05_Switching_Imaging",
    r"NRXN78.1A/2020_12_09_Switching_Imaging",

    r"NRXN78.1D/2020_12_07_Switching_Imaging",
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

    r"NRXN78.1A/2020_11_14_Discrimination_Imaging",
    r"NRXN78.1A/2020_11_15_Discrimination_Imaging",
    r"NRXN78.1A/2020_11_24_Discrimination_Imaging",
    r"NRXN78.1A/2020_11_21_Discrimination_Imaging",

    r"NRXN78.1D/2020_11_14_Discrimination_Imaging",
    r"NRXN78.1D/2020_11_15_Discrimination_Imaging",
    r"NRXN78.1D/2020_11_25_Discrimination_Imaging",
    r"NRXN78.1D/2020_11_23_Discrimination_Imaging",

    r"NXAK4.1B/2021_02_04_Discrimination_Imaging",
    r"NXAK4.1B/2021_02_06_Discrimination_Imaging",
    r"NXAK4.1B/2021_02_22_Discrimination_Imaging",
    r"NXAK4.1B/2021_02_14_Discrimination_Imaging",

    r"NXAK7.1B/2021_02_01_Discrimination_Imaging",
    r"NXAK7.1B/2021_02_03_Discrimination_Imaging",
    r"NXAK7.1B/2021_02_24_Discrimination_Imaging",
    r"NXAK7.1B/2021_02_22_Discrimination_Imaging",

    r"NXAK14.1A/2021_04_29_Discrimination_Imaging",
    r"NXAK14.1A/2021_05_01_Discrimination_Imaging",
    r"NXAK14.1A/2021_05_09_Discrimination_Imaging",
    r"NXAK14.1A/2021_05_07_Discrimination_Imaging",

    r"NXAK22.1A/2021_09_25_Discrimination_Imaging",
    r"NXAK22.1A/2021_09_29_Discrimination_Imaging",
    r"NXAK22.1A/2021_10_08_Discrimination_Imaging",
    r"NXAK22.1A/2021_10_07_Discrimination_Imaging",

]

full_session_list = []
for item in session_list:
    full_session_list.append(os.path.join("/media/matthew/Expansion/Control_Data", item))


full_session_list = [

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_14_Discrimination_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_09_Discrimination_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_04_Discrimination_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_02_Discrimination_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_14_Discrimination_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_02_Discrimination_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_15_Discrimination_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_30_Discrimination_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging",

        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_22_Discrimination_Imaging",
        r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_08_Discrimination_Imaging",
    ]

for base_directory in tqdm(full_session_list):
    print("Base directory", base_directory)
    extract_face_motion(base_directory)
    #view_face_components(base_directory)


