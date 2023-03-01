import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from matplotlib.gridspec import GridSpec
import mvlearn

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
    for frame_index in tqdm(range(frameCount), desc="Getting Face Energy"):
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


def face_data_cca(base_directory):

    # Get Save Directory
    save_directory = os.path.join(base_directory, "Mousecam_Analysis")

    """
    # Load Whisker Pixels
    face_pixels = np.load(os.path.join(base_directory, "Mousecam_Analysis", "Face_Pixels.npy"))
    face_pixels = np.transpose(face_pixels)

    # Get Bodycam Filename
    bodycam_filename = get_bodycam_filename(base_directory)
    bodycam_file = os.path.join(base_directory, bodycam_filename)

    # Get Face Data
    face_data, frame_height, frame_width = get_face_data(bodycam_file, face_pixels)
    face_data = np.ndarray.astype(face_data, float)

    # Get Face Motion Energy
    face_motion_energy = np.diff(face_data, axis=0)
    face_motion_energy = np.abs(face_motion_energy)
    print("Face Motion Energy", np.shape(face_motion_energy))

    # Match This TO Widefield Frames
    matched_face_data = match_whisker_motion_to_widefield_motion(base_directory, face_motion_energy)
    np.save(os.path.join(save_directory, "Matched_Face_Motion_energy.npy"), matched_face_data)
    """

    face_motion_energy = np.load(os.path.join(save_directory, "Matched_Face_Motion_energy.npy"))
    face_motion_energy = face_motion_energy[1:]
    print("Matched Face Data Shape", np.shape(face_motion_energy))

    # load Neural Data
    delta_f = np.load(os.path.join(base_directory, "Delta_F_Matrix_100_by_100_SVD.npy"))
    print("Delta F Shape", np.shape(delta_f))

    # Create CCA Model
    cca_model = mvlearn.embed.CCA(n_components=20, regs=0.01)
    cca_model.fit([face_motion_energy, delta_f])

    loading = cca_model.loadings_
    print("Loadings", np.shape(loading))
    np.save(os.path.join(save_directory, "CCA_Loadings.npy"), loading)


def place_roi_into_mousecam(roi_pixels, image_height, image_width, vector):
    number_pixels = np.shape(roi_pixels)[0]
    template = np.zeros((image_height, image_width))

    for pixel_index in range(number_pixels):
        pixel_value = vector[pixel_index]
        pixel_x = roi_pixels[pixel_index, 1]
        pixel_y = roi_pixels[pixel_index, 0]
        template[pixel_y, pixel_x] = pixel_value

    return template


def view_loadings(base_directory):
    face_image_width = 640
    face_image_height = 480

    # Get Save Directory
    save_directory = os.path.join(base_directory, "Mousecam_Analysis")

    # Load Face Pixels
    face_pixels = np.load(os.path.join(save_directory, "Face_Pixels.npy"))
    face_pixels = np.transpose(face_pixels)

    cca_loadings = np.load(os.path.join(save_directory, "CCA_Loadings.npy"), allow_pickle=True)
    print("CCA Loadings", np.shape(cca_loadings))

    face_loadings = cca_loadings[0]
    brain_loadings = cca_loadings[1]

    n_components = np.shape(face_loadings)[1]

    # Get Face Pixel Extent
    face_y_min = np.min(face_pixels[:, 0])
    face_y_max = np.max(face_pixels[:, 0])
    face_x_min = np.min(face_pixels[:, 1])
    face_x_max = np.max(face_pixels[:, 1])


    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)


    for component_index in range(n_components):

        face_data = face_loadings[:, component_index]
        brain_data = brain_loadings[:, component_index]

        component_magnitude = np.max(np.abs(face_data))
        face_image = place_roi_into_mousecam(face_pixels, face_image_height, face_image_width, face_data)
        face_image = face_image[face_y_min:face_y_max, face_x_min:face_x_max]
        plt.imshow(face_image,  cmap='seismic', vmin=-component_magnitude, vmax=component_magnitude)
        plt.show()

        brain_image = widefield_utils.create_image_from_data(brain_data, indicies, image_height, image_width)
        plt.imshow(brain_image)
        plt.show()


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

for base_directory in tqdm(full_session_list[0:1]):

    #face_data_cca(base_directory)
    view_loadings(base_directory)

