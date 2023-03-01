import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from matplotlib.gridspec import GridSpec

from Widefield_Utils import widefield_utils



def place_roi_into_mousecam(roi_pixels, image_height, image_width, vector):

    number_pixels = np.shape(roi_pixels)[0]
    template = np.zeros((image_height, image_width))

    for pixel_index in range(number_pixels):
        pixel_value = vector[pixel_index]
        pixel_x = roi_pixels[pixel_index, 1]
        pixel_y = roi_pixels[pixel_index, 0]
        template[pixel_y, pixel_x] = pixel_value

    return template



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



session_list = [

        r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_28_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_05_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_09_Switching_Imaging",

        r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_29_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_05_Switching_Imaging",

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

        r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_14_Discrimination_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_24_Discrimination_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_21_Discrimination_Imaging",

        r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_14_Discrimination_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_15_Discrimination_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_25_Discrimination_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_23_Discrimination_Imaging",

        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_04_Discrimination_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_06_Discrimination_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_22_Discrimination_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_14_Discrimination_Imaging",

        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_01_Discrimination_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_03_Discrimination_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_24_Discrimination_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_22_Discrimination_Imaging",

        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_04_29_Discrimination_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_01_Discrimination_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_09_Discrimination_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_07_Discrimination_Imaging",

        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_25_Discrimination_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_29_Discrimination_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_07_Discrimination_Imaging",

    ]
for session in session_list:
    view_face_components(session)

