import pandas as pd
import tables
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
from sklearn.decomposition import TruncatedSVD, IncrementalPCA
from scipy import ndimage
from tqdm import tqdm
import pickle
import pybresenham
import cv2


def get_deeplabcut_filename(base_directory):
    file_list = os.listdir(base_directory)
    for file_name in file_list:
        if "DLC_resnet101_Matt_Bodycam" in file_name:
            return file_name


def create_limb_portrait_video(base_directory):

    # Load Transformed Data
    transformed_data = np.load(os.path.join(base_directory, "Limb_Movements", "Transformed_Limb_Data.npy"))
    model = pickle.load(open(os.path.join(base_directory,  "Limb_Movements", "limb_portrait_model.sav"), 'rb'))

    # Create Video File
    limb_video_file = os.path.join(os.path.join(base_directory, "Limb_Movements", "Limb_Movement.avi"))
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(limb_video_file, video_codec, frameSize=(640, 480), fps=30)  # 0, 12

    # Create Colourmap
    colourmap = plt.cm.ScalarMappable(norm=None, cmap='viridis')
    colour_max = 0.01
    colour_min = 0
    colourmap.set_clim(vmin=colour_min, vmax=colour_max)

    sample_size = 1000
    reconstructed_sample = model.inverse_transform(transformed_data[0:sample_size])
    for frame_index in tqdm(range(sample_size)):
        frame_data = reconstructed_sample[frame_index]
        frame_data = np.reshape(frame_data, (480, 640))
        frame_data = colourmap.to_rgba(frame_data)
        frame_data = np.multiply(frame_data, 255)
        frame_data = np.ndarray.astype(frame_data, np.uint8)
        frame_data = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)
        video.write(frame_data)

    cv2.destroyAllWindows()
    video.release()




def extract_limb_positions(base_directory, deeplabcut_file_name, number_of_limbs):

    # Get Limb Positions
    deeplabcut_dataframe = pd.read_hdf(os.path.join(base_directory, deeplabcut_file_name))
    columns = deeplabcut_dataframe.columns

    # Extract Data
    x_coords_list = []
    y_coords_list = []
    probability_list = []
    for limb_index in range(number_of_limbs):
        limb_x_coords = deeplabcut_dataframe[columns[limb_index * 3 + 0]]
        limb_y_coords = deeplabcut_dataframe[columns[limb_index * 3 + 1]]
        limb_likelihood = deeplabcut_dataframe[columns[limb_index * 3 + 2]]

        x_coords_list.append(limb_x_coords)
        y_coords_list.append(limb_y_coords)
        probability_list.append(limb_likelihood)

    x_coords_list = np.array(x_coords_list)
    y_coords_list = np.array(y_coords_list)
    return x_coords_list, y_coords_list, probability_list



def get_frame_motion(number_of_limbs, frame_height, frame_width, timepoint_index, probability_list, x_coords_list, y_coords_list, probability_threshold):

    combined_limb_template = []
    for limb_index in range(number_of_limbs):

        # Create Template
        template = np.zeros((frame_height, frame_width))

        # Threshold Probability
        previous_likelihood = probability_list[limb_index][timepoint_index - 1]
        current_likelihood = probability_list[limb_index][timepoint_index]

        # If We are Confident About Limb Positions - Create Movement Map
        if previous_likelihood > probability_threshold and current_likelihood > probability_threshold:

            position_1_x = x_coords_list[limb_index][timepoint_index - 1]
            position_1_y = y_coords_list[limb_index][timepoint_index - 1]

            position_2_x = x_coords_list[limb_index][timepoint_index]
            position_2_y = y_coords_list[limb_index][timepoint_index]

            line_object = pybresenham.line(position_1_x, position_1_y, position_2_x, position_2_y)
            for coord in line_object:
                template[coord[1], coord[0]] = 1

            # Add Gaussian Filter
            template = ndimage.gaussian_filter(template, sigma=10)

        # Add To Template
        combined_limb_template.append(template)

    combined_limb_template = np.array(combined_limb_template)
    combined_limb_template = np.mean(combined_limb_template, axis=0)
    combined_limb_template = np.reshape(combined_limb_template, (frame_height * frame_width))
    return combined_limb_template


def get_limb_movement_portraits(limb_x_coords, limb_y_coords, probability_list, probability_threshold, number_of_limbs, frame_height=480, frame_width=640):

    # Get Data Structure
    number_of_timepoints = np.shape(limb_x_coords)[1]
    print("Number of timepoints", number_of_timepoints)

    # Create Model
    n_components = 20
    model = IncrementalPCA(n_components=n_components)
    chunk_size = 1000

    # Fit Model
    print("Fitting Model")
    current_data = []
    for timepoint_index in tqdm(range(1, number_of_timepoints)):
        frame_limb_motion = get_frame_motion(number_of_limbs, frame_height, frame_width, timepoint_index, probability_list, limb_x_coords, limb_y_coords, probability_threshold)
        current_data.append(frame_limb_motion)

        if timepoint_index % chunk_size == 0:
            model.partial_fit(current_data)
            current_data = []

    if len(current_data) > n_components:
        model.partial_fit(current_data)

    # Transform Data
    print("Transformnig Data")
    transformed_data = []
    for timepoint_index in tqdm(range(1, number_of_timepoints)):
        frame_limb_motion = get_frame_motion(number_of_limbs, frame_height, frame_width, timepoint_index, probability_list, limb_x_coords, limb_y_coords, probability_threshold)
        transformed_frame = model.transform(frame_limb_motion.reshape(1, -1))
        transformed_data.append(transformed_frame)

    return transformed_data, model



def match_limb_movements_to_widefield_motion(base_directory, save_directory):

    # Load Transformed Limb Daa
    transformed_limb_data = np.load(os.path.join(save_directory, "Transformed_Limb_Data.npy"))

    # Load Widefield To Mousecam Frame Dict
    widefield_to_mousecam_frame_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy"), allow_pickle=True)[()]
    widefield_frame_list = list(widefield_to_mousecam_frame_dict.keys())

    # Match Whisker Activity To Widefield Frames
    matched_limb_data = []
    for widefield_frame in widefield_frame_list:
        corresponding_mousecam_frame = widefield_to_mousecam_frame_dict[widefield_frame]
        matched_limb_data.append(transformed_limb_data[corresponding_mousecam_frame])

    matched_limb_data = np.array(matched_limb_data)
    matched_limb_data = np.squeeze(matched_limb_data)

    return matched_limb_data



def extract_limb_movements(base_directory, probability_threshold=0.8, number_of_limbs=4):

    # Create Save Directory
    save_directory = os.path.join(base_directory, "Mousecam_Analysis")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Get Deeplabcut FIle Name
    deeplabcut_file_name = get_deeplabcut_filename(base_directory)
    print("Using Deeplabcut File: ", deeplabcut_file_name)

    # Get Limb Positions
   # limb_x_coords, limb_y_coords, probability_list = extract_limb_positions(base_directory, deeplabcut_file_name, number_of_limbs)

    # Get Movement Portraits
    #transformed_data, model = get_limb_movement_portraits(limb_x_coords, limb_y_coords, probability_list, probability_threshold, number_of_limbs)

    # Save Transformed Data and Model
    #np.save(os.path.join(save_directory, "Transformed_Limb_Data.npy"), transformed_data)
    #pickle.dump(model, open(os.path.join(save_directory, "limb_portrait_model.sav"), 'wb'))

    # Match Limb Movements To Widefield Frames
    matched_limb_movements = match_limb_movements_to_widefield_motion(base_directory, save_directory)
    print("Matched Limb Movement Shape", np.shape(matched_limb_movements))
    np.save(os.path.join(save_directory, "Matched_Limb_Movements.npy"), matched_limb_movements)

    # Create Limb Portrait Video
    #create_limb_portrait_video(base_directory)


base_directory = r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_17_Switching_Imaging"
#r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging"
extract_limb_movements(base_directory)