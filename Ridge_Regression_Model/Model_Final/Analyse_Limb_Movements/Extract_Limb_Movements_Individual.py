import pandas as pd
import tables
import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec
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
from skimage.transform import rescale


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



def get_frame_motion(number_of_limbs, frame_height, frame_width, timepoint_index, probability_list, x_coords_list, y_coords_list, probability_threshold, downscale_factor):

    limb_movement_portrait_list = []
    for limb_index in range(number_of_limbs):

        # Create Template
        template = np.zeros((int(frame_height/downscale_factor), int(frame_width/downscale_factor)))

        # Threshold Probability
        previous_likelihood = probability_list[limb_index][timepoint_index - 1]
        current_likelihood = probability_list[limb_index][timepoint_index]

        # If We are Confident About Limb Positions - Create Movement Map
        if previous_likelihood > probability_threshold and current_likelihood > probability_threshold:

            position_1_x = x_coords_list[limb_index][timepoint_index - 1]
            position_1_y = y_coords_list[limb_index][timepoint_index - 1]

            position_2_x = x_coords_list[limb_index][timepoint_index]
            position_2_y = y_coords_list[limb_index][timepoint_index]

            # Divide by Downscale Factor
            position_1_x = int(position_1_x/downscale_factor)
            position_1_y = int(position_1_y/downscale_factor)
            position_2_x = int(position_2_x/downscale_factor)
            position_2_y = int(position_2_y/downscale_factor)

            line_object = pybresenham.line(position_1_x, position_1_y, position_2_x, position_2_y)
            for coord in line_object:
                template[coord[1], coord[0]] = 1

            # Add Gaussian Filter
            template = ndimage.gaussian_filter(template, sigma=2)

        # Downsample
        #template = rescale(template, float(1)/downscale_factor, preserve_range=True)

        # Add To Template
        template = np.reshape(template, (int(frame_height/downscale_factor) * int(frame_width/downscale_factor)))
        limb_movement_portrait_list.append(template)

    limb_movement_portrait_list = np.array(limb_movement_portrait_list)
    return limb_movement_portrait_list


def get_limb_movement_portraits(limb_x_coords, limb_y_coords, probability_list, probability_threshold, number_of_limbs, downsample_factor, frame_height=480, frame_width=640):

    # Get Data Structure
    number_of_timepoints = np.shape(limb_x_coords)[1]
    print("Number of timepoints", number_of_timepoints)

    # Create Models
    n_components = 20
    model_list = []
    for model_index in range(number_of_limbs):
        model = IncrementalPCA(n_components=n_components)
        model_list.append(model)
    chunk_size = 1000

    # Fit Model
    print("Fitting Model")
    current_data = np.zeros((chunk_size, number_of_limbs, int(frame_height/downsample_factor) * int(frame_width/downsample_factor)))
    print("Current Data Shape", np.shape(current_data))

    chunk_position = 0
    for timepoint_index in tqdm(range(1, number_of_timepoints)):
        limb_movement_portrait_list = get_frame_motion(number_of_limbs, frame_height, frame_width, timepoint_index, probability_list, limb_x_coords, limb_y_coords, probability_threshold, downsample_factor)
        current_data[chunk_position] = limb_movement_portrait_list
        chunk_position += 1

        # Fit Chunk If We're Full
        if chunk_position == chunk_size:
            for limb_index in range(number_of_limbs):
                model_list[limb_index].partial_fit(current_data[:, limb_index])
            chunk_position = 0



    if chunk_position > n_components:
        for limb_index in range(number_of_limbs):
            model_list[limb_index].partial_fit(current_data[0:chunk_position, limb_index])


    # Transform Data
    print("Transformnig Data")
    current_data = np.zeros((chunk_size, number_of_limbs, int(frame_height/downsample_factor) * int(frame_width/downsample_factor)))
    transformed_data = [[],[],[],[]]
    chunk_position = 1
    for timepoint_index in tqdm(range(1, number_of_timepoints)):
        limb_movement_portrait_list = get_frame_motion(number_of_limbs, frame_height, frame_width, timepoint_index, probability_list, limb_x_coords, limb_y_coords, probability_threshold, downsample_factor)
        current_data[chunk_position] = limb_movement_portrait_list
        chunk_position += 1

        print("Chunk Position", chunk_position)
        # Fit Chunk If We're Full
        if chunk_position == chunk_size:
            print("Transforming")
            for limb_index in range(number_of_limbs):
                limb_transformed_data = model_list[limb_index].transform(current_data[:, limb_index])
                for frame in limb_transformed_data:
                    transformed_data[limb_index].append(frame)
            chunk_position = 0

    # Do Last Chunk
    if chunk_position > 0:
        print("Transforming Last: ", chunk_position)
        for limb_index in range(number_of_limbs):
            limb_transformed_data = model_list[limb_index].transform(current_data[:chunk_position, limb_index])
            for frame in limb_transformed_data:
                transformed_data[limb_index].append(frame)



    transformed_data = np.array(transformed_data)
    print("Transformed Data", np.shape(transformed_data))

    return transformed_data, model_list



def match_limb_movements_to_widefield_motion(base_directory, save_directory):

    # Load Transformed Limb Daa
    transformed_limb_data = np.load(os.path.join(save_directory, "Transformed_Limb_Data_Individual.npy"))
    print("MAtching transformed limb data shape", np.shape(transformed_limb_data))

    transformed_limb_data = np.hstack([
        transformed_limb_data[0],
        transformed_limb_data[1],
        transformed_limb_data[2],
        transformed_limb_data[3]
    ])

    print("MAtching transformed limb data shape", np.shape(transformed_limb_data))

    # Load Widefield To Mousecam Frame Dict
    widefield_to_mousecam_frame_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy"), allow_pickle=True)[()]
    widefield_frame_list = list(widefield_to_mousecam_frame_dict.keys())

    # Match Whisker Activity To Widefield Frames
    matched_limb_data = []
    for widefield_frame in widefield_frame_list:
        corresponding_mousecam_frame = widefield_to_mousecam_frame_dict[widefield_frame]
        matched_limb_data.append(transformed_limb_data[corresponding_mousecam_frame])

    matched_limb_data = np.array(matched_limb_data)
    print("matched_limb_data", np.shape(matched_limb_data))

    matched_limb_data = np.squeeze(matched_limb_data)
    print("matched_limb_data", np.shape(matched_limb_data))

    return matched_limb_data

def check_limb_movements_individual(base_directory):

    # Load Individual Limb Data
    mousecam_folder = os.path.join(base_directory, "Mousecam_Analysis")
    transformed_data = np.load(os.path.join(mousecam_folder, "Transformed_Limb_Data_Individual.npy"))
    number_of_limbs, number_of_frames, number_of_components = np.shape(transformed_data)
    print("Transformed Limb Data Shape", np.shape(transformed_data))

    # Load Limb Data
    reconstructed_data_list = []
    for limb_index in range(number_of_limbs):
        limb_model = pickle.load(open(os.path.join(mousecam_folder, "limb_portrait_model_Individual" + str(limb_index) + ".sav"), 'rb'))
        limb_data = transformed_data[limb_index]
        reconstructed_data = limb_model.inverse_transform(limb_data)
        reconstructed_data = np.reshape(reconstructed_data, (number_of_frames, 120, 160))
        reconstructed_data_list.append(reconstructed_data)
        print("reconstructed Data Shape", np.shape(reconstructed_data))

    # Plot This
    plt.ion()
    figure_1 = plt.figure()
    for timepoint_index in range(number_of_frames):
        figure_1.suptitle(str(timepoint_index))
        for limb_index in range(number_of_limbs):
            limb_axis = figure_1.add_subplot(2, 2, limb_index + 1)
            limb_axis.imshow(reconstructed_data_list[limb_index][timepoint_index], vmin=0, vmax=1)

        plt.draw()
        plt.pause(0.1)
        plt.clf()


def extract_limb_movements(base_directory, probability_threshold=0.8, number_of_limbs=4, downscale_factor=4):

    # Create Save Directory
    save_directory = os.path.join(base_directory, "Mousecam_Analysis")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Get Deeplabcut FIle Name
    deeplabcut_file_name = get_deeplabcut_filename(base_directory)
    print("Using Deeplabcut File: ", deeplabcut_file_name)

    # Get Limb Positions
    limb_x_coords, limb_y_coords, probability_list = extract_limb_positions(base_directory, deeplabcut_file_name, number_of_limbs)
    print("Limb x cords shape", np.shape(limb_x_coords))
    #limb_x_coords = limb_x_coords[:, 0:2000]
    #limb_y_coords = limb_y_coords[:, 0:2000]


    # Get Movement Portraits
    """
    transformed_data, model_list = get_limb_movement_portraits(limb_x_coords, limb_y_coords, probability_list, probability_threshold, number_of_limbs, downscale_factor)

    # Save Transformed Data and Model
    np.save(os.path.join(save_directory, "Transformed_Limb_Data_Individual.npy"), transformed_data)
    for limb_index in range(number_of_limbs):
        pickle.dump(model_list[limb_index], open(os.path.join(save_directory, "limb_portrait_model_Individual" + str(limb_index) + ".sav"), 'wb'))

    """

    # Match Limb Movements To Widefield Frames
    matched_limb_movements = match_limb_movements_to_widefield_motion(base_directory, save_directory)
    print("Matched Limb Movement Shape", np.shape(matched_limb_movements))
    np.save(os.path.join(save_directory, "Matched_Limb_Movements_Individual.npy"), matched_limb_movements)

    # Create Limb Portrait Video
    #create_limb_portrait_video(base_directory)


base_directory = r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_17_Switching_Imaging"
#r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging"
extract_limb_movements(base_directory)
#check_limb_movements_individual(base_directory)