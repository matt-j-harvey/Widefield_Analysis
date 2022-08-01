import numpy as np
import tables
import h5py
import numpy
import os
from scipy.ndimage import gaussian_filter
from datetime import datetime
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cv2

def create_image_from_data(data, indicies, image_height, image_width):
    template = np.zeros((image_height, image_width))
    data = np.nan_to_num(data)
    np.put(template, indicies, data)
    image = np.ndarray.reshape(template, (image_height, image_width))

    return image


def load_mask(home_directory):

    # Loads the mask for a video, returns a list of which pixels are included, as well as the original image height and width
    mask = np.load(os.path.join(home_directory, "Generous_Mask.npy"))

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask>0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width


def get_chunk_structure(chunk_size, array_size):
    number_of_chunks = int(np.ceil(array_size / chunk_size))
    remainder = array_size % chunk_size

    # Get Chunk Sizes
    chunk_sizes = []
    if remainder == 0:
        for x in range(number_of_chunks):
            chunk_sizes.append(chunk_size)

    else:
        for x in range(number_of_chunks - 1):
            chunk_sizes.append(chunk_size)
        chunk_sizes.append(remainder)

    # Get Chunk Starts
    chunk_starts = []
    chunk_start = 0
    for chunk_index in range(number_of_chunks):
        chunk_starts.append(chunk_size * chunk_index)

    # Get Chunk Stops
    chunk_stops = []
    chunk_stop = 0
    for chunk_index in range(number_of_chunks):
        chunk_stop += chunk_sizes[chunk_index]
        chunk_stops.append(chunk_stop)


    return number_of_chunks, chunk_sizes, chunk_starts, chunk_stops


def process_chunk(chunk_data,  indicies, image_height, image_width):

    # Remove Nans
    chunk_data = np.nan_to_num(chunk_data)

    smoothed_data = []
    for frame in chunk_data:
        reconstructed_frame = create_image_from_data(frame, indicies, image_height, image_width)
        reconstructed_frame = gaussian_filter(reconstructed_frame, sigma=1)
        reconstructed_frame = np.reshape(reconstructed_frame, (image_height * image_width))
        frame = reconstructed_frame[indicies]
        frame = np.multiply(frame, 65535)
        frame = np.clip(frame, a_min=0, a_max=65535)
        smoothed_data.append(frame)

    smoothed_data = np.array(smoothed_data)
    return smoothed_data


def create_sample_video_integer(base_directory):

    print("Creating Sample Delta F Video")

    # Load Mask
    indicies, frame_height, frame_width = load_mask(base_directory)

    # Load Processed Data
    delta_f_file_location = os.path.join(base_directory, "Delta_F.h5")
    delta_f_file = tables.open_file(delta_f_file_location, mode='r')
    processed_data = delta_f_file.root.Data

    # Get Sample Data
    sample_size = 7000
    sample_data = processed_data[1000:1000 + sample_size]
    sample_data = np.nan_to_num(sample_data)

    # Denoise with dimensionality reduction
    model = PCA(n_components=150)
    transformed_data = model.fit_transform(sample_data)
    sample_data = model.inverse_transform(transformed_data)

    # Get Colour Boundaries
    cm = plt.cm.ScalarMappable(norm=None, cmap='inferno')

    scale_factor = 0.7
    colour_max = 65535 * scale_factor
    colour_min = 0
    cm.set_clim(vmin=colour_min, vmax=colour_max)

    # Get Original Pixel Dimenions
    frame_width = 608
    frame_height = 600

    video_name = base_directory + "/Movie_Baseline.avi"
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, video_codec, frameSize=(frame_width, frame_height), fps=30)  # 0, 12

    # plt.ion()
    window_size = 3

    for frame in range(sample_size - window_size):  # number_of_files:
        template = np.zeros((frame_height * frame_width))

        image = sample_data[frame:frame + window_size]
        image = np.mean(image, axis=0)
        image = np.nan_to_num(image)
        np.put(template, indicies, image)
        image = np.reshape(template, (frame_height, frame_width))
        #image = ndimage.gaussian_filter(image, blur_size)

        colored_image = cm.to_rgba(image)
        colored_image = colored_image * 255

        image = np.ndarray.astype(colored_image, np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        video.write(image)

    cv2.destroyAllWindows()
    video.release()
    delta_f_file.close()


def convert_delta_f(base_directory):

    print("Processing Session: ", base_directory)

    # Open Delta F File
    delta_f_file_location = os.path.join(base_directory, "Delta_F.hdf5")

    with h5py.File(delta_f_file_location, 'r') as delta_f_file:

        # Extract Delta F Matrix
        delta_f_matrix = delta_f_file["Data"]
        number_of_timepoints, number_of_pixels = np.shape(delta_f_matrix)

        # Create New Filename
        new_filename = os.path.join(base_directory, "Delta_F.h5")
        new_file = tables.open_file(new_filename, mode='w')
        new_storage = new_file.create_earray(new_file.root, 'Data', tables.UInt16Atom(), shape=(0, number_of_pixels), expectedrows=number_of_timepoints)

        # Load Mask
        indicies, image_height, image_width = load_mask(base_directory)

        # Get Chunk Structure
        preferred_chunk_size = 10000
        number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = get_chunk_structure(preferred_chunk_size, number_of_timepoints)

        for chunk_index in range(number_of_chunks):
            print("Chunk: ", chunk_index, " of ", number_of_chunks, " at ", datetime.now())
            chunk_start = chunk_starts[chunk_index]
            chunk_stop = chunk_stops[chunk_index]

            chunk_data = delta_f_matrix[chunk_start:chunk_stop]
            chunk_data = process_chunk(chunk_data,  indicies, image_height, image_width)

            for frame in chunk_data:
                new_storage.append([frame])

            new_file.flush()

        new_file.close()

    # Both Files Should Now Be Closed

    # Create Demonstration Video
    create_sample_video_integer(base_directory)

    # Delete Original File
    os.remove(delta_f_file_location)



session_list = [

    # Controls 46 sessions

    # 78.1A - 6
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_16_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_17_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_19_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_21_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_24_Discrimination_Imaging",

    # 78.1D - 8
    #r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_14_Discrimination_Imaging",
    #r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_15_Discrimination_Imaging",
    #r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_16_Discrimination_Imaging",
    #r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_17_Discrimination_Imaging",
    #r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_19_Discrimination_Imaging",
    #r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_21_Discrimination_Imaging",
    #r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_23_Discrimination_Imaging",
    #r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_25_Discrimination_Imaging",

    #4.1B - 7
    #"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    #"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    #"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
    #"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
    #"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
    #"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging", - # Check At The ENd
    #"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging",

    #22.1A - 7
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_09_25_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_09_29_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_01_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_03_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_05_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_07_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging",

    # 14.1A - 6
    #r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_04_29_Discrimination_Imaging",
    #r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_01_Discrimination_Imaging",
    #r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_03_Discrimination_Imaging",
    #r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_05_Discrimination_Imaging",
    #r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_07_Discrimination_Imaging",
    #r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_09_Discrimination_Imaging",

    # 7.1B - 12
    #r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_01_Discrimination_Imaging",
    #r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_03_Discrimination_Imaging",
    #r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_05_Discrimination_Imaging",
    #r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_07_Discrimination_Imaging",
    #r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_09_Discrimination_Imaging",
    #r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_11_Discrimination_Imaging",
    #r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_13_Discrimination_Imaging",
    #r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_15_Discrimination_Imaging",
    #r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_17_Discrimination_Imaging",
    #r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_19_Discrimination_Imaging",
    #r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_22_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_24_Discrimination_Imaging",


    # Mutants

    # 4.1A - 15
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_02_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_04_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_06_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_08_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_10_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_12_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_14_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_16_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_18_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_23_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_25_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_27_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_03_01_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_03_03_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging",

    #20.1B - 11
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_09_28_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_09_30_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_02_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_04_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_06_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_09_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_11_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_13_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_15_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_17_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging",

    # 24.1C - 10
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_20_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_22_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_24_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_26_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_28_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_30_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_10_02_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_10_04_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_10_06_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_10_08_Discrimination_Imaging",

    # 10.1A - 8
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_04_30_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_02_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_04_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_06_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_08_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_10_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_12_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_14_Discrimination_Imaging"

    # NXAK16.1B - 16
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_04_30_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_02_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_04_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_06_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_08_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_10_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_12_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_14_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_16_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_18_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_20_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_22_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_24_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_26_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_06_04_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_06_15_Discrimination_Imaging",

    # 71.2A - 16

]


number_of_sessions = len(session_list)
session_count = 0
for base_directory in session_list:
    print("Session: ", session_count, " of ", number_of_sessions)
    convert_delta_f(base_directory)
    session_count += 1