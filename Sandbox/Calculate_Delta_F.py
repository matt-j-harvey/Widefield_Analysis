import numpy as np
import matplotlib.pyplot as plt
import h5py
import tables
from scipy import signal, ndimage, stats
import os
import cv2
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA, FactorAnalysis, TruncatedSVD
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import time

import Preprocessing_Utils


def get_session_name(base_directory):

    # Split File Path By Forward Slash
    split_base_directory = base_directory.split("/")

    # Take The Last Two and Join By Underscore
    session_name = split_base_directory[-2] + "_" + split_base_directory[-1]

    return session_name


def load_downsampled_mask(base_directory):

    mask = np.load(os.path.join(base_directory, "Generous_Mask.npy"))

    # Transform Mask
    mask = resize(mask, (300, 304), preserve_range=True, order=0, anti_aliasing=True)

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask > 0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width


def reconstruct_sample_video(base_directory, save_directory):
    print("Reconstructing Sample Video For Session", base_directory)

    # Load Data
    motion_corrected_data_file = get_motion_corrected_data_filename(base_directory)
    data_file = os.path.join(base_directory, motion_corrected_data_file)
    data_container = h5py.File(data_file, 'r')
    blue_array = data_container["Blue_Data"]
    violet_array = data_container["Violet_Data"]

    # Take Sample of Data
    blue_array   = blue_array[:, 1000:2000]
    violet_array = violet_array[:, 1000:2000]

    # Transpose Data
    blue_array = np.transpose(blue_array)
    violet_array = np.transpose(violet_array)

    # Convert From 16 bit to 8 bit
    blue_array   = np.divide(blue_array, 65536)
    violet_array = np.divide(violet_array, 65536)

    blue_array = np.multiply(blue_array, 255)
    violet_array = np.multiply(violet_array, 255)

    # Get Original Pixel Dimensions
    frame_width = 608
    frame_height = 600

    # Load Mask
    mask = np.load(os.path.join(base_directory, "Generous_Mask.npy"))
    mask = np.where(mask > 0.1, 1, 0)
    mask = mask.astype(int)

    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    # Create Video File
    reconstructed_video_file = os.path.join(save_directory, "Greyscale_Reconstruction.avi")
    video_name = reconstructed_video_file
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, video_codec, frameSize=(frame_width * 2, frame_height), fps=30)  # 0, 12

    number_of_frames = np.shape(blue_array)[0]


    for frame in range(number_of_frames):

        blue_template = np.zeros(frame_height * frame_width)
        violet_template = np.zeros(frame_height * frame_width)

        blue_frame = blue_array[frame]
        violet_frame = violet_array[frame]

        blue_template[indicies] = blue_frame
        violet_template[indicies] = violet_frame

        blue_template = np.ndarray.astype(blue_template, np.uint8)
        violet_template = np.ndarray.astype(violet_template, np.uint8)

        blue_frame   = np.reshape(blue_template, (600,608))
        violet_frame = np.reshape(violet_template, (600, 608))

        image = np.hstack((violet_frame, blue_frame))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        video.write(image)

    cv2.destroyAllWindows()
    video.release()

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



def heamocorrection_regression(blue_data, violet_data):

    # Perform Regression
    chunk_size = np.shape(blue_data)[0]
    for pixel in range(chunk_size):

        # Load Pixel Traces
        violet_trace = violet_data[pixel]
        blue_trace = blue_data[pixel]

        # Perform Regression
        slope, intercept, r, p, stdev = stats.linregress(violet_trace, blue_trace)

        # Scale Violet Trace
        violet_trace = np.multiply(violet_trace, slope)
        violet_trace = np.add(violet_trace, intercept)

        # Subtract From Blue Trace
        blue_trace = np.subtract(blue_trace, violet_trace)

        # Insert Back Corrected Trace
        blue_data[pixel] = blue_trace

    return blue_data



def calculate_delta_f(activity_matrix):

    # Get Baseline
    baseline_vector = np.mean(activity_matrix, axis=1)

    # Transpose Baseline Vector so it can be used by numpy subtract
    baseline_vector = baseline_vector[:, np.newaxis]

    # Get Delta F
    delta_f = np.subtract(activity_matrix, baseline_vector)

    # Divide by baseline
    delta_f_over_f = np.divide(delta_f, baseline_vector)

    # Remove NANs
    delta_f_over_f = np.nan_to_num(delta_f_over_f)

    return delta_f_over_f






def save_session_metadata(base_directory, delta_f_file, violet_baseline_frames, blue_baseline_frames, lowcut_filter, lowcut_freq, exclusion_point, gaussian_filter_width, pixel_baseline_list, pixel_maximum_list):

    # Add Metadata
    session_name = get_session_name(base_directory)
    ai_filename = Preprocessing_Utils.get_ai_filename(base_directory)
    metadata_table = delta_f_file.create_table(where=delta_f_file.root, name='metadata_table', description=metadata_particle, title="metadata_table")
    metadata_row = metadata_table.row
    metadata_row['session_name'] = session_name
    metadata_row['ai_filename'] = ai_filename
    metadata_row['lowcut_filter'] = lowcut_filter
    metadata_row['lowcut_freq'] = lowcut_freq
    metadata_row['exclusion_point'] = exclusion_point
    metadata_row['gaussian_filter'] = gaussian_filter
    metadata_row['gaussian_filter_width'] = gaussian_filter_width
    metadata_row.append()
    metadata_table.flush()

    # Add Baseline Frames
    if violet_baseline_frames!= None:
        delta_f_file.create_array(delta_f_file.root, 'violet_baseline_frames', np.array(violet_baseline_frames), "violet_baseline_frames")
        delta_f_file.create_array(delta_f_file.root, 'blue_baseline_frames', np.array(blue_baseline_frames), "blue_baseline_frames")

    # Add Pixel Baselines and Pixel Maximums
    delta_f_file.create_array(delta_f_file.root, 'pixel_baseline_list', np.array(pixel_baseline_list), "pixel_baseline_list")
    delta_f_file.create_array(delta_f_file.root, 'pixel_maximum_list', np.array(pixel_maximum_list), "pixel_maximum_list")



def get_baseline_mean_and_sd(processed_data, baseline_frames):
    processed_data = np.transpose(processed_data)
    baseline_mean = np.nanmean(processed_data[baseline_frames], axis=0)
    baseline_sd = np.nanstd(processed_data[baseline_frames], axis=0)
    return baseline_mean, baseline_sd


def lowcut_filter(X, w = 0.0033, fs = 30.):
    from scipy.signal import butter, filtfilt
    b, a = butter(2,w/(fs/2.), btype='highpass');
    return filtfilt(b, a, X, padlen=10000, axis=0)

def highcut_filter(X, w = 7.5, fs = 30.):
    from scipy.signal import butter, filtfilt
    b, a = butter(2,w/(fs/2.), btype='lowpass');
    return filtfilt(b, a, X, padlen=10000, axis=0)




def process_chunk(data_matrix, chunk_indicies, exclusion_point, lowcut, highcut):

    # Remove Early Data
    chunk_data = data_matrix[chunk_indicies, exclusion_point:]

    # Remove NaNs
    chunk_data = np.nan_to_num(chunk_data)

    # Calculate Delta F
    chunk_data = calculate_delta_f(chunk_data)

    # Transpose
    chunk_data = np.transpose(chunk_data)

    if lowcut == True:
        chunk_data = lowcut_filter(chunk_data)

    if highcut == True:
        chunk_data = highcut_filter(chunk_data)

    # Convert To 32 Bit Float
    processed_data = np.ndarray.astype(chunk_data, np.float32)

    return processed_data



def create_delta_f_file(base_directory, output_directory, exclusion_point=3000):

    """
    Order of operations taken from Anne Churchland Group Github:  https://github.com/churchlandlab/wfield/tree/master/wfield
    Also See Paper: Chronic, cortex-wide imaging of specific cell populations during behavior - Joao Couto - Nat Protoc. 2021 Jul; 16(7): 3241–3263. - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8788140/
    Data analysis (Stage 4) pg 9

    Steps
    1 - Motion Correction
    2 - Delta F Over F (F is mean value over trial)
    3 - Denoising + Compression with SVD
    4 - Lowpass Filtering
    5 - Regression And Subtraction

    I also calculate the Mean and SD Of Each Pixel In the Baseline Periods Prior To Stimuli Onsets and Save These, This Allows Me to Z Score The Data Later If I Want

    This First 2-3 Mins (approx 3000 frames) The LEDs Are Initially Quite Bright And Then Dim, I Think This Is Due to Heating Effects, So I Exclude THe First 2-3 Mins of Each Session
    """

    # Get Filenames
    uncorrected_delta_f = os.path.join(output_directory, "Uncorrected_Delta_F.hdf5")

    # Load Data
    motion_corrected_filename = "Motion_Corrected_Downsampled_Data.hdf5"
    motion_corrected_file = os.path.join(base_directory, motion_corrected_filename)
    motion_corrected_data_container = h5py.File(motion_corrected_file, 'r')
    blue_matrix = motion_corrected_data_container["Violet_Data"]
    violet_matrix = motion_corrected_data_container["Blue_Data"]

    # Load Downsampled Mask
    indicies, image_height, image_width = load_downsampled_mask(base_directory)

    # Get Data Structure
    number_of_images = np.shape(blue_matrix)[1]
    number_of_pixels = len(indicies)
    print("Number of images", number_of_images)
    print("number of pixels", number_of_pixels)

    # Define Chunking Settings
    preferred_chunk_size = 20000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = get_chunk_structure(preferred_chunk_size, number_of_pixels)

    r2_coef_map = []
    r2_intercept_map = []
    print("Calcularing Delta F")
    with h5py.File(uncorrected_delta_f, "w") as f:
       blue_dataset = f.create_dataset("Blue_DF", (number_of_images, number_of_pixels), dtype=np.float32, chunks=True, compression=False)
       violet_dataset = f.create_dataset("Violet_DF", (number_of_images, number_of_pixels), dtype=np.float32, chunks=True, compression=False)

       for chunk_index in tqdm(range(number_of_chunks)):

           # Get Selected Indicies
           chunk_start = int(chunk_starts[chunk_index])
           chunk_stop = int(chunk_stops[chunk_index])
           chunk_indicies = indicies[chunk_start:chunk_stop]

           # Process This Chunk
           blue_chunk = process_chunk(blue_matrix, chunk_indicies, exclusion_point, lowcut=True, highcut=False)
           violet_chunk = process_chunk(violet_matrix, chunk_indicies, exclusion_point, lowcut=True, highcut=True)



           #plt.plot(np.mean(blue_chunk, axis=1), c='b')
           #plt.plot(np.mean(violet_chunk, axis=1), c='m')
           #plt.show()

           # Insert Back
           blue_dataset[exclusion_point:, chunk_start:chunk_stop] = blue_chunk
           violet_dataset[exclusion_point:, chunk_start:chunk_stop] = violet_chunk

    # Close Motion Correction File
    motion_corrected_data_container.close()


base_directory = r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging/Downsampled_Raw_Data"
create_delta_f_file(base_directory, base_directory)