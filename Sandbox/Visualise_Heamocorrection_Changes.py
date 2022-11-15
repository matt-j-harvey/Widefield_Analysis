
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
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import time

import Preprocessing_Utils


def get_motion_corrected_data_filename(base_directory):

    file_list = os.listdir(base_directory)
    for file in file_list:
        if "Motion_Corrected_Mask_Data" in file:
            return file




def heamocorrection_regression(blue_data, violet_data):

    # Perform Regression
    corrected_data = []

    chunk_size = np.shape(blue_data)[0]
    for pixel in range(chunk_size):

        """
        figure_1 = plt.figure()
        rows = 2
        columns = 1
        uncorrected_axis = figure_1.add_subplot(rows, columns, 1)
        corrected_axis = figure_1.add_subplot(rows, columns, 2)
        """

        # Load Pixel Traces
        violet_trace = violet_data[pixel]
        blue_trace = blue_data[pixel]

        #uncorrected_axis.plot(blue_trace, c='b')
        #uncorrected_axis.plot(violet_trace, c='m')


        # Perform Regression
        #slope, intercept, r, p, stdev = stats.linregress(violet_trace, blue_trace)

        # Scale Violet Trace
        #violet_trace = np.multiply(violet_trace, slope)
        #violet_trace = np.add(violet_trace, intercept)

        #corrected_axis.plot(blue_trace, c='b')
        #corrected_axis.plot(violet_trace, c='m')
        #plt.show()

        # Subtract From Blue Trace
        corrected_trace = np.subtract(blue_trace, violet_trace)


        # Insert Back Corrected Trace
        corrected_data.append(corrected_trace)

    corrected_data = np.array(corrected_data)
    return corrected_data




def get_lowcut_coefs(w=0.0033, fs=28.):
    b, a = signal.butter(2, w/(fs/2.), btype='highpass');


    return b, a


def perform_lowcut_filter(data, b, a):
    filtered_data = signal.filtfilt(b, a, data, padlen=10000)
    return filtered_data





def plot_pre_and_post_filter_traces(pre_blue, pre_violet, post_blue, post_violet, output_directory, chunk_index):

    figure_1 = plt.figure()

    rows = 2
    columns = 1
    pre_filter_axis = figure_1.add_subplot(rows, columns, 1)
    post_filter_axis = figure_1.add_subplot(rows, columns, 2)
    #delta_f_axis = figure_1.add_subplot(rows, columns, 3)

    pre_filter_axis.set_title("Pre Filtering")
    pre_filter_axis.plot(pre_blue, c='b')
    pre_filter_axis_purple = pre_filter_axis.twinx()
    pre_filter_axis_purple.plot(pre_violet, c='m')

    post_filter_axis.set_title("Post Filtering")
    post_filter_axis.plot(post_blue, c='b')
    post_filter_axis.plot(post_violet, c='m')

    #delta_f_axis.set_title("Delta F")
    #delta_f_axis.plot(blue_delta_f, c='b')
    #delta_f_axis.plot(violet_delta_f, c='m')

    plt.savefig(os.path.join(output_directory, "Blue_Violet_Filter_Comparison_" + str(chunk_index).zfill(3) + ".png"))
    plt.close()


def view_violet_movie(base_directory):

    # Load Violet File
    violet_df_file = os.path.join(output_directory, "violet_DF.hdf5")
    violet_df_file_container = h5py.File(violet_df_file, 'r')
    violet_df_dataset = violet_df_file_container["Data"]

    sample_start = 3000
    sample_finish = 10000

    indicies, image_height, image_width = Preprocessing_Utils.load_generous_mask(base_directory)
    violet_sample = violet_df_dataset[sample_start:sample_finish]

    # Denoise with dimensionality reduction
    model = PCA(n_components=150)
    transformed_data = model.fit_transform(violet_sample)
    violet_sample = model.inverse_transform(transformed_data)

    # Create Video File
    video_name = os.path.join(base_directory, "Violet_DF_Movie.avi")
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, video_codec, frameSize=(image_width, image_height), fps=30)  # 0, 12

    # Create Colourmap
    colourmap = Preprocessing_Utils.get_musall_cmap()
    cm = plt.cm.ScalarMappable(norm=None, cmap=colourmap)
    colour_magnitude = 0.04
    cm.set_clim(vmin=-1 * colour_magnitude, vmax=colour_magnitude)

    for frame_index in range(sample_start, sample_finish-4):
        violet_frame = violet_sample[frame_index:frame_index+3]
        violet_frame = np.mean(violet_frame, axis=0)
        violet_frame = Preprocessing_Utils.create_image_from_data(violet_frame, indicies, image_height, image_width)
        violet_frame = ndimage.gaussian_filter(violet_frame, 1)

        # Set Image Colours
        violet_frame = cm.to_rgba(violet_frame)
        violet_frame = violet_frame * 255

        violet_frame = np.ndarray.astype(violet_frame, np.uint8)

        violet_frame = cv2.cvtColor(violet_frame, cv2.COLOR_RGB2BGR)

        video.write(violet_frame)

    cv2.destroyAllWindows()
    video.release()
    violet_df_file_container.close()


def lowpass(X, w = 7.5, fs = 30.):
    from scipy.signal import butter, filtfilt
    b, a = butter(2,w/(fs/2.), btype='lowpass');
    return filtfilt(b, a, X, padlen=50)


def create_comparison_movie(base_directory, output_directory):

    # Get Filenames
    blue_df_file = os.path.join(output_directory, "Blue_DF.hdf5")
    violet_df_file = os.path.join(output_directory, "violet_DF.hdf5")
    delta_f_file = os.path.join(output_directory, "Delta_F.hdf5")

    # Open Files
    blue_df_file_container = h5py.File(blue_df_file, 'r')
    violet_df_file_container = h5py.File(violet_df_file, 'r')
    delta_f_file_container = h5py.File(delta_f_file, 'r')

    # Create Datasets
    blue_df_dataset = blue_df_file_container["Data"]
    violet_df_dataset = violet_df_file_container["Data"]
    df_dataset = delta_f_file_container["Data"]

    sample_start = 3000
    sample_finish = 4000

    # Create Colourmap
    colourmap = Preprocessing_Utils.get_musall_cmap()
    full_colourmap = plt.cm.ScalarMappable(norm=None, cmap=colourmap)
    colour_magnitude = 0.04
    full_colourmap.set_clim(vmin=-1 * colour_magnitude, vmax=colour_magnitude)

    figure_1 = plt.figure()
    rows = 1
    columns = 4


    indicies, image_height, image_width = Preprocessing_Utils.load_generous_mask(base_directory)

    for frame_index in range(sample_start, sample_finish):

        blue_axis = figure_1.add_subplot(rows, columns, 1)
        violet_axis = figure_1.add_subplot(rows, columns, 2)
        subtracted_axis = figure_1.add_subplot(rows, columns, 3)
        difference_axis = figure_1.add_subplot(rows, columns, 4)

        blue_frame = blue_df_dataset[frame_index]
        violet_frame = violet_df_dataset[frame_index]
        df_frame = df_dataset[frame_index]

        blue_frame = Preprocessing_Utils.create_image_from_data(blue_frame, indicies, image_height, image_width)
        violet_frame = Preprocessing_Utils.create_image_from_data(violet_frame, indicies, image_height, image_width)
        df_frame = Preprocessing_Utils.create_image_from_data(df_frame, indicies, image_height, image_width)
        difference_frame = np.subtract(blue_frame, df_frame)

        blue_frame = full_colourmap.to_rgba(blue_frame)
        violet_frame = full_colourmap.to_rgba(violet_frame)
        df_frame = full_colourmap.to_rgba(df_frame)
        difference_frame = full_colourmap.to_rgba(difference_frame)

        blue_axis.set_title("Blue Signal")
        violet_axis.set_title("VioletSignal")
        subtracted_axis.set_title("Corrected Signal")
        difference_axis.set_title("Blue-Corrected")

        blue_axis.imshow(blue_frame)
        violet_axis.imshow(violet_frame)
        subtracted_axis.imshow(df_frame)
        difference_axis.imshow(difference_frame)


        plt.draw()
        plt.pause(0.1)
        plt.clf()

    blue_df_file_container.close()
    violet_df_file_container.close()
    delta_f_file_container.close()


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


def create_sample_video_integer(base_directory, save_directory):

    print("Creating Sample Delta F Video")

    # Load Mask
    indicies, frame_height, frame_width = Preprocessing_Utils.load_generous_mask(base_directory)

    # Load Processed Data
    delta_f_file_location = os.path.join(save_directory, "Delta_F.hdf5")
    delta_f_file_container = h5py.File(delta_f_file_location, 'r')
    processed_data = delta_f_file_container["Data"]

    # Get Sample Data
    start_time = 10000
    sample_size = 10000
    sample_data = processed_data[start_time:start_time + sample_size]
    sample_data = np.nan_to_num(sample_data)

    # Filter
    sampling_frequency = 28  # In Hertz
    cutoff_frequency = 12  # In Hertz
    w = cutoff_frequency / (sampling_frequency / 2)  # Normalised frequency
    b, a = signal.butter(1, w, 'lowpass')
    sample_data = signal.filtfilt(b, a, sample_data, axis=0)

    # Denoise with dimensionality reduction
    model = PCA(n_components=150)
    transformed_data = model.fit_transform(sample_data)
    sample_data = model.inverse_transform(transformed_data)

    # Get Colour Map
    colourmap = Preprocessing_Utils.get_musall_cmap()
    cm = plt.cm.ScalarMappable(norm=None, cmap=colourmap)
    colour_magnitude = 0.05
    cm.set_clim(vmin=-1 * colour_magnitude, vmax=colour_magnitude)

    # Get Background Pixels
    #background_pixels = get_background_pixels(indicies, frame_height, frame_width)

    # Get Original Pixel Dimenions
    frame_width = 608
    frame_height = 600

    # Create Video File
    video_name = os.path.join(save_directory, "Movie_Baseline.avi")
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
        image = ndimage.gaussian_filter(image, 1)

        # Set Image Colours
        colored_image = cm.to_rgba(image)
        #colored_image[background_pixels] = [1, 1, 1, 1]
        colored_image = colored_image * 255

        image = np.ndarray.astype(colored_image, np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        video.write(image)

    cv2.destroyAllWindows()
    video.release()
    delta_f_file.close()



def visualise_heamocorrection_changes(base_directory, output_directory, exclusion_point=3000, lowcut_filter=True, low_cut_freq=0.0033, gaussian_filter=True, gaussian_filter_width=1):

    """
    Order of operations taken from Anne Churchland Group Github:  https://github.com/churchlandlab/wfield/tree/master/wfield
    Also See Paper: Chronic, cortex-wide imaging of specific cell populations during behavior - Joao Couto - Nat Protoc. 2021 Jul; 16(7): 3241–3263. - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8788140/
    Data analysis (Stage 4) pg 9

    Steps
    Data should already be motion corrected
    1 - Delta F Over F (F is mean value over trial)
    2 - Lowpass Filtering
    3 - Regression And Subtraction

    I also calculate the Mean and SD Of Each Pixel In the Baseline Periods Prior To Stimuli Onsets and Save These, This Allows Me to Z Score The Data Later If I Want

    This First 2-3 Mins (approx 3000 frames) The LEDs Are Initially Quite Bright And Then Dim, I Think This Is Due to Heating Effects, So I Exclude THe First 2-3 Mins of Each Session
    """

    # Load Data
    motion_corrected_filename = get_motion_corrected_data_filename(base_directory)
    motion_corrected_file = os.path.join(base_directory, motion_corrected_filename)
    motion_corrected_data_container = h5py.File(motion_corrected_file, 'r')
    blue_matrix = motion_corrected_data_container["Blue_Data"]
    violet_matrix = motion_corrected_data_container["Violet_Data"]

    # Get Data Structure
    number_of_pixels, number_of_images = np.shape(blue_matrix)
    print("Number of images", number_of_images)
    print("number of pixels", number_of_pixels)

    # Get Butterworth Filter Coefficients
    b, a, = get_lowcut_coefs(w=low_cut_freq)

    # Get Filenames
    blue_df_file = os.path.join(output_directory, "Blue_DF.hdf5")
    violet_df_file = os.path.join(output_directory, "violet_DF.hdf5")
    delta_f_file = os.path.join(output_directory, "Delta_F.hdf5")

    # Open Files
    blue_df_file_container = h5py.File(blue_df_file, 'w')
    violet_df_file_container = h5py.File(violet_df_file, 'w')
    delta_f_file_container = h5py.File(delta_f_file, 'w')

    # Create Datasets
    blue_df_dataset = blue_df_file_container.create_dataset("Data", (number_of_images, number_of_pixels), dtype=np.float32, chunks=True, compression=False)
    violet_df_dataset = violet_df_file_container.create_dataset("Data", (number_of_images, number_of_pixels), dtype=np.float32, chunks=True, compression=False)
    df_dataset = delta_f_file_container.create_dataset("Data", (number_of_images, number_of_pixels), dtype=np.float32, chunks=True, compression=False)

    # Define Chunking Settings
    preferred_chunk_size = 5000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = Preprocessing_Utils.get_chunk_structure(preferred_chunk_size, number_of_pixels)

    print("Heamocorrecting")
    for chunk_index in tqdm(range(number_of_chunks)):
        chunk_start = int(chunk_starts[chunk_index])
        chunk_stop = int(chunk_stops[chunk_index])

        # Extract Data
        blue_data = blue_matrix[chunk_start:chunk_stop]
        violet_data = violet_matrix[chunk_start:chunk_stop]

        # Remove Early Cutoff
        blue_data = blue_data[:, exclusion_point:]
        violet_data = violet_data[:, exclusion_point:]

        # Remove NaNs
        blue_data = np.nan_to_num(blue_data)
        violet_data = np.nan_to_num(violet_data)

        # Calculate Delta F
        blue_data = calculate_delta_f(blue_data)
        violet_data = calculate_delta_f(violet_data)

        pre_filter_blue_mean = np.mean(blue_data, axis=0)
        pre_filter_violet_mean = np.mean(violet_data, axis=0)

        print("Pre Filter shape", np.shape(blue_data))

        # Lowcut Filter
        if lowcut_filter == True:
            blue_data = perform_lowcut_filter(blue_data, b, a)
            violet_data = perform_lowcut_filter(violet_data, b, a)

        print("Post Filter shape", np.shape(blue_data))

        post_filter_blue_mean = np.mean(blue_data, axis=0)
        post_filter_violet_mean = np.mean(violet_data, axis=0)

        # Plot Traces
        plot_pre_and_post_filter_traces(pre_filter_blue_mean, pre_filter_violet_mean, post_filter_blue_mean, post_filter_violet_mean, output_directory, chunk_index)

        # Perform Regression
        processed_data = heamocorrection_regression(blue_data, violet_data)

        # Transpose
        processed_data = np.transpose(processed_data)
        blue_data = np.transpose(blue_data)
        violet_data = np.transpose(violet_data)

        # Convert to 32 Bit Float
        processed_data = np.ndarray.astype(processed_data, np.float32)
        blue_data = np.ndarray.astype(blue_data, np.float32)
        violet_data = np.ndarray.astype(violet_data, np.float32)

        # Put Back
        blue_df_dataset[exclusion_point:, chunk_start:chunk_stop] = blue_data
        violet_df_dataset[exclusion_point:, chunk_start:chunk_stop] = violet_data
        df_dataset[exclusion_point:, chunk_start:chunk_stop] = processed_data

    # Close Motion Corrected Data
    motion_corrected_data_container.close()

    # Close Heamocorrection Files
    blue_df_file_container.close()
    violet_df_file_container.close()
    delta_f_file_container.close()

base_directory = r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging"
output_directory = r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging/Heamocorrection_Visualisation"
#visualise_heamocorrection_changes(base_directory, output_directory)
#create_comparison_movie(base_directory, output_directory)
#view_violet_movie(base_directory)
create_sample_video_integer(output_directory, output_directory)