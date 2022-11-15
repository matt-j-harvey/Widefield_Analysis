import os

import numpy as np
import tables
import numpy
import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec
from scipy import signal
from sklearn.decomposition import PCA

import Preprocessing_Utils


def create_sample_video_integer(base_directory, output_directory):

   print("Creating Sample Delta F Video")

   # Load Mask
   indicies, frame_height, frame_width = load_mask(base_directory)

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

   # Get Colour Boundaries
   colourmap = Preprocessing_Utils.get_musall_cmap()
   cm = plt.cm.ScalarMappable(norm=None, cmap=colourmap)
   colour_magnitude = 4
   cm.set_clim(vmin=-1 * colour_magnitude, vmax=colour_magnitude)

   # Get Original Pixel Dimenions
   frame_width = 608
   frame_height = 600

   video_name = os.path.join(output_directory, "Movie_Baseline.avi")
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

       colored_image = cm.to_rgba(image)
       colored_image = colored_image * 255

       image = np.ndarray.astype(colored_image, np.uint8)

       image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

       video.write(image)

   cv2.destroyAllWindows()
   video.release()
   delta_f_file.close()


def get_baseline_means(delta_f_container, )

def create_comparison_movie(base_directory, early_cutoff=3000, sample_size=1000):

    # Load Processed Data
    delta_f_file_location = os.path.join(base_directory, "Delta_F.h5")
    delta_f_file = tables.open_file(delta_f_file_location, mode='r')
    print("Delta F File", delta_f_file)
    processed_data = delta_f_file.root.Data
    processed_data_sample = processed_data[early_cutoff:early_cutoff + sample_size]


    # Read Metadata
    metadata = delta_f_file.root['metadata_table']
    read_session_metadata(metadata)

    # Load Pixel SDs
    pixel_sds = delta_f_file.root['pixel_baseline_list']

    # Load Mask
    indicies, image_height, image_width = Preprocessing_Utils.load_generous_mask(base_directory)
    sd_map = Preprocessing_Utils.create_image_from_data(pixel_sds, indicies, image_height, image_width)
    plt.imshow(sd_map, vmax=np.percentile(sd_map, q=99))
    plt.show()

    # Convert Processed Data Sample
    pixel_sds = np.nan_to_num(pixel_sds)
    unnormalised_sample = np.multiply(processed_data_sample, pixel_sds)

    """
    processed_data_sample = np.multiply(processed_data_sample, pixel_sds)
    processed_data_sample = np.subtract(processed_data_sample, pixel_sds)
    processed_data_sample = np.divide(processed_data_sample, pixel_sds)
    """

    # Clean BOth Samples
    processed_data_sample = clean_video_data(processed_data_sample)
    unnormalised_sample = clean_video_data(unnormalised_sample)

    figure_1 = plt.figure()
    rows = 1
    columns = 2
    gridspec_1 = GridSpec(nrows=rows, ncols=columns, figure=figure_1)

    colourmap = Preprocessing_Utils.get_musall_cmap()

    norm_mangitude = np.percentile(np.abs(processed_data_sample), 99)
    raw_magnitude = np.percentile(np.abs(unnormalised_sample), 99)

    plt.ion()
    for x in range(sample_size):

        raw_axis = figure_1.add_subplot(gridspec_1[0, 0])
        norm_axis = figure_1.add_subplot(gridspec_1[0, 1])

        raw_data = unnormalised_sample[x]
        norm_data = processed_data_sample[x]

        raw_image = Preprocessing_Utils.create_image_from_data(raw_data, indicies, image_height, image_width)
        norm_image = Preprocessing_Utils.create_image_from_data(norm_data, indicies, image_height, image_width)

        raw_axis.imshow(raw_image, vmin=-raw_magnitude, vmax=raw_magnitude, cmap=colourmap)
        norm_axis.imshow(norm_image, vmin=-norm_mangitude, vmax=norm_mangitude, cmap=colourmap)

        raw_axis.set_title("Raw Delta F")
        norm_axis.set_title("Z Score Delta F")

        plt.draw()
        plt.pause(0.1)
        plt.clf()
    """
    # Get Sample Data
    start_time = 10000
    sample_size = 3000
    sample_data = processed_data[start_time:start_time + sample_size]
    sample_data = np.nan_to_num(sample_data)
    
    # Un-Normalise A Sample
    """

    delta_f_file.close()

def clean_video_data(sample_data):

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

    return sample_data

def read_session_metadata(metadata_table):
    metadata_table = metadata_table[0]

    print("")
    print("")
    print("Session Report:")
    print("session_name:", metadata_table['session_name'])
    print("ai_filename:", metadata_table['ai_filename'])
    print("lowcut_filter:", metadata_table['lowcut_filter'])
    print("lowcut_freq:", metadata_table['lowcut_freq'])
    print("exclusion_point:", metadata_table['exclusion_point'])
    print("gaussian_filter:", metadata_table['gaussian_filter'])
    print("gaussian_filter_width:", metadata_table['gaussian_filter_width'])
    print("")
    print("")



base_directory = r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Control_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging"
create_comparison_movie(base_directory)