import h5py
import os
from skimage.transform import rescale, downscale_local_mean, resize
from sklearn.decomposition import TruncatedSVD
from scipy import ndimage, stats
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import Normalize
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from tqdm import tqdm
import cv2

from Widefield_Utils import widefield_utils


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def get_violet_file(base_directory):
    file_list = os.listdir(base_directory)
    for file in file_list:
        if "Violet_Data" in file:
            return base_directory + "/" + file


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


def highcut_filter(X, w = 7.5, fs = 30.):
    from scipy.signal import butter, filtfilt
    b, a = butter(2,w/(fs/2.), btype='lowpass');
    return filtfilt(b, a, X, padlen=0, axis=0)

def calculate_detla_f(activity_matrix):
    mean_frame = np.mean(activity_matrix, axis=0)
    activity_matrix = np.subtract(activity_matrix, mean_frame)
    activity_matrix = np.divide(activity_matrix, mean_frame)
    activity_matrix = np.nan_to_num(activity_matrix)
    return activity_matrix


def denoise_data(activity_matrix):
    print("Denosing Data")
    model = TruncatedSVD(n_components=100)
    transformed_data = model.fit_transform(activity_matrix)
    activity_matrix = model.inverse_transform(transformed_data)
    return activity_matrix

def regress_out_violet_traces(blue_data, violet_data):

    regression_coefs = []
    corrected_data = []

    number_of_frames, number_of_pixels = np.shape(blue_data)
    for pixel_index in tqdm(range(number_of_pixels)):
        blue_trace = blue_data[:, pixel_index]
        violet_trace = violet_data[:, pixel_index]

        slope, intercept, r, p, stdev = stats.linregress(violet_data[:, pixel_index], blue_data[:, pixel_index])
        regression_coefs.append(slope)

        # Scale Violet Trace
        scaled_violet_trace = np.multiply(violet_trace, slope)
        scaled_violet_trace = np.add(scaled_violet_trace, intercept)

        # Subtract Violet Trace From Blue Trace
        corrected_trace = np.subtract(blue_trace, scaled_violet_trace)
        corrected_data.append(corrected_trace)

    corrected_data = np.array(corrected_data)
    corrected_data = np.transpose(corrected_data










                                  )
    return regression_coefs, corrected_data


def apply_tight_mask(data, indicies, image_height, image_width):
    data = np.reshape(data, (image_height * image_width))
    data = data[indicies]
    template = np.zeros(image_height * image_width)
    template[indicies] = data
    data = np.reshape(template, (image_height, image_width))
    return data

def view_heamodynamic_signals(base_directory, output_directory):

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_downsampled_mask(base_directory)

    # Load Violet Data
    motion_corrected_data_file = os.path.join(base_directory, "Motion_Corrected_Downsampled_Data.hdf5")
    violet_file = h5py.File(motion_corrected_data_file, mode="r")
    violet_data = violet_file["Violet_Data"]
    blue_data = violet_file["Blue_Data"]
    print("Violet data shape", np.shape(violet_data))

    violet_sample = np.array(violet_data[:, 20000:30000])
    blue_sample = np.array(blue_data[:, 20000:30000])
    violet_sample = np.transpose(violet_sample)
    blue_sample = np.transpose(blue_sample)
    print("Violet sample shape", np.shape(violet_sample))

    # Calculate Delta F
    violet_delta_f = calculate_detla_f(violet_sample)
    blue_delta_f = calculate_detla_f(blue_sample)
    print("Violet delta f shape", np.shape(violet_delta_f))

    # Denoise Data
    violet_delta_f = denoise_data(violet_delta_f)
    blue_delta_f = denoise_data(blue_delta_f)
    print("Violet delta f denoised shape", np.shape(violet_delta_f))

    # Filter Violet
    violet_delta_f = highcut_filter(violet_delta_f)
    print("Violet delta f filtered shape", np.shape(violet_delta_f))

    # Perform Regression
    regression_coefs, blue_delta_f = regress_out_violet_traces(blue_delta_f, violet_delta_f)
    print("Corrected Data", np.shape(blue_delta_f))

    regression_coef_image = widefield_utils.create_image_from_data(regression_coefs, indicies, image_height, image_width)
    plt.imshow(regression_coef_image)
    plt.colorbar()
    plt.show()

    # Temporal Smoothing
    violet_delta_f = moving_average(violet_delta_f)
    blue_delta_f = moving_average(blue_delta_f)

    np.save(os.path.join(base_directory, "Violet_Delta_F.npy"), violet_delta_f)

    # Load Alignment dicts
    within_mouse_alignment_dict = np.load(os.path.join(base_directory, "Within_Mouse_Alignment_Dictionary.npy"), allow_pickle=True)[()]
    across_mouse_alignment_dict = widefield_utils.load_across_mice_alignment_dictionary(base_directory)

    # Load Common Mask
    common_indicies, common_height, common_width = widefield_utils.load_tight_mask()

    # Get Background Pixels
    background_pixels = widefield_utils.get_background_pixels(common_indicies, common_height, common_width)

    # Create Colourmaps
    widefield_colourmap = widefield_utils.get_musall_cmap()
    violet_colourmap = plt.cm.ScalarMappable(norm=Normalize(vmin=-0.02, vmax=0.02), cmap=widefield_colourmap)
    blue_colourmap = plt.cm.ScalarMappable(norm=Normalize(vmin=-0.05, vmax=0.05), cmap=widefield_colourmap)

    # Create Video File
    video_name = os.path.join(output_directory, "Heamodynamic_Signals.avi")
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, video_codec, frameSize=(1000, 500), fps=30)  # 0, 12

    figure_1 = plt.figure(figsize=(10, 5))
    canvas = FigureCanvasAgg(figure_1)
    sample_length = np.shape(violet_delta_f)[0]
    for frame_index in tqdm(range(sample_length)):

        # Create Axes
        rows = 1
        columns = 2
        blue_axis = figure_1.add_subplot(rows, columns, 1)
        violet_axis = figure_1.add_subplot(rows, columns, 2)

        # Extract Frames
        blue_frame = blue_delta_f[frame_index]
        violet_frame = violet_delta_f[frame_index]

        # Reconstruct Frames
        blue_frame = widefield_utils.create_image_from_data(blue_frame, indicies, image_height, image_width)
        violet_frame = widefield_utils.create_image_from_data(violet_frame, indicies, image_height, image_width)

        # Align Within Mice
        blue_frame = widefield_utils.transform_image(blue_frame, within_mouse_alignment_dict)
        violet_frame = widefield_utils.transform_image(violet_frame, within_mouse_alignment_dict)

        # Align across Mice
        blue_frame = widefield_utils.transform_image(blue_frame, across_mouse_alignment_dict)
        violet_frame = widefield_utils.transform_image(violet_frame, across_mouse_alignment_dict)

        # Apply Tight Mask
        blue_frame = apply_tight_mask(blue_frame, common_indicies, common_height, common_width)
        violet_frame = apply_tight_mask(violet_frame, common_indicies, common_height, common_width)

        # Gaussian Filter
        blue_frame = ndimage.gaussian_filter(blue_frame, sigma=1)
        violet_frame = ndimage.gaussian_filter(violet_frame, sigma=1)

        # Set Colours
        blue_frame = blue_colourmap.to_rgba(blue_frame)
        violet_frame = violet_colourmap.to_rgba(violet_frame)

        # Remove Background
        blue_frame[background_pixels] = (1,1,1,1)
        violet_frame[background_pixels] = (1,1,1,1)

        # Display Images
        blue_axis.imshow(blue_frame)
        violet_axis.imshow(violet_frame)

        # Remove Axis
        blue_axis.axis('off')
        violet_axis.axis('off')

        figure_1.canvas.draw()

        # Write To Video
        canvas.draw()
        buf = canvas.buffer_rgba()
        image_from_plot = np.asarray(buf)
        image_from_plot = cv2.cvtColor(image_from_plot, cv2.COLOR_RGB2BGR)
        video.write(image_from_plot)


        plt.clf()

    cv2.destroyAllWindows()
    video.release()







base_directory = r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_14_Switching_Imaging"
output_directory = r"/home/matthew/Documents/widefield_fMRI_PhD_Meeting"
view_heamodynamic_signals(base_directory, output_directory)