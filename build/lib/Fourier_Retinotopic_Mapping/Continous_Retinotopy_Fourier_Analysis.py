import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
import os
import tables
from scipy import signal, ndimage, stats
from sklearn.neighbors import KernelDensity
import cv2
from matplotlib import gridspec
import sys
import numpy as np
import math
from scipy import fftpack
from math import log, pi
from matplotlib import pyplot as plt
from skimage import color, morphology, measure
import skimage.segmentation as seg

import Retinotopy_Utils

def get_intervals(trial_list):

    number_of_trials = np.shape(trial_list)[0]
    number_of_onsets = np.shape(trial_list)[1]

    print("Number of trials", number_of_trials)
    print("Number of onsets", number_of_onsets)

    intervals = []
    for trial in range(number_of_trials):
        for onset_index in range(1, number_of_onsets):
            interval = trial_list[trial, onset_index] - trial_list[trial, onset_index-1]
            intervals.append(interval)

    return intervals





def create_map(base_directory, values, filter_width=2):
    indicies, image_height, image_width = Re(base_directory)
    image = Retinotopy_Utils.create_image_from_data(values, indicies, image_height, image_width)
    image = np.nan_to_num(image)
    image = np.clip(image, a_min=np.percentile(image, 0.1), a_max=np.percentile(image, 99.9))
    image = ndimage.gaussian_filter(image, filter_width)
    return image


"""
def perform_fourrier_mapping_all_trials(base_directory, onsets, preprocessed_data):

    power_map_list = []
    phase_map_list = []

    for trial in onsets:
        number_of_cycles = len(trial)
        intervals = get_intervals(trial)
        mean_interval = np.int(np.mean(intervals))

        print("Intervals", intervals)
        print("Mean interval", mean_interval)

        data = preprocessed_data[trial[0]:trial[-1] + mean_interval]

        phase_map, power_map = perform_fourrier_mapping_trial(base_directory, data, number_of_cycles)

        power_map_list.append(power_map)
        phase_map_list.append(phase_map)

    power_map_list = np.array(power_map_list)
    phase_map_list = np.array(phase_map_list)

    mean_power_map = np.mean(power_map_list, axis=0)
    mean_phase_map = np.mean(phase_map_list, axis=0)

    plt.imshow(mean_power_map, cmap='plasma')
    plt.show()

    plt.imshow(mean_phase_map, cmap='hsv')
    plt.show()

    return mean_power_map, mean_phase_map
"""


def get_average_movie(base_directory, onsets, preprocessed_data):

    # Get Number Of Trials
    number_of_trials = np.shape(onsets)[0]
    number_of_pixels = np.shape(preprocessed_data)[1]

    # Get Mean Interval
    intervals = get_intervals(onsets)
    mean_interval = int(np.mean(intervals))

    # Get Trial Lengths
    trial_length_list = []
    for trial in onsets:
        trial_length = (trial[-1] + mean_interval) - trial[0]
        trial_length_list.append(trial_length)
    mean_trial_length = int(np.median(trial_length_list))

    # Get Average Movie
    data_array = np.zeros((number_of_trials, mean_trial_length, number_of_pixels))
    for trial in range(number_of_trials):
        trial_start = onsets[trial][0]
        data = preprocessed_data[trial_start:trial_start + mean_trial_length]
        data_array[trial] = data

    mean_movie = np.mean(data_array, axis=0)

    # Turn Flattend Movie 3D
    number_of_frames = np.shape(mean_movie)[0]
    indicies, image_height, image_width = Retinotopy_Utils.load_downsampled_mask(base_directory)

    mean_movie_3d = np.zeros((number_of_frames, image_height, image_width))

    for frame in range(number_of_frames):
        frame_2d = mean_movie[frame]
        frame_3d = Retinotopy_Utils.create_image_from_data(frame_2d, indicies, image_height, image_width)
        mean_movie_3d[frame] = frame_3d

    return mean_movie_3d



def generatePhaseMap2(movie, cycles, isReverse=False, isPlot=False):

    '''
    generating phase map of a 3-d movie, on the frequency defined by cycles.
    the movie should have the same length of 'cycles' number of cycles.
    '''

    if isReverse:
        movie = np.amax(movie) - movie

    spectrumMovie = np.fft.fft(movie, axis=0)

    # generate power movie
    powerMovie = (np.abs(spectrumMovie) * 2.) / np.size(movie, 0)
    powerMap = np.abs(powerMovie[cycles, :, :])

    # generate phase movie
    phaseMovie = np.angle(spectrumMovie)
    # phaseMap = phaseMovie[cycles,:,:]
    phaseMap = -1 * phaseMovie[cycles, :, :]
    phaseMap = phaseMap % (2 * np.pi)

    if isPlot == True:
        plt.figure()
        plotMap = 180 * (phaseMap / np.pi)
        plt.imshow(plotMap,
                   aspect='equal',
                   cmap='hsv',
                   vmax=360,
                   vmin=0,
                   interpolation='nearest')
        plt.colorbar()
        plt.show()

    return phaseMap, powerMap




def visualSignMap(phasemap1, phasemap2):

    gradmap1 = np.gradient(phasemap1)
    gradmap2 = np.gradient(phasemap2)

    # gradmap1 = ni.filters.median_filter(gradmap1,100.)
    # gradmap2 = ni.filters.median_filter(gradmap2,100.)

    graddir1 = np.zeros(np.shape(gradmap1[0]))
    # gradmag1 = np.zeros(np.shape(gradmap1[0]))

    graddir2 = np.zeros(np.shape(gradmap2[0]))
    # gradmag2 = np.zeros(np.shape(gradmap2[0]))

    for i in range(phasemap1.shape[0]):
        for j in range(phasemap2.shape[1]):
            graddir1[i, j] = math.atan2(gradmap1[1][i, j], gradmap1[0][i, j])
            graddir2[i, j] = math.atan2(gradmap2[1][i, j], gradmap2[0][i, j])

            # gradmag1[i,j] = np.sqrt((gradmap1[1][i,j]**2)+(gradmap1[0][i,j]**2))
            # gradmag2[i,j] = np.sqrt((gradmap2[1][i,j]**2)+(gradmap2[0][i,j]**2))

    vdiff = np.multiply(np.exp(1j * graddir1), np.exp(-1j * graddir2))

    areamap = np.sin(np.angle(vdiff))

    return areamap


def overlay_visual_sign_map(base_directory, max_projection, sign_map):

    max_projection = np.divide(max_projection, np.percentile(max_projection, 99))


    save_directory = base_directory + "/Maps/"
    alpha = 0.6

    rows, cols = np.shape(max_projection)

    # Construct a colour image to superimpose
    color_mask = np.zeros((rows, cols, 3))

    absolute_sign_map = np.abs(sign_map)
    threshold = np.percentile(absolute_sign_map, 90)
    max_sign = np.max(sign_map)
    scaled_sign_map = np.divide(sign_map, max_sign)

    # Create Sign Map Array
    sign_map_array = np.zeros(np.shape(max_projection))

    for y in range(rows):
        for x in range(cols):
            pixel_sign = sign_map[y, x]
            abs_pixel_sign  = np.abs(pixel_sign)

            if abs_pixel_sign > threshold:

                if pixel_sign > 0:
                    color_mask[y, x] = [abs_pixel_sign, 0, 0]  # Red block
                    sign_map_array[y, x] = 1
                else:
                    color_mask[y, x] = [0, 0, abs_pixel_sign]  # Blue block
                    sign_map_array[y, x] = -1


    # Construct RGB version of grey-level image
    img_color = np.dstack((max_projection, max_projection, max_projection))

    # Convert the input image and color mask to Hue Saturation Value (HSV)
    # colorspace
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)

    # Display the output
    f, (ax0, ax1, ax2) = plt.subplots(1, 3,
                                      subplot_kw={'xticks': [], 'yticks': []})
    ax0.imshow(max_projection, cmap=plt.cm.gray)
    ax1.imshow(color_mask)
    ax2.imshow(img_masked)
    plt.savefig(save_directory + "/Three_Overlay.png")
    plt.close()

    # Create Single Overlay Map
    plt.imshow(img_masked)
    plt.savefig(save_directory + "Visual_Areas_Overlay.png")
    plt.close()

    # Save Raw Overlay
    np.save(save_directory + "/_Overlay_Array.npy", img_masked)
    np.save(save_directory + "/_Sign_Map_Array.npy", sign_map_array)

def plot_maps(base_directory, horizontal_power_map, horizontal_phase_map, vertical_power_map, vertical_phase_map, sign_map):

    # Create Map Save Directory
    save_directory = base_directory + "/Maps/"
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Load Max Projection
    max_projection = np.load(base_directory + "/max_projection.npy")

    # Plot Individual Maps
    plt.imshow(horizontal_power_map, cmap='plasma', vmin=0, vmax=np.percentile(horizontal_power_map, 99))
    plt.savefig(save_directory + "horizontal_power_map.png")
    plt.close()

    plt.imshow(horizontal_phase_map, cmap='hsv')
    plt.savefig(save_directory + "horizontal_phase_map.png")
    plt.close()

    plt.imshow(vertical_power_map, cmap='plasma', vmin=0, vmax=np.percentile(vertical_power_map, 99))
    plt.savefig(save_directory + "vertical_power_map.png")
    plt.close()

    plt.imshow(vertical_phase_map, cmap='hsv')
    plt.savefig(save_directory + "vertical_phase_map.png")
    plt.close()

    plt.imshow(max_projection, cmap='gray')
    plt.savefig(save_directory + "max_projection.png")
    plt.close()

    plt.imshow(sign_map, cmap='jet', vmin=-1*np.max(np.abs(sign_map)), vmax=np.max(np.abs(sign_map)))
    plt.savefig(save_directory + "sign_map.png")
    plt.close()


    # Plot Combined Map
    figure_1 = plt.figure()
    horizontal_power_axis   = figure_1.add_subplot(2, 3, 1)
    horizontal_phase_axis   = figure_1.add_subplot(2, 3, 4)
    vertical_power_axis     = figure_1.add_subplot(2, 3, 2)
    vertical_phase_axis     = figure_1.add_subplot(2, 3, 5)
    max_projection_axis     = figure_1.add_subplot(2, 3, 3)
    sign_map_axis           = figure_1.add_subplot(2, 3, 6)

    horizontal_power_axis.set_title("Horizontal Power Map")
    horizontal_phase_axis.set_title("Horizontal Phase Map")
    vertical_power_axis.set_title("Vertical Power Map")
    vertical_phase_axis.set_title("Vertical Phase Map")
    max_projection_axis.set_title("Max Projection")
    sign_map_axis.set_title("Sign Map")

    horizontal_power_axis.axis('off')
    horizontal_phase_axis.axis('off')
    vertical_power_axis.axis('off')
    vertical_phase_axis.axis('off')
    max_projection_axis.axis('off')
    sign_map_axis.axis('off')

    horizontal_power_axis.imshow(horizontal_power_map, cmap='plasma', vmin=0, vmax=np.percentile(horizontal_power_map, 99))
    horizontal_phase_axis.imshow(horizontal_phase_map, cmap='jet')
    vertical_power_axis.imshow(vertical_power_map, cmap='plasma', vmin=0, vmax=np.percentile(vertical_power_map, 99))
    vertical_phase_axis.imshow(vertical_phase_map, cmap='jet')
    max_projection_axis.imshow(max_projection, cmap='gray')
    print("Sign min", np.min(sign_map))
    print("Sign max", np.max(sign_map))
    sign_magnitude = np.max(np.abs(sign_map))
    sign_map_axis.imshow(sign_map, cmap='jet', vmin=-sign_magnitude, vmax=sign_magnitude)


    plt.savefig(save_directory + "/Fourrier_Mapping.png")
    plt.close()



def perform_fourier_analysis(base_directory):


    # Load Data
    preprocessed_data_file_location = os.path.join(base_directory, "Downsampled_Delta_F.h5")

    # Load Onsets
    horizontal_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Horizontal_Frame_Onsets.npy"))
    vertical_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Vertical_Frame_Onsets.npy"))

    # Load Full Data
    preprocessed_data_file = tables.open_file(preprocessed_data_file_location, mode='r')
    preprocessed_data = preprocessed_data_file.root['Data']

    # Get Save Directory
    save_directory = os.path.join(base_directory, "Stimuli_Evoked_Responses")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Get Average Movies
    horizontal_average_movie = get_average_movie(base_directory, horizontal_onsets, preprocessed_data)
    vertical_average_movie = get_average_movie(base_directory, vertical_onsets, preprocessed_data)
    np.save(os.path.join(save_directory, "horizontal_average_movie.npy"), horizontal_average_movie)
    np.save(os.path.join(save_directory, "vertical_average_movie.npy"), vertical_average_movie)
  
    horizontal_average_movie = np.load(os.path.join(save_directory, "horizontal_average_movie.npy"))
    vertical_average_movie = np.load(os.path.join(save_directory, "vertical_average_movie.npy"))

    # Perform Fourier Mapping
    number_of_cycles = np.shape(horizontal_onsets)[1]
    horizontal_phase_map, horizontal_power_map = generatePhaseMap2(horizontal_average_movie, number_of_cycles)
    vertical_phase_map, vertical_power_map = generatePhaseMap2(vertical_average_movie, number_of_cycles)

    np.save(os.path.join(save_directory, "Horizontal_Phase_Map"), horizontal_phase_map)
    np.save(os.path.join(save_directory, "Horizontal_Power_Map"), horizontal_power_map)
    np.save(os.path.join(save_directory, "Vertical_Phase_Map"), vertical_phase_map)
    np.save(os.path.join(save_directory, "Vertical_Power_Map"), vertical_power_map)

    # Plot Sign Maps
    horizontal_phase_map = np.load(os.path.join(save_directory, "Horizontal_Phase_Map.npy"))
    horizontal_power_map = np.load(os.path.join(save_directory, "Horizontal_Power_Map.npy"))
    vertical_phase_map = np.load(os.path.join(save_directory, "Vertical_Phase_Map.npy"))
    vertical_power_map = np.load(os.path.join(save_directory, "Vertical_Power_Map.npy"))

    # Create Visual Sign Map
    sign_map = visualSignMap(horizontal_phase_map, vertical_phase_map)

    # Smooth Sign Map
    sign_map = ndimage.gaussian_filter(sign_map, sigma=3)

    # Threshold Sign Map
    sign_standard_deviation = np.std(sign_map)
    threshold = 1.5 * sign_standard_deviation
    thresholded_sign_map = np.where(abs(sign_map) > threshold, sign_map, 0)
    np.save(os.path.join(save_directory, "Thresholded_Sign_Map.npy"), thresholded_sign_map)

    # View Maps
    plot_maps(base_directory, horizontal_power_map, horizontal_phase_map, vertical_power_map, vertical_phase_map, thresholded_sign_map)

    # Overlay Maps
    example_image = np.load(os.path.join(base_directory, "Blue_Example_Image.npy"))
    overlay_visual_sign_map(base_directory, example_image, thresholded_sign_map)

    #binarise_image
    thresholded_sign_map = ndimage.gaussian_filter(thresholded_sign_map, sigma=2)
    threshold = 0.1
    positive_binarised_sign_map = np.where(thresholded_sign_map > threshold, 1, 0)
    negative_binarised_sign_map = np.where(thresholded_sign_map < -1 * threshold, -1, 0)
    binarised_map = np.add(positive_binarised_sign_map, negative_binarised_sign_map)

    # Detect Contours
    contours = measure.find_contours(abs(binarised_map), level=0.8)
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
    plt.title("Contours")
    plt.imshow(np.load(os.path.join(base_directory, "Blue_Example_Image.npy")), cmap="Greys_r")
    plt.savefig(base_directory + "/Maps/Contours")
    plt.close()

