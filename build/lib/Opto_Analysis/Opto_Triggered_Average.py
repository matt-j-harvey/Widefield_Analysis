import numpy as np
import tables
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import h5py
from scipy.io import loadmat
from scipy.ndimage.filters import uniform_filter1d
from scipy import ndimage
from skimage.transform import resize
from skimage.measure import find_contours
from matplotlib.colors import Normalize
from sklearn.decomposition import TruncatedSVD
import cv2

import Opto_Utils


def get_stimuli_indexes(stimuli_pool, selected_stimulus):

    stimuli_indexes = []
    for x in range(len(stimuli_pool)):
        stimuli = stimuli_pool[x]
        if stimuli == selected_stimulus:
            stimuli_indexes.append(x)

    return stimuli_indexes

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def get_stim_log_file(base_directory):

    file_list = os.listdir(base_directory)

    for file in file_list:
        if file[0:10] == 'opto_stim_':
            return file

def match_frames(stimuli_onsets, frame_onsets):

    nearest_frames = []
    frame_onsets_array = np.array(frame_onsets)

    for onset in stimuli_onsets:
        nearest_frame_onset = find_nearest(frame_onsets_array, onset)
        nearest_frame_index = frame_onsets.index(nearest_frame_onset)
        nearest_frames.append(nearest_frame_index)

    nearest_frames = np.array(nearest_frames)
    return nearest_frames


def check_blue_violet_files(base_directory, blue_file, violet_file):

    # Load Blue Data
    blue_filepath = os.path.join(base_directory, blue_file)
    blue_file_container = h5py.File(blue_filepath, 'r')
    blue_matrix = blue_file_container['Data']

    # Load Violet Data
    violet_filepath = os.path.join(base_directory, violet_file)
    violet_file_container = h5py.File(violet_filepath, 'r')
    violet_matrix = violet_file_container['Data']

    blue_image = blue_matrix[:, 0]
    violet_image = violet_matrix[:, 0]

    blue_image = np.ndarray.reshape(blue_image, (600, 608))
    violet_image = np.ndarray.reshape(violet_image, (600, 608))

    figure_1 = plt.figure()
    blue_axis = figure_1.add_subplot(1,2,1)
    violet_axis = figure_1.add_subplot(1,2,2)

    blue_axis.set_title("Blue")
    blue_axis.imshow(blue_image)

    violet_axis.set_title("Violet")
    violet_axis.imshow(violet_image)

    plt.show()




def get_average_raw_video(base_directory, blue_file, number_of_stimuli, stim_pool, opto_onset_frames):

    # Load Blue Data
    blue_filepath = os.path.join(base_directory, blue_file)
    blue_file_container = h5py.File(blue_filepath, 'r')
    blue_matrix = blue_file_container['Data']
    blue_matrix = np.array(blue_matrix)
    blue_matrix = np.transpose(blue_matrix)

    print("Blue matrix shape", np.shape(blue_matrix))

    for stimuli_index in range(1, number_of_stimuli+1):

        output_directory = os.path.join(base_directory, "Blue_Only_Stimuli_" + str(stimuli_index))
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        stimuli_indexes = get_stimuli_indexes(stim_pool, stimuli_index)
        stimuli_onsets = opto_onset_frames[stimuli_indexes]

        # Create Tensor
        trial_frames = Widefield_General_Functions.get_selected_widefield_frames(stimuli_onsets, -10, 100)
        trial_tensor = Widefield_General_Functions.get_selected_widefield_data(trial_frames, blue_matrix)
        print("Trial Tensor Shape", np.shape(trial_tensor))

        # Get Mean Response
        mean_response = np.mean(trial_tensor, axis=0)

        print("Trial Tensor Shape", np.shape(mean_response))



        # Get Baseline
        baseline = np.mean(a=mean_response[0:10], axis=0)

        # Get Delta
        delta_f = np.subtract(mean_response, baseline)

        vmax = np.max(np.abs(delta_f))

        plt.ion()
        for frame_index in range(0, 110):
            plt.title(str(frame_index-10))
            frame_data = delta_f[frame_index]
            frame_image = np.ndarray.reshape(frame_data, (600, 608))
            plt.imshow(frame_image, cmap='bwr', vmin=-1 * vmax, vmax=vmax)
            plt.draw()
            plt.pause(0.1)
            plt.savefig(os.path.join(output_directory, str(frame_index).zfill(3) + ".png"))
            plt.clf()


def transform_image(transformation_details, image):

    # Load Variables From Dictionary
    rotation = transformation_details['rotation']
    x_shift = transformation_details['x_shift']
    y_shift = transformation_details['y_shift']

    # Rotate Mask
    image = ndimage.rotate(image, rotation, reshape=False)

    # Translate
    image = np.roll(a=image, axis=0, shift=y_shift)
    image = np.roll(a=image, axis=1, shift=x_shift)

    # Re-Binarise
    #image = np.where(image > 0.1, 1, 0)
    #image = np.ndarray.astype(image, int)

    return image


def plot_composite_activations(base_directory):

    # Load Stim Log File
    stim_log_file = get_stim_log_file(base_directory)
    stim_log = loadmat(os.path.join(base_directory, stim_log_file))
    stim_log = stim_log['opto_session_data']

    stim_pool       = stim_log[0][0][0]
    roi_masks       = stim_log[1][0][0]
    roi_intensities = stim_log[2][0][0]
    number_of_stimuli = np.shape(roi_masks)[0]

    # Load Activity Matricies
    activity_matricies_list = []
    for iterator_variable in range(number_of_stimuli):
        stimuli_directory = os.path.join(base_directory, "Stimuli_" + str(iterator_variable + 1))
        activity_matrix = np.load(os.path.join(stimuli_directory, "mean_response.npy"))
        print(np.shape(activity_matrix))
        activity_matricies_list.append(activity_matrix)

    # Get Trial Structure
    number_of_timepoints = np.shape(activity_matricies_list[0])[0]

    # Load Mask
    indicies, image_height, image_width = Opto_Utils.load_downsampled_mask(base_directory)

    # Get ROI Edges
    roi_mask_edge_list = []
    dilated_roi_mask_edge_list = []
    for roi_index in range(number_of_stimuli):


        roi_mask = roi_masks[roi_index]


        roi_mask = np.flip(roi_mask, axis=0)

        #roi_mask = transform_image(transformation_dictionary, roi_mask)

        edges = cv2.Canny(roi_mask, 0.5, 1)
        
        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel=kernel, iterations=2)

        edges = resize(edges, (300, 304))
        dilated_edges = resize(dilated_edges, (300, 304))

        roi_mask_edge_list.append(edges)
        dilated_roi_mask_edge_list.append(dilated_edges)

    # Create Save Directory
    save_directory = os.path.join(base_directory, "All_ROIs")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Get Colourmap
    colourmap = Opto_Utils.get_musall_cmap()
    colourmap = cm.ScalarMappable(cmap=colourmap, norm=Normalize(vmin=-0.05, vmax=0.05))

    # Create Figure
    figure_1 = plt.figure(figsize=(10,10))

    [rows, columns] = [3, 3]

    black_rgba_colour = [0,0,0,1]
    white_rgba_colour = [1,1,1,1]

    interval = 36
    timelist = list(range(-100 * interval, 100 * interval,  interval))

    for timepoint in range(number_of_timepoints):
        time_value = timelist[timepoint]

        for stimuli_index in range(number_of_stimuli):

            # Create Axis
            axis = (figure_1.add_subplot(rows, columns, stimuli_index + 1))

            # Get Brain Activity
            baseline = activity_matricies_list[stimuli_index][90:100]
            baseline = np.mean(baseline, axis=0)

            brain_activity = activity_matricies_list[stimuli_index][timepoint]
            brain_image = Opto_Utils.create_image_from_data(brain_activity, indicies, image_height, image_width)

            # Colour Brain Image
            brain_image = colourmap.to_rgba(brain_image)

            # Get ROI Edges
            roi_edges = roi_mask_edge_list[stimuli_index]
            dilated_roi_edges = dilated_roi_mask_edge_list[stimuli_index]

            # Get Edges Indicies
            edge_indicies = np.nonzero(roi_edges)
            dilated_edge_indicies = np.nonzero(dilated_roi_edges)

            brain_image[dilated_edge_indicies] = white_rgba_colour
            brain_image[edge_indicies] = black_rgba_colour

            axis.imshow(brain_image)

            # Remove Axis
            axis.axis('off')

        figure_1.suptitle(str(time_value))

        plt.savefig(os.path.join(save_directory, str(timepoint).zfill(3) + ".png"))
        plt.clf()




    print("Number of stimuli", number_of_stimuli)


    print(roi_intensities)






def get_average_opto_responses(base_directory):

    # Check Correct LED Colour Labels

    # Load Delfa F Data
    delta_f_file = os.path.join(base_directory, "Downsampled_Delta_F.h5")
    delta_f_file_container = tables.open_file(delta_f_file, 'r')
    delta_f_matrix = delta_f_file_container.root.Data
    delta_f_matrix = np.array(delta_f_matrix)
    delta_f_matrix = np.nan_to_num(delta_f_matrix)
    print("Delta F MAtrix Shape", np.shape(delta_f_matrix))

    # Denoise WIth SVD
    model = TruncatedSVD(n_components=200)
    transformed_data = model.fit_transform(delta_f_matrix)
    delta_f_matrix = model.inverse_transform(transformed_data)

    # Load Ai File
    ai_data = Opto_Utils.load_ai_recorder_file(base_directory)

    # Extract Required Traces
    stimuli_dictionary = Opto_Utils.create_stimuli_dictionary()
    opto_trace = ai_data[stimuli_dictionary['Optogenetics']]
    frame_trace = ai_data[stimuli_dictionary["LED 1"]]

    # Get Opto Onsets
    plt.plot(opto_trace, alpha=0.2)
    plt.plot(frame_trace, alpha=0.2)
    plt.show()


    threshold = 2
    opto_onsets = Opto_Utils.get_step_onsets(opto_trace, threshold=threshold, window=1000)
    print("opto onsets", len(opto_onsets))

    plt.plot(opto_trace, alpha=0.2)
    plt.plot(frame_trace, alpha=0.2)
    plt.scatter(opto_onsets, np.multiply(np.ones(len(opto_onsets)), np.max(opto_trace)))
    plt.show()


    # Get Frame Onsets
    frame_onsets = Opto_Utils.get_step_onsets(frame_trace)

    # Get Matching Frames
    opto_onset_frames = match_frames(opto_onsets, frame_onsets)
    print("Opto onset Frames", opto_onset_frames)

    # Split Into Trial Type
    stim_log_file = get_stim_log_file(base_directory)
    print("Stim Log File", stim_log_file)
    stim_log = loadmat(os.path.join(base_directory, stim_log_file))
    print("stim log", stim_log)
    stim_pool = stim_log['opto_session_data'][0][0][0]
    print("Stim pool", stim_pool)

    number_of_stimuli = np.max(stim_pool)
    print("number of stimuli", number_of_stimuli)

    #get_average_raw_video(base_directory, blue_file, number_of_stimuli, stim_pool, opto_onset_frames)

    trial_start = -100
    trial_stop = 100


    for stimuli_index in range(1, number_of_stimuli+1):

        output_directory = os.path.join(base_directory, "Stimuli_" + str(stimuli_index))
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        stimuli_indexes = get_stimuli_indexes(stim_pool, stimuli_index)
        stimuli_onsets = opto_onset_frames[stimuli_indexes]
        print("Stimuli: ", stimuli_index, "Onsets: ", stimuli_onsets)

        # Save Stimuli Onsets
        np.save(os.path.join(output_directory, "Stimuli_Onsets.npy"), stimuli_onsets)

        # Create Tensor
        trial_frames = Opto_Utils.get_selected_widefield_frames(stimuli_onsets, trial_start, trial_stop)
        trial_tensor = Opto_Utils.get_selected_widefield_data(trial_frames, delta_f_matrix)
        print("Trial Tensor Shape", np.shape(trial_tensor))

        # Get Mean Response
        mean_response = np.mean(trial_tensor, axis=0)
        np.save(os.path.join(output_directory, "mean_response.npy"), mean_response)
        np.save(os.path.join(output_directory, "Activity_Tensor.npy"), trial_tensor)

        # Get Mask Details
        indicies, image_height, image_width = Opto_Utils.load_downsampled_mask(base_directory)

        plt.ion()
        colourmap = Opto_Utils.get_musall_cmap()

        for frame_index in range(0,trial_stop - trial_start):
            plt.title(str(frame_index + trial_start))
            frame_data = mean_response[frame_index]
            frame_image = Opto_Utils.create_image_from_data(frame_data, indicies, image_height, image_width)

            plt.imshow(frame_image, cmap=colourmap, vmin=-0.05, vmax=0.05)
            plt.draw()
            plt.pause(0.1)
            plt.savefig(os.path.join(output_directory, str(frame_index).zfill(3) + ".png"))
            plt.clf()



session_list = ["/media/matthew/External_Harddrive_3/Opto_Test/KPGC3.1F/2022_12_12_Opto_Test_Filter"]
for base_directory in session_list:
    get_average_opto_responses(base_directory)
    plot_composite_activations(base_directory)

