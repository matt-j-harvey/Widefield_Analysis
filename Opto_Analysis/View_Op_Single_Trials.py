import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec
from matplotlib.pyplot import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg

import os
from scipy.io import loadmat
import sys
import tables
import cv2
from tqdm import tqdm

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions



def get_stim_log_file(base_directory):

    file_list = os.listdir(base_directory)

    for file in file_list:
        if file[0:10] == 'opto_stim_':
            return file


def get_trial_activity(delta_f_matrix, onset, start_window, stop_window, indicies, image_height, image_width):

    trial_start = onset + start_window
    trial_stop = onset + stop_window
    trial_data = delta_f_matrix[trial_start:trial_stop]

    reconstruced_data = []
    for frame in trial_data:
        frame = Widefield_General_Functions.create_image_from_data(frame, indicies, image_height, image_width)
        reconstruced_data.append(frame)

    return reconstruced_data


def get_roi_outlines(roi_masks, number_of_stimuli):

    # Get ROI Edges
    roi_mask_edge_list = []
    dilated_roi_mask_edge_list = []
    for roi_index in range(number_of_stimuli):
        roi_mask = roi_masks[roi_index]

        roi_mask = np.flip(roi_mask, axis=0)
        edges = cv2.Canny(roi_mask, 0.5, 1)

        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel=kernel, iterations=2)

        edge_indicies = np.nonzero(edges)
        dilated_edge_indicies = np.nonzero(dilated_edges)

        roi_mask_edge_list.append(edge_indicies)
        dilated_roi_mask_edge_list.append(dilated_edge_indicies)

    return roi_mask_edge_list, dilated_roi_mask_edge_list


def view_single_trial_videos(base_directory):

    # Settings
    trials_to_display = 10
    start_window = -20
    stop_window = 100
    x_values = np.multiply(list(range(start_window, stop_window)), 36)
    number_of_timepoints = stop_window - start_window

    # Get Number Of Stimuli
    stim_log_file = get_stim_log_file(base_directory)
    stim_log = loadmat(os.path.join(base_directory, stim_log_file))
    stim_log = stim_log['opto_session_data']
    stimuli = stim_log[0][0][0]
    number_of_stimuli = len(np.unique(stimuli))
    print("Number of stimuli", number_of_stimuli)

    # Load ROI Masks
    roi_masks = stim_log[1][0][0]

    # Get ROI Mask Outlines
    roi_mask_edge_list, dilated_roi_mask_edge_list = get_roi_outlines(roi_masks, number_of_stimuli)

    # Get Mask Details
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    # Load Delta F File
    delta_f_file = os.path.join(base_directory, "Delta_F.h5")
    delta_f_file_object = tables.open_file(delta_f_file)
    delta_f_matrix = delta_f_file_object.root.Data

    # Create Colourmaps
    delta_f_colourmap = cm.ScalarMappable(norm=None, cmap='inferno')

    # Iterate Through Each Stimuli
    for stimuli_index in range(number_of_stimuli):
        print("Stimuli: ", stimuli_index, " of ", number_of_stimuli)
        # Create Empty List To Hold Trial Data
        stimuli_data = []

        # Load onsets
        stimuli_directory = os.path.join(base_directory, "Stimuli_" + str(int(stimuli_index + 1)))
        onset_list = np.load(os.path.join(stimuli_directory, "Stimuli_Onsets.npy"))

        # Get Data For Each Onset
        for onset in onset_list:
            trial_data = get_trial_activity(delta_f_matrix, onset, start_window, stop_window, indicies, image_height, image_width)
            stimuli_data.append(trial_data)

        # Convert TO Array
        stimuli_data = np.array(stimuli_data)
        print("Stimuli Data", np.shape(stimuli_data))

        # Plot These
        video_name = os.path.join(stimuli_directory, "stimuli_movie.avi")
        video_codec = cv2.VideoWriter_fourcc(*'DIVX')
        frame_width = 640
        frame_height = 480
        video = cv2.VideoWriter(video_name, video_codec, frameSize=(frame_width, frame_height), fps=30)  # 0, 12

        stimuli_roi_mask_indicies = roi_mask_edge_list[stimuli_index]
        stimuli_roi_mask_indicies_dilated = dilated_roi_mask_edge_list[stimuli_index]

        # Set Custom Colourmap
        vmin = 0
        vmax = np.percentile(stimuli_data, q=99)

        # Create Figure
        n_rows = 2
        n_columns = 5

        figure_1 = plt.figure()
        canvas_1 = FigureCanvasAgg(figure_1)

        for timepoint in tqdm(range(number_of_timepoints)):
            plt.clf()

            figure_1.suptitle(str(x_values[timepoint]) + "ms")

            for trial_index in range(trials_to_display):

                # Create Axis
                trial_axis = figure_1.add_subplot(n_rows, n_columns, trial_index + 1)

                # load Image
                trial_image = stimuli_data[trial_index, timepoint]
                trial_image = delta_f_colourmap.to_rgba(trial_image)

                # Add ROI Edges
                trial_image[stimuli_roi_mask_indicies] = [1, 1, 1, 1]
                trial_image[stimuli_roi_mask_indicies_dilated] = [0, 0, 0, 1]

                # Display image
                trial_axis.imshow(trial_image)

            # Write Frame To Video
            canvas_1.draw()
            buf = canvas_1.buffer_rgba()
            colored_image = np.asarray(buf)
            image = np.ndarray.astype(colored_image, np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video.write(image)

        cv2.destroyAllWindows()
        video.release()





base_directory = r"/media/matthew/External_Harddrive_1/Opto_Test/KVIP25.5H/2022_07_26_Opto_Test_No_Filter"
base_directory = r"/media/matthew/External_Harddrive_1/Opto_Test/KVIP25.5H/2022_07_27_Opto_Test_Grid"

view_single_trial_videos(base_directory)