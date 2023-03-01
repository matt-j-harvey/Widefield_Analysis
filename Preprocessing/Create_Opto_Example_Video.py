import numpy as np
import matplotlib.pyplot as plt
import os
from Widefield_Utils import widefield_utils
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.pyplot import cm
from matplotlib.colors import Normalize
from tqdm import tqdm
import cv2

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def reconstruct_stimulus(activity_matrix, session_indicies, session_height, session_width, common_indicies, common_height, common_width, within_mouse_alignment_dict):

    reconstructed_stimulus = []
    for frame in activity_matrix:

        # Reconstruct Frame
        frame = widefield_utils.create_image_from_data(frame,session_indicies, session_height, session_width)

        # Align Within Mouse
        frame = widefield_utils.transform_image(frame, within_mouse_alignment_dict)

        # Apply Tight Mask
        frame = np.reshape(frame, (common_height * common_width))
        template = np.zeros(common_height * common_width)
        template[common_indicies] = frame[common_indicies]
        template = np.reshape(template, (common_height, common_width))
        reconstructed_stimulus.append(template)


    # Apply Cheeky Moving Average
    reconstructed_stimulus = np.array(reconstructed_stimulus)
    print("Reconstructed Stimulus Shape", np.shape(reconstructed_stimulus))

    reconstructed_stimulus = moving_average(reconstructed_stimulus)
    print("Reconstructed Stimulus Shape", np.shape(reconstructed_stimulus))

    return reconstructed_stimulus


def create_opto_example_video(base_directory, stimuli_list, output_directory):

    # Load Within Mouse Alignment Dict
    within_mouse_alignment_dict = np.load(os.path.join(base_directory, "Within_Mouse_Alignment_Dictionary.npy"), allow_pickle=True)[()]

    # Load Session Mask
    session_indicies, session_height, session_width = widefield_utils.load_downsampled_mask(base_directory)

    # Load Common Mask
    common_indicies, common_height, common_width = widefield_utils.load_tight_mask()

    reconstructed_stimuli_images = []

    for stimuli in stimuli_list:

        # Load Average Response
        average_response = np.load(os.path.join(base_directory, "Stimuli_" + str(stimuli), "mean_response.npy"))

        # Reconstruct Response
        average_response = reconstruct_stimulus(average_response, session_indicies, session_height, session_width, common_indicies, common_height, common_width, within_mouse_alignment_dict)
        reconstructed_stimuli_images.append(average_response)


    background_pixels = widefield_utils.get_background_pixels(common_indicies, common_height, common_width )

    # Create Video File
    video_name = os.path.join(output_directory, "Opto_Widefield.avi")
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, video_codec, frameSize=(1500, 500), fps=30)  # 0, 12

    sample_length = np.shape(reconstructed_stimuli_images[0])[0]
    number_of_stimuli = len(stimuli_list)
    print("Sample Length", sample_length)

    # Create Figure
    figure_1 = plt.figure(figsize=(15, 5))
    canvas = FigureCanvasAgg(figure_1)

    # Create Colourmap
    widefield_colourmap = widefield_utils.get_musall_cmap()
    widefield_colourmap = plt.cm.ScalarMappable(norm=Normalize(vmin=-0.05, vmax=0.05), cmap=widefield_colourmap)

    time_values = list(range(sample_length))
    time_values = np.multiply(time_values, 36)
    time_values = np.subtract(time_values, 3600)

    for frame_index in tqdm(range(sample_length)):
        rows = 1
        columns = number_of_stimuli

        for stimuli_index in range(number_of_stimuli):

            # Create Axis
            brain_axis = figure_1.add_subplot(rows, columns, stimuli_index + 1)

            # Extract Frames
            brain_frame = reconstructed_stimuli_images[stimuli_index][frame_index]

            # Set Colours
            brain_frame = widefield_colourmap.to_rgba(brain_frame)

            # Remove Background
            brain_frame[background_pixels] = (1, 1, 1, 1)

            # Display Images
            brain_axis.imshow(brain_frame)

            # Remove Axis
            brain_axis.axis('off')

        if frame_index < 100:
            figure_1.suptitle("Time: " + str(time_values[frame_index]) + "ms" + " Stim Off")

        if frame_index > 100:
            figure_1.suptitle("Time: " + str(time_values[frame_index]) + "ms" + " Stim On")

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

base_directory = r"/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.1D/2022_12_12_Opto_Test_Filter"
output_directory = r"/home/matthew/Documents/widefield_fMRI_PhD_Meeting"
stimuli_list = [1,6,9]
create_opto_example_video(base_directory, stimuli_list, output_directory)

