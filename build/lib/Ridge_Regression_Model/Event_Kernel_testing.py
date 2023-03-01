import numpy as np
import matplotlib.pyplot as plt


def create_event_kernel_from_event_list(event_list, number_of_widefield_frames, preceeding_window=-14, following_window=28):

    kernel_size = following_window - preceeding_window
    design_matrix = np.zeros((number_of_widefield_frames, kernel_size))
    time_window = list(range(preceeding_window, following_window))

    for frame_index in range(number_of_widefield_frames):

        if event_list[frame_index] == 1:

            # Fill In Design Matrix
            for regressor_index in range(kernel_size):
                regressor_time = time_window[regressor_index] + frame_index

                if regressor_time >= 0 and regressor_time < number_of_widefield_frames:
                    design_matrix[regressor_time, regressor_index] = 1

    return design_matrix



number_of_widefield_frames = 100
event_list = np.zeros(number_of_widefield_frames)
event_list[99] = 1
event_kernel = create_event_kernel_from_event_list(event_list, number_of_widefield_frames)

plt.imshow(np.transpose(event_kernel))
plt.show()