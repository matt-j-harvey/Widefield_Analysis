import numpy as np
import h5py
import matplotlib.pyplot as plt

from Widefield_Utils import widefield_utils

def examine_read_noise(blue_filename):

    # Load Blue Data File
    blue_file_container = h5py.File(blue_filename, mode="r")
    blue_data = blue_file_container["Data"]

    # Reshape Sample
    sample_size = 1000
    image_height = 600
    image_width = 608
    sample = blue_data[:, 0:sample_size]

    sample = np.reshape(sample, (image_height, image_width, sample_size))
    sample_mean = np.mean(sample, axis=2)

    #plt.ion()
    for frame_index in range(sample_size):
        #plt.imshow(sample[0:40, 0:40, frame_index])
        sample_frame = sample[:, :, frame_index]
        sample_frame = np.subtract(sample_frame, sample_mean)
        sample_frame = np.divide(sample_frame, sample_mean)

        read_noise = np.mean(sample_frame[:, 0:20], axis=1)
        sample_frame = np.transpose(sample_frame)
        #sample_frame = np.subtract(sample_frame, read_noise)
        sample_frame = np.transpose(sample_frame)

        plt.imshow(sample_frame, vmin=-0.05, vmax=0.05)
        #plt.draw()
        #plt.pause(0.1)
        #plt.clf()
        plt.show()


    print(np.shape(blue_data))


blue_filename = r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_15_Spontaneous/NXAK22.1A_20210914-173905_Blue_Data.hdf5"

examine_read_noise(blue_filename)