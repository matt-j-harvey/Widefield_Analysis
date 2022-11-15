import h5py
import numpy as np
import tables
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import Preprocessing_Utils


def convert_df_to_tables(base_directory, output_directory):

    # Load Processed Data
    delta_f_file_location = os.path.join(base_directory, "300_delta_f.hdf5")
    delta_f_file = h5py.File(delta_f_file_location, mode='r')
    processed_data = delta_f_file["Data"]
    number_of_frames, number_of_pixels = np.shape(processed_data)

    # Create Tables File
    output_file = os.path.join(output_directory, "Downsampled_Delta_F.h5")
    output_file_container = tables.open_file(output_file, mode="w")
    output_e_array = output_file_container.create_earray(output_file_container.root, 'Data', tables.Float32Atom(), shape=(0, number_of_pixels), expectedrows=number_of_frames)

    # Define Chunking Settings
    preferred_chunk_size = 30000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = Preprocessing_Utils.get_chunk_structure(preferred_chunk_size, number_of_frames)

    for chunk_index in tqdm(range(number_of_chunks)):

        # Get Selected Indicies
        chunk_start = int(chunk_starts[chunk_index])
        chunk_stop = int(chunk_stops[chunk_index])

        chunk_data = processed_data[chunk_start:chunk_stop]

        for frame in chunk_data:
            output_e_array.append([frame])

        output_file_container.flush()

    delta_f_file.close()
    output_file_container.close()
