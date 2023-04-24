import numpy as np
import tables
from tqdm import tqdm
import os

def get_widefield_filename(base_directory):
    file_list = os.listdir(base_directory)
    for file_name in file_list:
        if "widefield.h5" in file_name:
            return file_name


def unpack_calibration_data(base_directory):

    # Get Widefield Filename
    widefield_filename = get_widefield_filename(base_directory)
    data_file = os.path.join(base_directory, widefield_filename)

    # Open Widefield File
    data_container = tables.open_file(data_file, mode='r')
    blue_data = data_container.root["blue"]
    violet_data = data_container.root["violet"]
    print("data cntainer", data_container)
    print("Blue data", np.shape(blue_data))

    # Get Data Shape
    expected_rows, image_height, image_width = np.shape(blue_data)

    # Create Output Files
    blue_output_file = os.path.join(base_directory, "Downsampled_Delta_F.h5")
    violet_output_file = os.path.join(base_directory, "Downsampled_Delta_F_Violet.h5")

    new_blue_tables_file = tables.open_file(blue_output_file, mode='w')
    new_violet_tables_file = tables.open_file(violet_output_file, mode='w')

    blue_storage = new_blue_tables_file.create_earray(new_blue_tables_file.root, 'Data', tables.UInt16Atom(), shape=(0, image_height * image_width), expectedrows=expected_rows)
    violet_storage = new_violet_tables_file.create_earray(new_violet_tables_file.root, 'Data', tables.UInt16Atom(), shape=(0, image_height * image_width), expectedrows=expected_rows)

    # New Tables File
    for frame_index in tqdm(range(expected_rows)):
        blue_storage.append([np.reshape(blue_data[frame_index], image_height * image_width)])
        if frame_index % 100 == 0:
            new_blue_tables_file.flush()

    # New Tables File
    for frame_index in tqdm(range(expected_rows)):
        violet_storage.append([np.reshape(violet_data[frame_index], image_height * image_width)])
        if frame_index % 100 == 0:
            new_violet_tables_file.flush()


