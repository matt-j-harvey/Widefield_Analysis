import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from sklearn.decomposition import IncrementalPCA
from pathlib import Path
import sys

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")
import Widefield_General_Functions


def get_bodycam_filename(base_directory):

    file_list = os.listdir(base_directory)

    for file in file_list:
        file_split = file.split('_')
        if file_split[-1] == '1.mp4' and file_split[-2] == 'cam':
            return file

def get_video_details(video_file):

    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return frameCount, frameHeight, frameWidth



def load_video_chunk(video_file, chunk_start, chunk_size):

    # Create List To Hold Video Chunk Data
    video_chunk_data = []

    # Open Video File
    cap = cv2.VideoCapture(video_file)

    # Tell Open CV2 To Start Reading Frames From Chunk Start
    cap.set(1, chunk_start)

    # Read The Required Number Of Frames
    for frame_index in range(chunk_size):
        ret, frame_data = cap.read()
        video_chunk_data.append(frame_data)

    # Close The Video
    cap.release()

    # Turn List Into Array
    video_chunk_data = np.array(video_chunk_data)
    video_chunk_data = video_chunk_data[:, :, :, 0]

    video_height = np.shape(video_chunk_data)[1]
    video_width  = np.shape(video_chunk_data)[2]


    video_chunk_data = np.ndarray.reshape(video_chunk_data, (chunk_size, video_width*video_height))


    return video_chunk_data



# Settings
root_directory = "/media/matthew/Seagate Expansion Drive2/Longitudinal_Analysis/NXAK4.1B/"
session_list = ["2021_02_04_Discrimination_Imaging",
                "2021_02_06_Discrimination_Imaging",
                "2021_02_08_Discrimination_Imaging",
                "2021_02_10_Discrimination_Imaging",
                "2021_02_12_Discrimination_Imaging",
                "2021_02_14_Discrimination_Imaging",
                "2021_02_22_Discrimination_Imaging"]

base_directory = root_directory + session_list[-1]
number_of_components = 30
chunk_size = 5000

# Create Save Directory
save_directory = base_directory + "/Mousecam_SVD"
Widefield_General_Functions.check_directory(save_directory)


# Get Bodycam File Name
bodycam_file_name = get_bodycam_filename(base_directory)
bodycam_file_path = base_directory + "/" + bodycam_file_name
print("Mousecam File name: ", bodycam_file_name)


# Get Video Details
bodycam_frames, video_height, video_width = get_video_details(bodycam_file_path)
print("Frames: ", bodycam_frames, "Video Height", video_height, "Video Width", video_width)


# Get Chunk Structure
number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = Widefield_General_Functions.get_chunk_structure(chunk_size, bodycam_frames)


# Ensure Chunk Sizes Are Always Greater Than The Number Of Components To Allow Us To Fit Our Model
if number_of_components > chunk_size:
    print("Error! Number of components must be less than chunk size!")

else:
    # If Last Chunk Smaller Than Number Of Components - Merge It Into Penultimate Chunk
    if chunk_sizes[-1] <= number_of_components:
        number_of_chunks = number_of_chunks -1
        del chunk_sizes[-1]
        del chunk_start[-1]
        del chunk_stops[-1]

        chunk_sizes[-1] = bodycam_frames - chunk_starts[-1]
        chunk_stops[-1] = bodycam_frames

print("Chunk starts: ", chunk_starts)
print("Chunk Sizes: ", chunk_sizes)

# Create Model
model = IncrementalPCA(n_components=number_of_components)


# Train Model
for chunk_index in range(number_of_chunks):
    print("Fitting Chunk: ", chunk_index, " of ", number_of_chunks)
    chunk_start = chunk_starts[chunk_index]
    chunk_size = chunk_sizes[chunk_index]
    video_data = load_video_chunk(bodycam_file_path, chunk_start, chunk_size)

    print("Video Data Chunk Shape", np.shape(video_data))
    model.partial_fit(video_data)


# Transform Data
transformed_data_list = []
for chunk_index in range(number_of_chunks):
    print("Transforming Chunk: ", chunk_index, " of ", number_of_chunks)
    chunk_start = chunk_starts[chunk_index]
    chunk_size = chunk_sizes[chunk_index]
    video_data = load_video_chunk(bodycam_file_path, chunk_start, chunk_size)
    transformed_data = model.transform(video_data)

    # Add Tansformed Points To List
    for data_point in transformed_data:
        transformed_data_list.append(data_point)


# Get Numpy Arrays Of Model Components and Transformed Data
transformed_data_list = np.array(transformed_data_list)
components = model.components_

# Save Components and Transformed Data
np.save(save_directory + "/components.npy", components)
np.save(save_directory + "/transformed_data.npy", transformed_data)
