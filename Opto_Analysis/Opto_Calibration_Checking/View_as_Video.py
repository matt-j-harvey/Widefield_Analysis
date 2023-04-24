import numpy as np
import matplotlib.pyplot as plt
import tables
import os
import cv2
from tqdm import tqdm

def create_calibration_sample_video(base_directory):

    # Load Camera Data
    camera_data_file = os.path.join(base_directory, "Downsampled_Delta_F.h5")
    camera_data_file_container = tables.open_file(camera_data_file, mode='r')
    camera_data = camera_data_file_container.root["Data"]
    print("Camera Data Shape", np.shape(camera_data))

    sample_length = 10000
    data_sample = camera_data[0:sample_length]
    data_sample = np.divide(data_sample, 65536)
    data_sample = np.multiply(data_sample, 255)
    data_sample = np.ndarray.astype(data_sample, np.uint8)
    data_sample = np.reshape(data_sample, (sample_length, 600, 608))
    print("Data Sample", np.shape(data_sample))

    # Create Video File
    video_name = os.path.join(base_directory, "Blue_Data_Example_Video.avi")
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, video_codec, frameSize=(608, 600), fps=30)  # 0, 12

    for frame_index in tqdm(range(sample_length)):

        # Extract Frames
        frame = data_sample[frame_index]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()


create_calibration_sample_video(r"/media/matthew/External_Harddrive_3/Opto_FSM_Calibration/Calibration/Opto_Test_2023_04_11")