import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from Behaviour_Analysis import Behaviour_Utils

def extract_video_data(video_file):

    # Open Video File
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #frame_width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    even_frame_intensities = []
    odd_frame_intensities = []

    for frame_index in tqdm(range(frameCount), desc="Extracting Camera Data"):

        # Read Next Frame
        ret, current_frame = cap.read()
        current_frame = current_frame[:, :, 0]
        frame_intensity = np.mean(current_frame)

        if frame_index % 2 == 0:
            even_frame_intensities.append(frame_intensity)
        else:
            odd_frame_intensities.append(frame_intensity)

    cap.release()

    return frameCount, even_frame_intensities, odd_frame_intensities



def get_trigger_data(base_directory):
    frame_data = np.loadtxt(os.path.join(base_directory, "CameraTriggers.csv"))
    frame_triggers = Behaviour_Utils.get_step_onsets(frame_data, threshold=5000)
    number_of_frame_triggers = len(frame_triggers)
    return number_of_frame_triggers



base_directory = r"/home/matthew/Documents/Fillipe_Camera_Check_2022_02_2023"

# Get Number Of Triggers
number_of_frame_triggers = get_trigger_data(base_directory)

# Get Number Of Frames
number_of_cam_1_frames, cam_1_even_frame_intensities, cam_1_odd_frame_intensities = extract_video_data(os.path.join(base_directory, "test_2023-02-22-16-27-42_cam_1.mp4"))
number_of_cam_2_frames, cam_2_even_frame_intensities, cam_2_odd_frame_intensities = extract_video_data(os.path.join(base_directory, "test_2023-02-22-16-27-42_cam_2.mp4"))



print("Number of frame triggers", number_of_frame_triggers)
print("Number of cam 1 frames", number_of_cam_1_frames)
print("Number of cam 2 frames", number_of_cam_2_frames)


plt.plot(cam_1_even_frame_intensities)
plt.plot(cam_1_odd_frame_intensities)
plt.show()