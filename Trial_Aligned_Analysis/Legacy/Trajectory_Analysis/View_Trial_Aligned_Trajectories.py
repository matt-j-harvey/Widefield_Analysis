import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import networkx as nx
import cv2
from matplotlib.pyplot import cm
from matplotlib.colors import to_rgb
import os
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist, cdist
from matplotlib import pyplot as plt
from pathlib import Path
from mpl_toolkits import mplot3d
from sklearn.decomposition import IncrementalPCA

#import Create_Activity_Tensor
import Trajectory_Utils


def downsample_mask_further(indicies, image_height, image_width, downsample_size=100):
    template = np.zeros((image_height*image_width))
    template[indicies] = 1
    template = np.reshape(template, (image_height, image_width))
    template = resize(template, (downsample_size, downsample_size))
    template = np.reshape(template, (downsample_size * downsample_size))
    template_indicies = np.nonzero(template)
    return template_indicies


def downsample_tensor(activity_tensor, indicies, image_height, image_width, downsample_size=100):

    template_indicies = downsample_mask_further(indicies, image_height, image_width, downsample_size)

    downsampled_tensor = []
    for trial in activity_tensor:
        downsampled_trial = []
        for frame in trial:
            frame = Trajectory_Utils.create_image_from_data(frame, indicies, image_height, image_width)
            frame = resize(frame, (downsample_size, downsample_size))
            frame = np.reshape(frame, (downsample_size * downsample_size))
            frame = frame[template_indicies]
            downsampled_trial.append(frame)
        downsampled_tensor.append(downsampled_trial)

    return downsampled_tensor


def analyse_trial_aligned_trajectories(base_directory, onset_files, tensor_names, start_window, stop_window, tensor_save_directory):

    # Get File Structure
    split_base_directory = Path(base_directory).parts
    mouse_name = split_base_directory[-2]
    session_name = split_base_directory[-1]

    # Get Data Structure
    number_of_conditions = len(onset_files)

    # Load Combined Mask
    indicies, image_height, image_width = Trajectory_Utils.load_tight_mask()

    # Load Downsampled Tensors
    activity_tensor_list = []
    for condition_index in range(number_of_conditions):

        # Create Activity Tensor
        # Create_Activity_Tensor.create_activity_tensor(base_directory, onset_files[condition_index], start_window, stop_window, tensor_save_directory, start_cutoff=3000)

        # Load Activity Tensor
        activity_tensor = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, tensor_names[condition_index] + "_Activity_Tensor.npy"))

        # Downsample Tensor
        activity_tensor = downsample_tensor(activity_tensor, indicies, image_height, image_width)
        activity_tensor_list.append(activity_tensor)

   # Fit Model
    model = IncrementalPCA(n_components=3)
    for condition in activity_tensor_list:
        for trial in condition:
            model.partial_fit(trial)

    # Transform Data
    transformed_tensor_list = []
    for condition in activity_tensor_list:
        transformed_condition_list = []
        for trial in condition:
            transformed_trial = model.inverse_transform(trial)
            transformed_condition_list.append(transformed_trial)
        transformed_tensor_list.append(transformed_condition_list)

    # Plot Trajectories
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1, projection='3d')
    for trial in transformed_tensor_list[0]:
        axis_1.plot(trial[:, 0], trial[:, 1], trial[:, 2], alpha=0.2, c='b')

    for trial in transformed_tensor_list[1]:
        axis_1.plot(trial[:, 0], trial[:, 1], trial[:, 2], alpha=0.2, c='r')

    plt.show()


session_list = ["/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_31_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_02_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_04_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_08_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_10_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_12_Transition_Imaging",

                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_20_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_22_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_24_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_14_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_16_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_18_Transition_Imaging",

                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_17_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_19_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_23_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_30_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_06_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_08_Transition_Imaging",

                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_15_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_17_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_19_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_22_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_24_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_26_Transition_Imaging",

                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_14_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_20_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_26_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_05_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_08_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_10_Transition_Imaging",

                ]

# Get Analysis Details
analysis_name = "Unrewarded_Contextual_Modulation"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Trajectory_Utils.load_analysis_container(analysis_name)
tensor_save_directory = r"/media/matthew/29D46574463D2856/Widefield_Analysis/Noise_Correlation_Analysis/Activity_Tensors"
for base_directory in session_list:
    print("Analysing Session: ", base_directory)
    analyse_noise_correlations(base_directory, onset_files, tensor_names, start_window, stop_window, tensor_save_directory)