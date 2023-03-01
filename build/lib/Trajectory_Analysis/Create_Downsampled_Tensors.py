import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm
import os
from sklearn.decomposition import IncrementalPCA
import pickle
import tables
from skimage.transform import resize

from Files import Session_List
from Widefield_Utils import widefield_utils, Create_Activity_Tensor, Create_Video_From_Tensor


def downsample_trial(trial_data, full_indicies, full_image_height, full_image_width, downsampled_indicies, downsampled_height, downsampled_width):

    # Create List To Hold New Downsampled Trial
    downsampled_trial = []

    for timepoint in trial_data:

        # Reconstruct Image
        reconstructed_timepoint = widefield_utils.create_image_from_data(timepoint, full_indicies, full_image_height, full_image_width)

        # Downsample Image
        downsampled_timepoint = resize(reconstructed_timepoint, (downsampled_height, downsampled_width), anti_aliasing=True)

        # Flatten Image
        downsampled_timepoint = np.reshape(downsampled_timepoint, (downsampled_height * downsampled_width))

        # Take Downsampled Mask Indicies
        downsampled_timepoint = downsampled_timepoint[downsampled_indicies]

        downsampled_trial.append(downsampled_timepoint)

    downsampled_trial = np.array(downsampled_trial)
    return downsampled_trial



def create_downsampled_tensors(nested_session_list, condition_name, tensor_save_directory, output_directory, trial_length, indicies, image_height, image_width, downsample_size=100):

    """
    Tensors Must All Be Aligned Into Common Space
    Delta F File = N Trials x N Timepoints, X Pixels
    Delta F Key (N Trials, 3) - 0 = Mouse Index, 1 = Session Index
    """

    # Get Further Downsampled Mask
    downsampled_indicies, downsampled_image_height, downsampled_image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width, downsample_size)
    print("Downsampled Indicies", downsampled_indicies)

    # Get Number of Pixels
    number_of_pixels = np.shape(downsampled_indicies)[1]
    condition_name = condition_name.replace('_onsets', '')
    condition_name = condition_name.replace('.npy', '')

    print("Trial Length", trial_length)
    print("Number Of Pixels", number_of_pixels)

    # Create Output File
    output_file_path = os.path.join(output_directory, condition_name + "_Combined_Downsampled_Data.h5")
    output_file = tables.open_file(output_file_path, mode='w')
    output_data = output_file.create_earray(output_file.root, 'Data', tables.Float32Atom(), shape=(0, trial_length, number_of_pixels))
    data_key = output_file.create_earray(output_file.root, 'Trial_Key', tables.UInt8Atom(), shape=(0, 2))


   # Iterate Through Mouse
    mouse_index = 0
    session_index = 0
    for mouse in tqdm(nested_session_list):
        for session in mouse:

            # Get Path Details
            mouse_name, session_name = widefield_utils.get_mouse_name_and_session_name(session)

            # Load Activity Tensor
            activity_tensor = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, condition_name + "_Activity_Tensor_Aligned_Across_Mice.npy"), allow_pickle=True)

            # Add Each Trial To Storage
            for trial in activity_tensor:

                # Downsample Trial
                trial = downsample_trial(trial, indicies, image_height, image_width, downsampled_indicies, downsampled_image_height, downsampled_image_width)

                # Add Trial Data To Matrix
                output_data.append([trial])

                # Add Trial Details To Trial Key
                data_key.append([np.array([mouse_index, session_index])])

            # Flush Storage
            data_key.flush()
            output_data.flush()

            session_index += 1
        mouse_index += 1

    # Close File
    output_file.close()