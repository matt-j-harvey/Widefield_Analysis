import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage

def create_image_from_data(data, indicies, image_height, image_width):
    template = np.zeros((image_height, image_width))
    data = np.nan_to_num(data)
    np.put(template, indicies, data)
    image = np.ndarray.reshape(template, (image_height, image_width))
    return image


def get_blue_black_cmap():
    cmap = LinearSegmentedColormap.from_list('mycmap', [
        (0, 0.87, 0.9, 1),
        (0, 0, 1, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 1, 0, 1),

    ])
    return cmap


def load_tight_mask(tight_mask_file):
    tight_mask_dict = np.load(tight_mask_file, allow_pickle=True)[()]
    indicies = tight_mask_dict["indicies"]
    image_height = tight_mask_dict["image_height"]
    image_width = tight_mask_dict["image_width"]
    return indicies, image_height, image_width


def visualise_individual_trials(activity_tensor_file, mask_file, vmin=-0.05, vmax=0.05):

    # Load Activity Tensor
    activity_tensor = np.load(activity_tensor_file)

    # Get Tensor Shape
    n_trials, n_timepoints, n_pixels = np.shape(activity_tensor)

    # Load Mask Details
    pixel_indicies, image_height, image_width = load_tight_mask(mask_file)

    # Load Colourmap
    colourmap = get_blue_black_cmap()

    # Visualise Each Trial
    figure_1 = plt.figure()
    plt.ion()
    for trial_index in range(n_trials):
        for timepoint_index in range(n_timepoints):

            # Get Data For This Timepoint
            frame_data = activity_tensor[trial_index, timepoint_index]

            # Reconstruct Into Brain Space
            frame_image = create_image_from_data(frame_data, pixel_indicies, image_height, image_width)
            #frame_image = ndimage.gaussian_filter(frame_image, sigma=1)

            # Create Axis
            axis_1 = figure_1.add_subplot(1,1,1)

            # Display Image
            axis_1.imshow(frame_image, cmap=colourmap, vmin=vmin, vmax=vmax)

            axis_1.axis('off')
            plt.title("Trial: " + str(trial_index) + "Timepoint:" + str(timepoint_index))
            plt.draw()
            plt.pause(0.1)
            plt.clf()




def visualise_mean_response(activity_tensor_file, mask_file, vmin=-0.05, vmax=0.05):

    # Load Activity Tensor
    activity_tensor = np.load(activity_tensor_file)
    mean_response = np.mean(activity_tensor, axis=0)


    # Get Tensor Shape
    n_timepoints, n_pixels = np.shape(mean_response)

    # Load Mask Details
    pixel_indicies, image_height, image_width = load_tight_mask(mask_file)

    # Load Colourmap
    colourmap = get_blue_black_cmap()

    # Visualise Each Trial
    figure_1 = plt.figure()
    plt.ion()

    for timepoint_index in range(n_timepoints):

        # Get Data For This Timepoint
        frame_data = mean_response[timepoint_index]

        # Reconstruct Into Brain Space
        frame_image = create_image_from_data(frame_data, pixel_indicies, image_height, image_width)

        # Create Axis
        axis_1 = figure_1.add_subplot(1,1,1)

        # Display Image
        axis_1.imshow(frame_image, cmap=colourmap, vmin=vmin, vmax=vmax)

        axis_1.axis('off')
        plt.title("Mean Response - Timepoint:" + str(timepoint_index))
        plt.draw()
        plt.pause(0.1)
        plt.clf()


activity_tensor_file = r"/media/matthew/External_Harddrive_2/Angus_Collaboration/Activity_Tensors/NRXN78.1A/2020_12_05_Switching_Imaging/visual_context_stable_vis_2_Activity_Tensor_Aligned_Across_Mice.npy"
mask_file = r"/home/matthew/Documents/Github_Code_Clean/Widefield_Analysis/Files/Tight_Mask_Dict.npy"

#visualise_mean_response(activity_tensor_file, mask_file)
visualise_individual_trials(activity_tensor_file, mask_file)