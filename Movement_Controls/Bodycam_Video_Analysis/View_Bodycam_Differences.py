import Get_Bodycam_SVD_Tensor
import numpy as np
import matplotlib.pyplot as plt


def get_mousecam_motion_energy_tensor(base_directory, onsets_file_list, video_file, start_window, stop_window):

    # Load Widefield Frame Indexes of Trial Starts
    onsets = Get_Bodycam_SVD_Tensor.load_onsets(base_directory, onsets_file_list)

    # Convert Widefield Frames Into Mousecam Frames
    onsets = Get_Bodycam_SVD_Tensor.convert_widefield_onsets_into_mousecam_onsets(base_directory, onsets)

    # Load Video Data Into A Tensor Of Shape (N_trials , Trial Length, Image Height, Image Width)
    mousecam_tensor = Get_Bodycam_SVD_Tensor.create_mousecam_tensor(base_directory, video_file, onsets, start_window, stop_window)

    #Get "Motion Energy" - (Absolute Value Of THe Difference Between Subsequent Frames)
    mousecam_tensor = Get_Bodycam_SVD_Tensor.get_motion_energy(mousecam_tensor, visualise=False)

    return mousecam_tensor


def compare_motion_energy_tensors(condition_1_mean_tensor, condition_2_mean_tensor):

    # Create Figure


    # Get Image Magnitudes
    image_magnitude = np.max(np.array([condition_1_mean_tensor, condition_2_mean_tensor]))
    difference_tensor = np.diff([condition_1_mean_tensor, condition_2_mean_tensor], axis=0)[0]
    difference_magntiude = np.max(np.abs(difference_tensor))

    # Get Tensor Structure
    number_of_timepoints = np.shape(condition_1_mean_tensor)[0]
    print("Number of timepoints", number_of_timepoints)

    plt.ion()
    figure_1 = plt.figure()
    for timepoint in range(number_of_timepoints):

        condition_1_axis = figure_1.add_subplot(1, 3, 1)
        condition_2_axis = figure_1.add_subplot(1, 3, 2)
        difference_axis = figure_1.add_subplot(1, 3, 3)

        condition_1_image = condition_1_mean_tensor[timepoint]
        condition_2_image = condition_2_mean_tensor[timepoint]
        difference_image = difference_tensor[timepoint]

        condition_1_axis.imshow(condition_1_image, cmap='jet', vmin=0, vmax=image_magnitude)
        condition_2_axis.imshow(condition_2_image, cmap='jet', vmin=0, vmax=image_magnitude)
        difference_axis.imshow(difference_image, cmap='bwr', vmin=-1*difference_magntiude, vmax=difference_magntiude)

        condition_1_axis.axis('off')
        condition_2_axis.axis('off')
        difference_axis.axis('off')

        plt.title(str(timepoint))
        #plt.show()
        plt.draw()
        plt.pause(0.1)
        plt.clf()

base_directory = r"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging"
start_window = -10
stop_window = 60
onset_files = [["visual_context_stable_vis_1_onsets.npy"], ["odour_context_stable_vis_1_onsets.npy"]]
tensor_names = ["Vis_2_Stable_Visual", "Vis_2_Stable_Odour"]
video_file = "NXAK14.1A_2021-06-17-14-30-28_cam_1.mp4"

"""
condition_1_tensor = get_mousecam_motion_energy_tensor(base_directory, onset_files[0], video_file, start_window, stop_window)
condition_1_tensor = np.mean(condition_1_tensor, axis=0)
np.save("/media/matthew/29D46574463D2856/Bodycam_Test_Tensors/condition_1_tensor.npy", condition_1_tensor)

condition_2_tensor = get_mousecam_motion_energy_tensor(base_directory, onset_files[1], video_file, start_window, stop_window)
condition_2_tensor = np.mean(condition_2_tensor, axis=0)
np.save("/media/matthew/29D46574463D2856/Bodycam_Test_Tensors/condition_2_tensor.npy", condition_2_tensor)
"""

condition_1_tensor = np.load("/media/matthew/29D46574463D2856/Bodycam_Test_Tensors/condition_1_tensor.npy")
condition_2_tensor = np.load("/media/matthew/29D46574463D2856/Bodycam_Test_Tensors/condition_2_tensor.npy")
compare_motion_energy_tensors(condition_1_tensor, condition_2_tensor)