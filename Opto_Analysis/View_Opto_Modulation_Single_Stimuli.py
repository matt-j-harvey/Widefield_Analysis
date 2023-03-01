import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from Widefield_Utils import widefield_utils, Create_Activity_Tensor




def view_tensors(base_directory, condition_1_opto_tensor, condition_1_non_opto_tensor, condition_2_opto_tensor, condition_2_non_opto_tensor, image_save_directory):

    condition_1_opto_mean = np.mean(condition_1_opto_tensor, axis=0)
    condition_2_opto_mean = np.mean(condition_2_opto_tensor, axis=0)
    condition_1_non_opto_mean = np.mean(condition_1_non_opto_tensor, axis=0)
    condition_2_non_opto_mean = np.mean(condition_2_non_opto_tensor, axis=0)

    condition_1_opto_effect = np.subtract(condition_1_opto_mean, condition_1_non_opto_mean)
    condition_2_opto_effect = np.subtract(condition_2_opto_mean, condition_2_non_opto_mean)
    
    condition_non_opto_difference = np.subtract(condition_1_non_opto_mean, condition_2_non_opto_mean)
    condition_opto_difference = np.subtract(condition_1_opto_mean, condition_2_opto_mean)
    
    modulation_difference = np.subtract(condition_1_opto_effect, condition_2_opto_effect)

    # Load Colourmap
    cmap = widefield_utils.get_musall_cmap()

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_downsampled_mask(base_directory)

    plt.ion()
    figure_1 = plt.figure()
    rows = 3
    columns = 3
    figure_1_gridspec = GridSpec(figure=figure_1, nrows=rows, ncols=columns)

    number_of_timepoints = np.shape(condition_1_opto_tensor)[1]
    for timepoint_index in range(number_of_timepoints):

        # Create Axes
        condition_1_opto_axis = figure_1.add_subplot(figure_1_gridspec[0, 0])
        condition_1_non_opto_axis = figure_1.add_subplot(figure_1_gridspec[0, 1])
        condition_1_opto_effect_axis = figure_1.add_subplot(figure_1_gridspec[0, 2])

        condition_2_opto_axis = figure_1.add_subplot(figure_1_gridspec[1, 0])
        condition_2_non_opto_axis = figure_1.add_subplot(figure_1_gridspec[1, 1])
        condition_2_opto_effect_axis = figure_1.add_subplot(figure_1_gridspec[1, 2])

        opto_diff_axis = figure_1.add_subplot(figure_1_gridspec[2, 0])
        non_opto_diff_axis = figure_1.add_subplot(figure_1_gridspec[2, 1])
        opto_effect_modulation_axis = figure_1.add_subplot(figure_1_gridspec[2, 2])
        
        
        # Create Images 
        condition_1_opto_image = widefield_utils.create_image_from_data(condition_1_opto_mean[timepoint_index], indicies, image_height, image_width)
        condition_1_non_opto_image = widefield_utils.create_image_from_data(condition_1_non_opto_mean[timepoint_index], indicies, image_height, image_width)
        condition_1_opto_effect_image = widefield_utils.create_image_from_data(condition_1_opto_effect[timepoint_index], indicies, image_height, image_width)

        condition_2_opto_image = widefield_utils.create_image_from_data(condition_2_opto_mean[timepoint_index], indicies, image_height, image_width)
        condition_2_non_opto_image = widefield_utils.create_image_from_data(condition_2_non_opto_mean[timepoint_index], indicies, image_height, image_width)
        condition_2_opto_effect_image = widefield_utils.create_image_from_data(condition_2_opto_effect[timepoint_index], indicies, image_height, image_width)

        opto_diff_image = widefield_utils.create_image_from_data(condition_opto_difference[timepoint_index], indicies, image_height, image_width)
        non_opto_diff_image = widefield_utils.create_image_from_data(condition_non_opto_difference[timepoint_index], indicies, image_height, image_width)
        opto_effect_modulation_image = widefield_utils.create_image_from_data(modulation_difference[timepoint_index], indicies, image_height, image_width)

        
        # Display Images
        magnitude = 0.03
        condition_1_opto_axis.imshow(condition_1_opto_image, vmin=-magnitude, vmax=magnitude, cmap=cmap)
        condition_1_non_opto_axis.imshow(condition_1_non_opto_image, vmin=-magnitude, vmax=magnitude, cmap=cmap)
        condition_1_opto_effect_axis.imshow(condition_1_opto_effect_image, vmin=-magnitude, vmax=magnitude, cmap=cmap)

        condition_2_opto_axis.imshow(condition_2_opto_image, vmin=-magnitude, vmax=magnitude, cmap=cmap)
        condition_2_non_opto_axis.imshow(condition_2_non_opto_image, vmin=-magnitude, vmax=magnitude, cmap=cmap)
        condition_2_opto_effect_axis.imshow(condition_2_opto_effect_image, vmin=-magnitude, vmax=magnitude, cmap=cmap)

        opto_diff_axis.imshow(opto_diff_image, vmin=-magnitude, vmax=magnitude, cmap=cmap)
        non_opto_diff_axis.imshow(non_opto_diff_image, vmin=-magnitude, vmax=magnitude, cmap=cmap)
        opto_effect_modulation_axis.imshow(opto_effect_modulation_image, vmin=-magnitude, vmax=magnitude, cmap=cmap)


        # Set Titles
        condition_1_opto_axis.set_title("Condition 1 Opto")
        condition_1_non_opto_axis.set_title("Condition 1 Non-Opto")
        condition_1_opto_effect_axis.set_title("Condition 1 Opto Effect")

        condition_2_opto_axis.set_title("Condition 2 Opto")
        condition_2_non_opto_axis.set_title("Condition 2 Non Opto")
        condition_2_opto_effect_axis.set_title("Condition 2 Opto Effect")

        opto_diff_axis.set_title("Opto Difference")
        non_opto_diff_axis.set_title("Non-Opto Difference")
        opto_effect_modulation_axis.set_title("Opto Effect Modulation")


        # Remove Axis
        condition_1_opto_axis.axis('off')
        condition_1_non_opto_axis.axis('off')
        condition_1_opto_effect_axis.axis('off')
        condition_2_opto_axis.axis('off')
        condition_2_non_opto_axis.axis('off')
        condition_2_opto_effect_axis.axis('off')
        opto_diff_axis.axis('off')
        non_opto_diff_axis.axis('off')
        opto_effect_modulation_axis.axis('off')



        figure_1.suptitle(str(timepoint_index))
        plt.draw()
        plt.savefig(os.path.join(image_save_directory, str(timepoint_index).zfill(3) + ".png"))
        plt.clf()


base_directory = r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Switching_Opto/KPGC2.2G/2022_12_08_Switching_Opto"

# Create Activity Tensors
start_window = -10
stop_window = 100
tensor_save_directory = r"/media/matthew/29D46574463D2856/Widefield_Opto_Modulation"

condition_1_opto = "visual_context_vis_2_opto_correct_onset_frames.npy"
condition_1_non_opto = "visual_context_vis_2_nonopto_correct_onset_frames.npy"

condition_2_opto = "odour_context_vis_2_opto_correct_onset_frames.npy"
condition_2_non_opto = "odour_context_vis_2_nonopto_correct_onset_frames.npy"


Create_Activity_Tensor.create_activity_tensor(base_directory, condition_1_opto, start_window, stop_window, tensor_save_directory, start_cutoff=3000, align_within_mice=False, align_across_mice=False, baseline_correct=True)
Create_Activity_Tensor.create_activity_tensor(base_directory, condition_1_non_opto, start_window, stop_window, tensor_save_directory, start_cutoff=3000, align_within_mice=False, align_across_mice=False, baseline_correct=True)
Create_Activity_Tensor.create_activity_tensor(base_directory, condition_2_opto, start_window, stop_window, tensor_save_directory, start_cutoff=3000, align_within_mice=False, align_across_mice=False, baseline_correct=True)
Create_Activity_Tensor.create_activity_tensor(base_directory, condition_2_non_opto, start_window, stop_window, tensor_save_directory, start_cutoff=3000, align_within_mice=False, align_across_mice=False, baseline_correct=True)

condition_1_opto_tensor_name = condition_1_opto.replace("onset_frames.npy", "Activity_Tensor_Unaligned.npy")
condition_1_non_opto_tensor_name = condition_1_non_opto.replace("onset_frames.npy", "Activity_Tensor_Unaligned.npy")
condition_2_opto_tensor_name = condition_2_opto.replace("onset_frames.npy", "Activity_Tensor_Unaligned.npy")
condition_2_non_opto_tensor_name = condition_2_non_opto.replace("onset_frames.npy", "Activity_Tensor_Unaligned.npy")

# Load Activity Tensors
mouse_save_directory = r"/media/matthew/29D46574463D2856/Widefield_Opto_Modulation/KPGC2.2G/2022_12_08_Switching_Opto"
condition_1_opto_tensor = np.load(os.path.join(mouse_save_directory, condition_1_opto_tensor_name))
condition_1_non_opto_tensor = np.load(os.path.join(mouse_save_directory, condition_1_non_opto_tensor_name))
condition_2_opto_tensor = np.load(os.path.join(mouse_save_directory, condition_2_opto_tensor_name))
condition_2_non_opto_tensor = np.load(os.path.join(mouse_save_directory, condition_2_non_opto_tensor_name))


image_save_directory = r"/media/matthew/29D46574463D2856/Widefield_Opto_Modulation/KPGC2.2G/2022_12_08_Switching_Opto/Images"

print("Condition 1 opto tensor", np.shape(condition_1_opto_tensor))
print("Condition 2 opto tensor", np.shape(condition_2_opto_tensor))
print("Condition 1 non opto tensor", np.shape(condition_1_non_opto_tensor))
print("Condition 2 non opto tensor", np.shape(condition_2_non_opto_tensor))


view_tensors(base_directory, condition_1_opto_tensor, condition_1_non_opto_tensor, condition_2_opto_tensor, condition_2_non_opto_tensor, image_save_directory)