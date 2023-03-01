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
from tqdm import tqdm
from scipy import stats

from Widefield_Utils import Create_Activity_Tensor
from Widefield_Utils import widefield_utils


def analyse_group_matrix(mouse_list, group_name, correlation_matrix_name, tensor_save_directory, mean_matrix_save_directory, cmap):
    mean_context_1_matrix_list = []
    mean_context_2_matrix_list = []

    for mouse in tqdm(mouse_list):
        mouse_context_1_matricies = []
        mouse_context_2_matricies = []

        for base_directory in mouse:
            # Get File Structure
            split_base_directory = Path(base_directory).parts
            mouse_name = split_base_directory[-2]
            session_name = split_base_directory[-1]

            context_1_matrix = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, tensor_names[0] + correlation_matrix_name))
            context_2_matrix = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, tensor_names[1] + correlation_matrix_name))


            mouse_context_1_matricies.append(context_1_matrix)
            mouse_context_2_matricies.append(context_2_matrix)

        mean_mouse_context_1_matrix = np.mean(mouse_context_1_matricies, axis=0)
        mean_mouse_context_2_matrix = np.mean(mouse_context_2_matricies, axis=0)

        mean_context_1_matrix_list.append(mean_mouse_context_1_matrix)
        mean_context_2_matrix_list.append(mean_mouse_context_2_matrix)

    mean_context_1_matrix_list = np.array(mean_context_1_matrix_list)
    mean_context_2_matrix_list = np.array(mean_context_2_matrix_list)

    t_stats, p_values = stats.ttest_rel(mean_context_1_matrix_list, mean_context_2_matrix_list, axis=0)

    modulation_matricies = np.subtract(mean_context_1_matrix_list, mean_context_2_matrix_list)
    mean_modulation_matrix = np.mean(modulation_matricies, axis=0)

    thresholded_mean_modulation_matrix = np.where(p_values < 0.05, mean_modulation_matrix, 0)

    np.save(os.path.join(mean_matrix_save_directory, "Mean_" + group_name + "_" + correlation_matrix_name + "_Modulation.npy"), mean_modulation_matrix)
    np.save(os.path.join(mean_matrix_save_directory, "Mean_" + group_name + "_" + correlation_matrix_name + "_Modulation_Significant.npy"), thresholded_mean_modulation_matrix)

    plt.title(group_name + "_" + correlation_matrix_name + "_t_stats")
    t_stat_mangnitude = 4
    plt.imshow(t_stats, cmap=cmap, vmin=-t_stat_mangnitude, vmax=t_stat_mangnitude)
    plt.colorbar()
    plt.show()

    plt.title(group_name + "_" + correlation_matrix_name + "Mean_Modulation")
    plt.imshow(mean_modulation_matrix, cmap=cmap, vmin=-0.5, vmax=0.5)
    plt.colorbar()
    plt.show()

    plt.title(group_name + "_" + correlation_matrix_name + "_Thresholded_Mean")
    plt.imshow(thresholded_mean_modulation_matrix, cmap=cmap, vmin=-0.5, vmax=0.5)
    plt.colorbar()
    plt.show()


def get_mouse_modulation_matrix(session_list, tensor_save_directory, tensor_names, signal_or_noise):

    modulation_matrix_list = []
    for base_directory in session_list:

        # Get File Structure
        split_base_directory = Path(base_directory).parts
        mouse_name = split_base_directory[-2]
        session_name = split_base_directory[-1]

        context_1_matrix = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, tensor_names[0] + "_" + signal_or_noise + "_Correlation_Matrix.npy"))
        context_2_matrix = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, tensor_names[1] + "_" + signal_or_noise + "_Correlation_Matrix.npy"))

        # Get Contextual Modulation
        contextual_modulation = np.subtract(context_1_matrix, context_2_matrix)
        contextual_modulation = np.nan_to_num(contextual_modulation)

        # Add To List
        modulation_matrix_list.append(contextual_modulation)

    mean_modulation = np.mean(modulation_matrix_list, axis=0)

    return mean_modulation



def downsample_mask_further(indicies, image_height, image_width, downsample_size=100):
    template = np.zeros((image_height*image_width))
    template[indicies] = 1
    template = np.reshape(template, (image_height, image_width))
    template = resize(template, (downsample_size, downsample_size), anti_aliasing=True)
    template = np.reshape(template, (downsample_size * downsample_size))
    template = np.where(template > 0.5, 1, 0)
    template_indicies = np.nonzero(template)
    return template_indicies, downsample_size, downsample_size


def compare_genotype_modulation(control_mouse_list, mutant_mouse_list, tensor_save_directory, tensor_names, signal_or_noise):

    control_modulation_matrix_list = []
    mutant_modulation_matrix_list = []

    for mouse in control_mouse_list:
        modulation_matrix = get_mouse_modulation_matrix(mouse, tensor_save_directory, tensor_names, signal_or_noise)
        control_modulation_matrix_list.append(modulation_matrix)

    for mouse in mutant_mouse_list:
        modulation_matrix = get_mouse_modulation_matrix(mouse, tensor_save_directory, tensor_names, signal_or_noise)
        mutant_modulation_matrix_list.append(modulation_matrix)

    # Convert To Arrays
    control_modulation_matrix_list = np.array(control_modulation_matrix_list)
    mutant_modulation_matrix_list = np.array(mutant_modulation_matrix_list)

    # Compare Significance
    t_stats, p_values = stats.ttest_ind(control_modulation_matrix_list, mutant_modulation_matrix_list, axis=0)
    plt.title("Genotype_Modulation_Differences" + "_" + signal_or_noise + "_t_stats")
    t_stat_mangnitude = 2
    plt.imshow(t_stats, cmap=cmap, vmin=-t_stat_mangnitude, vmax=t_stat_mangnitude)
    plt.show()

    # Get Mean Genotype Modulation
    mean_control_modulation = np.mean(control_modulation_matrix_list, axis=0)
    mean_mutant_modulation = np.mean(mutant_modulation_matrix_list, axis=0)
    genotype_modulation_difference = np.subtract(mean_control_modulation, mean_mutant_modulation)
    plt.title("Genotype_Modulation_Difference" + "_" + signal_or_noise + "Mean_Modulation")
    plt.imshow(genotype_modulation_difference, cmap=cmap, vmin=-0.5, vmax=0.5)
    plt.show()

    # Get Thresholded Modulation
    thresholded_mean_modulation_matrix = np.where(p_values < 0.05, genotype_modulation_difference, 0)
    plt.title("Genotype_Modulation_Difference" + "_" + signal_or_noise + "_Thresholded_Mean")
    plt.imshow(thresholded_mean_modulation_matrix, cmap=cmap, vmin=-0.5, vmax=0.5)
    plt.show()

    # Create Thresholded Mean Modulation Map
    mean_modulation_vector = np.mean(thresholded_mean_modulation_matrix, axis=0)
    print("Mean Modulation vecotr", np.shape(mean_modulation_vector))

    # Load mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()

    # Downsample mask Further
    indicies, image_height, image_width = downsample_mask_further(indicies, image_height, image_width)


    print("Indicies", np.shape(indicies))
    modulation_map = widefield_utils.create_image_from_data(mean_modulation_vector, indicies, image_height, image_width)
    plt.imshow(modulation_map, cmap=cmap, vmin=-0.2, vmax=0.2)
    plt.show()





control_mouse_list = [

                ["/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_28_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_05_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_09_Switching_Imaging"],

                ["/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_29_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_05_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging"],

                ["/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_02_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_04_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_06_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_02_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_08_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_10_Transition_Imaging"],

                ["/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_26_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_28_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_02_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_23_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_31_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_04_02_Transition_Imaging"],

                ["/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_21_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_23_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_11_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_13_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_15_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_17_Transition_Imaging"],

                ["/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_14_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_20_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_22_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_29_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_03_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_05_Transition_Imaging"],
                ]


mutant_mouse_list = [

                 ["/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_13_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_15_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_17_Switching_Imaging"],

                ["/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_31_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_02_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_04_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_08_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_10_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_12_Transition_Imaging"],

               ["/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_20_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_22_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_24_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_14_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_16_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_18_Transition_Imaging"],

                ["/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_17_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_19_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_23_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_30_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_06_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_08_Transition_Imaging"],

                ["/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_15_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_17_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_19_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_22_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_24_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_26_Transition_Imaging"],

                ["/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_14_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_20_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_26_Switching_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_05_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_08_Transition_Imaging",
                "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_10_Transition_Imaging"],
                ]


# Get Analysis Details
analysis_name = "Unrewarded_Contextual_Modulation"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)
tensor_save_directory = r"/media/matthew/29D46574463D2856/Widefield_Analysis/Noise_Correlation_Analysis/Activity_Tensors"
mean_matrix_save_directory = r"/media/matthew/29D46574463D2856/Widefield_Analysis/Noise_Correlation_Analysis/Mean_Modulation_Matricies"

# Get Colourmap
cmap = widefield_utils.get_musall_cmap()


correlation_matrix_filename = "_Signal_Correlation_Matrix_Aligned_Across_Mice.npy"
analyse_group_matrix(control_mouse_list, "Control", correlation_matrix_filename, tensor_save_directory, mean_matrix_save_directory, cmap)

correlation_matrix_filename = "_Signal_Correlation_Matrix_Aligned_Across_Mice.npy"
analyse_group_matrix(mutant_mouse_list, "Mutant", correlation_matrix_filename, tensor_save_directory, mean_matrix_save_directory, cmap)

correlation_matrix_filename = "_Noise_Correlation_Matrix_Aligned_Across_Mice.npy"
analyse_group_matrix(control_mouse_list, "Control", correlation_matrix_filename, tensor_save_directory, mean_matrix_save_directory, cmap)

correlation_matrix_filename = "_Noise_Correlation_Matrix_Aligned_Across_Mice.npy"
analyse_group_matrix(mutant_mouse_list, "Mutant", correlation_matrix_filename, tensor_save_directory, mean_matrix_save_directory, cmap)


#analyse_group_matrix(mutant_mouse_list, "Mutant", "Signal", tensor_save_directory, mean_matrix_save_directory, cmap)

#analyse_group_matrix(control_mouse_list, "Control", "Signal", tensor_save_directory, mean_matrix_save_directory, cmap)
#analyse_group_matrix(mutant_mouse_list, "Mutant", "Signal", tensor_save_directory, mean_matrix_save_directory, cmap)

#compare_genotype_modulation(control_mouse_list, mutant_mouse_list, tensor_save_directory, tensor_names, "Signal")