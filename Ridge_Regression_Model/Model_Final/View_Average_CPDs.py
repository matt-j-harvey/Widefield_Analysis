import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import Regression_Utils


def view_coef_group(coef, group_name, indicies, image_height, image_width):
    difference_cmap = Regression_Utils.get_musall_cmap()
    coef_map = Regression_Utils.create_image_from_data(coef, indicies, image_height, image_width)
    coef_magnitude = np.max(np.abs(coef_map))
    plt.title(group_name)
    plt.imshow(coef_map, cmap='hot', vmax=coef_magnitude, vmin=0)
    plt.axis('off')
    plt.colorbar()
    plt.show()


"""

    print("Coef Shape", np.shape(regression_coefs))

    # Load Mask
    indicies, image_height, image_width = Regression_Utils.load_downsampled_mask(base_directory)

    # View Explained Variance
    explained_variance = Regression_Utils.create_image_from_data(explained_variance, indicies, image_height, image_width)
    plt.title("Explained Variance")
    plt.imshow(explained_variance, cmap='hot', vmin=0, vmax=0.4)
    plt.axis('off')
    plt.colorbar()
    plt.show()

    # View Coefs
    difference_cmap = Regression_Utils.get_musall_cmap()

    # View CPDs For Each Coef Group
    number_of_regression_groups = len(regression_group_names)
    for regression_group_index in range(number_of_regression_groups):
        group_name = regression_group_names[regression_group_index]
        group_cpd = regression_cpds[regression_group_index]
        view_coef_group(group_cpd, group_name, indicies, image_height, image_width)


    coef_count = 0
    for coef in regression_coefs:
        coef_map = Regression_Utils.create_image_from_data(coef,  indicies, image_height, image_width)
        coef_magnitude = np.max(np.abs(coef_map))

        plt.title(str(coef_count))
        plt.imshow(coef_map, cmap=difference_cmap, vmax=coef_magnitude, vmin=-coef_magnitude)

        plt.show()

        coef_count += 1
    """


def align_cpds(base_directory, cpd_list):

    # Load Mask
    indicies, image_height, image_width = Regression_Utils.load_downsampled_mask(base_directory)

    # Load Within Mouse Alignment Dictionary
    within_mouse_alignment_dictionary = np.load(os.path.join(base_directory, "Within_Mouse_Alignment_Dictionary.npy"), allow_pickle=True)[()]

    # Load Across Mouse Alignment Dictionary
    root_directory = Regression_Utils.get_root_directory(base_directory)
    across_mouse_alignment_dictionary = np.load(os.path.join(root_directory, "Across_Mouse_Alignment_Dictionary.npy"), allow_pickle=True)[()]

    # Load Tight Indicies
    tight_indicies, tight_height, tight_width = Regression_Utils.load_tight_mask()

    print("tight height", tight_height, "tight width", tight_width)

    aligned_cpds = []
    for cpd in cpd_list:

        # Reconstruct Image
        cpd = Regression_Utils.create_image_from_data(cpd, indicies, image_height, image_width)

        # Align Within Mouse
        cpd = Regression_Utils.transform_image(cpd, within_mouse_alignment_dictionary)

        # Align Across Mice
        cpd = Regression_Utils.transform_image(cpd, across_mouse_alignment_dictionary)

        # Apply Tight Mask
        template = np.zeros(tight_height * tight_width)
        cpd = np.reshape(cpd, (image_height * image_width))
        template[indicies] = cpd[indicies]
        template = np.reshape(template, (image_height, image_width))
        aligned_cpds.append(template)

    return aligned_cpds



def get_average_cpds(session_list):

    cpd_list = []
    cpd_name_list = []

    for base_directory in tqdm(session_list):

        # Load Regression Dict
        regression_dictionary = np.load(os.path.join(base_directory, "Regression_Coefs",  "Regression_Dictionary_Simple.npy"), allow_pickle=True)[()]
        regression_group_names = regression_dictionary["Coef_Names"]
        regression_cpds = regression_dictionary["Coefficients_of_Partial_Determination"]
        regression_cpds = np.transpose(regression_cpds)

        # Align CPDS
        aligned_cpds = align_cpds(base_directory, regression_cpds)

        cpd_list.append(aligned_cpds)
        cpd_name_list.append(regression_group_names)

    cpd_list = np.array(cpd_list)
    print("CPD List Shape", np.shape(cpd_list))
    return cpd_list, cpd_name_list


def view_average_variance_explained(session_list):

    variance_list = []

    for base_directory in tqdm(session_list):

        # Load Regression Dict
        regression_dictionary = np.load(os.path.join(base_directory, "Regression_Coefs", "Regression_Dictionary_Simple.npy"), allow_pickle=True)[()]
        variance_explained = regression_dictionary["Variance_Explained"]

        # Align CPDS
        [variance_explained] = align_cpds(base_directory, [variance_explained])

        variance_list.append(variance_explained)

    variance_list = np.array(variance_list)
    print("Var list", np.shape(variance_list))

    mean_variance_explained = np.mean(variance_list, axis=0)

    plt.title("Total Variance Explained")
    plt.imshow(mean_variance_explained, cmap='hot')
    plt.colorbar()
    plt.show()


def view_average_cpds(cpd_list, cpd_name_list):

    mean_cpds = np.mean(cpd_list, axis=0)

    number_of_cpds = len(cpd_name_list[0])

    for cpd_index in range(number_of_cpds):

        cpd_image = mean_cpds[cpd_index]
        cpd_name = cpd_name_list[0][cpd_index]

        plt.title(cpd_name)
        plt.imshow(cpd_image, cmap='hot')
        plt.axis('off')
        plt.colorbar()
        plt.show()




control_session_list = [

    r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_05_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_09_Switching_Imaging",

    r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_29_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_05_Switching_Imaging",

    r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_21_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_23_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_11_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_13_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_15_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_17_Transition_Imaging",

    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_14_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_20_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_22_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_29_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_03_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_05_Transition_Imaging",

    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_02_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_04_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_06_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_02_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_08_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_10_Transition_Imaging",

    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_26_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_28_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_02_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_23_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_31_Transition_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_04_02_Transition_Imaging",

]


#cpd_list, cpd_names = get_average_cpds(control_session_list)
#view_average_cpds(cpd_list, cpd_names)

view_average_variance_explained(control_session_list)