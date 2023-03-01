import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import fdrcorrection

from Widefield_Utils import widefield_utils



def view_signficiance_map(tensor_directory):


    # Load P Values and T Stats
    p_values = np.load(os.path.join(tensor_directory, "Full_Model_p_value_tensor_raw.npy"))
    t_stats = np.load(os.path.join(tensor_directory, "Full_Model_t_stat_tensor_raw.npy"))

    # Load mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

    # Load Colourmap
    colourmap = widefield_utils.get_musall_cmap()

    # Create Image
    raw_t_stat_image = widefield_utils.create_image_from_data(t_stats, indicies, image_height, image_width)
    plt.title("Raw T Stats")
    plt.imshow(raw_t_stat_image, cmap=colourmap, vmin=-4, vmax=4)
    plt.colorbar()
    plt.show()

    # Threshold Raw T Stats
    thresholded_t_stats = np.where(p_values < 0.05, t_stats, 0)
    thresholded_t_stat_image = widefield_utils.create_image_from_data(thresholded_t_stats, indicies, image_height, image_width)
    plt.title("Significant T Stats")
    plt.imshow(thresholded_t_stat_image, cmap=colourmap, vmin=-4, vmax=4)
    plt.colorbar()
    plt.show()

    # Multiple Comparisons Correction
    rejected, corrrected_p_values = fdrcorrection(p_values, alpha=0.1)


    corrected_effects = np.where(rejected == 1, t_stats, 0)
    corrected_effects_image = widefield_utils.create_image_from_data(corrected_effects, indicies, image_height, image_width)
    plt.title("Multiple Comparisons Correction")
    plt.imshow(corrected_effects_image, cmap=colourmap, vmin=-4, vmax=4)
    plt.colorbar()
    plt.show()

#tensor_directory = r"/media/matthew/External_Harddrive_2/Control_Switching_Analysis/Full_Model"
tensor_directory = r"/media/matthew/External_Harddrive_2/Neurexin_Switching_Analysis/Full_Model"
view_signficiance_map(tensor_directory)