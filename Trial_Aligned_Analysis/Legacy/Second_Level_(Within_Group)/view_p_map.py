import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from Widefield_Utils import widefield_utils
from statsmodels.stats.multitest import fdrcorrection


def downsample_mask_further(indicies, image_height, image_width, new_size=100):

    # Reconstruct To 2D
    template = np.zeros(image_height * image_width)
    template[indicies] = 1
    template = np.reshape(template, (image_height, image_width))

    # Downsample
    template = resize(template, (new_size, new_size))

    template = np.reshape(template, new_size * new_size)
    template = np.where(template > 0.5, 1, 0)
    indicies = np.nonzero(template)

    return indicies, new_size, new_size


def view_p_map_downsized(p_map_file):

    # Load P Map
    p_map = np.load(p_map_file)
    p_map = np.nan_to_num(p_map)

    # Perform FDR Correction
    rejected, corrected_p_map = fdrcorrection(p_map, alpha=0.05)

    # Invert
    p_map = 1 - p_map
    corrected_p_map = 1 - corrected_p_map

    # Load Mask Details
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = downsample_mask_further(indicies, image_height, image_width)

    # Reconstruct Images
    p_map = widefield_utils.create_image_from_data(p_map, indicies, image_height, image_width)
    corrected_p_map = widefield_utils.create_image_from_data(corrected_p_map, indicies, image_height, image_width)

    # View Raw
    plt.title("P Map All")
    plt.imshow(p_map, vmin=0, vmax=1, cmap='jet')
    plt.colorbar()
    plt.show()

    # View Raw
    plt.title("Raw P Map")
    plt.imshow(p_map, vmin=0.95, vmax=1)
    plt.colorbar()
    plt.show()

    # View Corrected
    plt.title("FDR Corrected P Map")
    plt.imshow(corrected_p_map, vmin=0.95, vmax=1)
    plt.colorbar()
    plt.show()
    print(np.shape(p_map))


#p_map_file = r"/media/matthew/29D46574463D2856/Significance_Testing/Cluster_Testing/Session_Control_Modulation/Window_p_map.npy"

#p_map_file = r"/media/matthew/29D46574463D2856/Significance_Testing/Control_Learning/Hits_Pre_Post_Learning_response_p_value_tensor.npy"
def view_p_map(p_map_file):
    p_map = np.load(p_map_file)

    p_map = np.nan_to_num(p_map)
    rejected, p_map = fdrcorrection(p_map)

    indicies, image_height, image_width = widefield_utils.load_tight_mask()

    p_map = -1 - p_map

    p_map = widefield_utils.create_image_from_data(p_map, indicies, image_height, image_width)


    plt.imshow(p_map,  vmax=2)
    plt.show()

    rejected_map = widefield_utils.create_image_from_data(rejected, indicies, image_height, image_width)
    plt.imshow(rejected_map)
    plt.show()
    print(np.shape(p_map))




#p_map_file = r"/media/matthew/29D46574463D2856/Significance_Testing/Control_Learning_Downsampled/Hits_Pre_Post_Learning_response_p_value_tensor.npy"
p_map_file = r"/media/matthew/29D46574463D2856/Significance_Testing/Mutant_Learning_Downsampled/Hits_Pre_Post_Learning_response_p_value_tensor.npy"
view_p_map_downsized(p_map_file)
