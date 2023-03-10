import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.transform import warp, resize, rescale


def transform_atlas_regions(image, variable_dictionary, image_height, image_width):

    unique_values = list(set(np.unique(image)))
    transformed_mask = np.zeros((image_height, image_width))

    for value in unique_values:
        if value != 0:
            value_mask = np.where(image == value, 1, 0)

            value_mask = transform_image(value_mask, variable_dictionary)

            value_indicies = np.where(value_mask > 0.1)
            transformed_mask[value_indicies] = value

    return transformed_mask


def transform_image(image, variable_dictionary):

    # Settings
    background_size = 1000
    background_offset = 200
    origional_height, origional_width = np.shape(image)
    window_y_start = background_offset
    window_y_stop = window_y_start + origional_height
    window_x_start = background_offset
    window_x_stop = window_x_start + origional_width

    # Unpack Transformation Details
    angle = variable_dictionary['rotation']
    x_shift = variable_dictionary['x_shift']
    y_shift = variable_dictionary['y_shift']
    scale_factor = variable_dictionary['zoom']

    # Copy
    transformed_image = np.copy(image)

    # Scale
    transformed_image = rescale(transformed_image, 1 + scale_factor, anti_aliasing=False, preserve_range=True)

    # Rotate
    transformed_image = ndimage.rotate(transformed_image, angle, reshape=False, prefilter=True)

    # Translate
    background = np.zeros((background_size, background_size))
    new_height, new_width = np.shape(transformed_image)

    y_start = background_offset + y_shift
    y_stop = y_start + new_height

    x_start = background_offset + x_shift
    x_stop = x_start + new_width

    background[y_start:y_stop, x_start:x_stop] = transformed_image

    # Get Normal Sized Window
    transformed_image = background[window_y_start:window_y_stop, window_x_start:window_x_stop]

    return transformed_image





def align_atlas_to_current_session(base_directory, template_directory):

    # Load Allen Reigon Atlas
    atlas_regions_location = os.path.join(template_directory, "Atlas_Regions_Aligned_To_Sign_Map.npy")
    atlas_regions = np.load(atlas_regions_location)
    plt.imshow(atlas_regions)
    plt.show()

    # Load Alignment Dictionary
    current_session_alignment_dictionary = np.load(os.path.join(base_directory, "Opto_Atlas_Alignment_Dictionary.npy"), allow_pickle=True)[()]
    print("alignment Dictionary", current_session_alignment_dictionary)

    matching_session = r"/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.1D/2023_02_27_Switching_v1_inhibition/Blue_Example_Image_Full_Size.npy"
    matching_imaging = np.load(matching_session)

    # Transform Atlas
    transformed_regions = transform_atlas_regions(atlas_regions, current_session_alignment_dictionary, 600, 608)

    plt.imshow(matching_imaging, alpha=1, cmap="Greys_r")
    plt.imshow(transformed_regions,  alpha=0.2, cmap="prism")
    plt.show()

    # Check This
    np.save(os.path.join(base_directory, "Session_Aligned_Atlas_Regions.npy"), transformed_regions)



base_directory = r"/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.1D/2023_02_27_Switching_v1_inhibition"
template_directory = r"/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.1D/2022_12_14_Retinotopy_Left"
align_atlas_to_current_session(base_directory, template_directory)