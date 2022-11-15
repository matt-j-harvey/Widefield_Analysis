import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from matplotlib.pyplot import cm
from matplotlib import cm
from matplotlib.pyplot import Normalize

import Trial_Aligned_Utils



def transform_image(image, variable_dictionary):

    angle = variable_dictionary['rotation']
    x_shift = variable_dictionary['x_shift']
    y_shift = variable_dictionary['y_shift']

    transformed_image = np.copy(image)
    transformed_image = ndimage.rotate(transformed_image, angle, reshape=False, prefilter=True)
    transformed_image = np.roll(a=transformed_image, axis=0, shift=y_shift)
    transformed_image = np.roll(a=transformed_image, axis=1, shift=x_shift)

    return transformed_image

def get_transformed_image(base_directory):

    # Load Example Image
    blue_example_image = np.load(os.path.join(base_directory, "Blue_Example_Image.npy"))

    # Load Alignment Dictionary
    alignment_dictionary = np.load(os.path.join(base_directory, "Brain_Alignment_Dictionary.npy"), allow_pickle=True)[()]

    example_image = transform_image(blue_example_image, alignment_dictionary)

    return example_image


def get_atlas_outline_pixels():

    # Load Atlas
    atlas_outline = np.load("/home/matthew/Documents/Github_Code_Clean/Transition_Analysis/Brain_Registration/Atlas_Outlines.npy")

    # Load Atlas Transformation Dict
    transformation_dict = np.load("/home/matthew/Documents/Github_Code_Clean/Transition_Analysis/Brain_Registration/Atlas_Alignment_Dictionary.npy", allow_pickle=True)[()]

    # Transform Atlas
    atlas_outline = Trial_Aligned_Utils.transform_mask_or_atlas(atlas_outline, transformation_dict)

    plt.imshow(atlas_outline)
    plt.show()

    atlas_pixels = np.nonzero(atlas_outline)
    return atlas_pixels


indicies, image_height, image_width = Trial_Aligned_Utils.load_tight_mask()
number_of_indicies = np.shape(indicies)[1]
print("Indiices shape", np.shape(indicies))

# Load Example Brain Image
example_mouse = r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_02_Transition_Imaging"
transformed_image = get_transformed_image(example_mouse)
transformed_image = np.ndarray.flatten(transformed_image)

brain_colourmap = cm.ScalarMappable(cmap=cm.get_cmap('Greys_r'), norm=Normalize(vmin=0, vmax=65000))

# Load ROI Pixels
roi_pixels = np.load(r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_23_Transition_Imaging/Selected_ROI.npy")

brain_values = transformed_image[indicies]
brain_values = brain_colourmap.to_rgba(brain_values)
brain_values[roi_pixels] = (1,1,1,1)

image = np.zeros((image_height * image_width, 4))
image[:, :] = (0, 0, 0, 1)
image[indicies] = brain_values

# Add Atlas
atlas_outline_pixels = get_atlas_outline_pixels()

image = np.reshape(image, (image_height, image_width, 4))
#image[atlas_outline_pixels] = (1,1,1,1)

plt.axis('off')
plt.imshow(image)
plt.show()


