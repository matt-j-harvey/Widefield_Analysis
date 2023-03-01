import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale
from skimage.morphology import binary_dilation

import Registration_Utils

# Load Tight Mask and Alignment Dictionary
tight_mask = np.load("Tight_Mask.npy")
mask_alignment_dictionary = np.load("Tight_Mask_Alignment_Dictionary.npy", allow_pickle=True)[()]

# Align Mask
tight_mask = Registration_Utils.transform_mask_or_atlas(tight_mask, mask_alignment_dictionary)

image_height, image_width = np.shape(tight_mask)
tight_mask = np.reshape(tight_mask, image_height * image_width)
indicies = np.nonzero(tight_mask)

# Create Tight Mask Dict
tight_mask_dict = {
    "image_height":image_height,
    "image_width":image_width,
    "indicies":indicies
}

np.save("Tight_Mask_Dict.npy", tight_mask_dict)

plt.imshow(tight_mask)
plt.show()

