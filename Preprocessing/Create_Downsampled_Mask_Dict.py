import os
import numpy as np
from tqdm import tqdm
from skimage.transform import resize

def create_downsampled_mask_dict(base_directory):

    mask = np.load(os.path.join(base_directory, "Generous_Mask.npy"))

    # Transform Mask
    mask = resize(mask, (300, 304), preserve_range=True, order=0, anti_aliasing=True)

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask > 0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    mask_dict = {
    "indicies":indicies,
    "image_height":image_height,
    "image_width":image_width
    }

    np.save(os.path.join(base_directory, "Downsampled_mask_dict.npy"), mask_dict)



