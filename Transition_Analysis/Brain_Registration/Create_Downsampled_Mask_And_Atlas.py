import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale
from skimage.morphology import binary_dilation

mask_location = "/home/matthew/Documents/Allen_Atlas_Templates/Mask_Array.npy"
atlas_outline_location = "/home/matthew/Documents/Allen_Atlas_Templates/New_Outline.npy"

# Downsample Mask
mask_array = np.load(mask_location)
mask_array = rescale(mask_array, 0.5, preserve_range=True, anti_aliasing=False)
mask_array = np.where(mask_array > 0.1, 1, 0)
np.save(r"/home/matthew/Documents/Github_Code_Clean/Transition_Analysis/Brain_Registration/Tight_Mask.npy", mask_array)

# Downsample Atlas
atlas_array = np.load(atlas_outline_location)
atlas_array = binary_dilation(atlas_array)
atlas_array = rescale(atlas_array, 0.5, preserve_range=True, anti_aliasing=False)
atlas_array = np.where(atlas_array > 0, 1, 0)
np.save(r"/home/matthew/Documents/Github_Code_Clean/Transition_Analysis/Brain_Registration/Atlas_Outlines.npy", atlas_array)