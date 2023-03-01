import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.segmentation import watershed

# Origional Template
atlas_file_location = r"/home/matthew/Documents/Github_Code (copy)/Widefield_Preprocessing/Allen_Atlas_Templates/Allen_Atlas_Mapping.npy"
atlas_outlines_location = r"/home/matthew/Documents/Github_Code_Clean/Widefield_Analysis/Files/Atlas_Outlines.npy"

# New Template
atlas_regions = np.load(atlas_file_location)
plt.imshow(atlas_regions)
plt.show()

atlas_outlines = np.load(atlas_outlines_location)
plt.imshow(atlas_outlines)
plt.show()

laballed_matrix = watershed(atlas_outlines)
plt.imshow(laballed_matrix)
plt.show()

number_of_regions = np.max(laballed_matrix)

"""
for region_index in range(number_of_regions):
    region_mask = np.where(laballed_matrix == region_index, 1, 0)
    region_mask_outlines = np.add(region_mask, atlas_outlines)
    plt.imshow(region_mask_outlines)
    plt.title(str(region_index))
    plt.show()
"""

region_dict = {

    "pixel_labels":laballed_matrix,

    "olfactory_bulb": 2,

    "m2_left": 5,
    "m2_right": 7,

    "m1_left": 9,
    "m1_right": 10,

    "s1_upper_limb_left": 12,
    "s1_upper_limb_right": 14,

    "s1_lower_limb_left":17,
    "s1_lower_limb_right": 19,

    "s1_barrel_left": 21,
    "s1_barrel_right": 27,

    "s1_trunk_left": 29,
    "s1_trunk_right": 35,

    "rsc_lateral_left": 31,
    "rsc_lateral_right": 34,

    "retrosplenial": 32,

    "anterior_visual_left": 40,
    "anterior_visual_right": 42,

    "rostrolateral_visual_left": 43,
    "rostrolateral_visual_right": 47,

    "anteromedial_visual_left": 45,
    "anteromedial_visual_right": 46,

    "primary_visual_left": 48,
    "primary_visual_right": 51,

    "anterolateral_visual_left": 49,
    "anterolateral_visual_right": 53,

    "posteriomedial_visual_left": 50,
    "posteriomedial_visual_right": 52,
}

np.save(os.path.join(r"/home/matthew/Documents/Github_Code_Clean/Widefield_Analysis/Files/Allen_Region_Dict.npy"), region_dict)