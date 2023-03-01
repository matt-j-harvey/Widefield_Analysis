import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from skimage.transform import resize

import Noise_Correlation_Utils




def downsample_mask_further(indicies, image_height, image_width, downsample_size=100):
    template = np.zeros((image_height*image_width))
    template[indicies] = 1
    template = np.reshape(template, (image_height, image_width))
    template = resize(template, (downsample_size, downsample_size), anti_aliasing=True)
    template = np.reshape(template, (downsample_size * downsample_size))
    template = np.where(template > 0.5, 1, 0)
    template_indicies = np.nonzero(template)
    return template_indicies, downsample_size, downsample_size


mean_matrix_save_directory = r"/media/matthew/29D46574463D2856/Widefield_Analysis/Noise_Correlation_Analysis/Mean_Modulation_Matricies"
correlation_matrix = np.load(os.path.join(mean_matrix_save_directory, "Mean_Control_Noise_Modulation.npy"))
correlation_matrix = np.nan_to_num(correlation_matrix)

model=PCA(n_components=5)

model.fit(correlation_matrix)

# Load Combined Mask
indicies, image_height, image_width = Noise_Correlation_Utils.load_tight_mask()

# Downsample Mask Further
indicies, image_height, image_width = downsample_mask_further(indicies, image_height, image_width)

colourmap = Noise_Correlation_Utils.get_musall_cmap()
for component in model.components_:
    component_image = Noise_Correlation_Utils.create_image_from_data(component, indicies, image_height, image_width)
    plt.imshow(component_image, cmap=colourmap, vmin=-0.02, vmax=0.02)
    plt.show()