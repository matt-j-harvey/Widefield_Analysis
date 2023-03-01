import numpy as np
import scipy.io as sio
from skimage.feature import canny
from skimage.morphology import binary_dilation
import matplotlib.pyplot as plt



def convert_mat_atlas_to_numpy_array(input_file, output_file):
    atlas = sio.loadmat(input_file)['atlas'].astype(float)
    np.save(output_file, atlas)


def get_atlas_outlines(atlas_file, output_file):

    atlas = np.load(atlas_file)
    plt.imshow(atlas)
    plt.show()

    edges = canny(atlas)
    edges = binary_dilation(edges)
    plt.imshow(edges)
    plt.show()

    np.save(output_file, edges)


input_file = r"/home/matthew/Documents/Allen_Atlas_Templates/churchland_atlas.mat"
output_file = r"/home/matthew/Documents/Allen_Atlas_Templates/churchland_atlas.npy"
output_file_edges = r"/home/matthew/Documents/Allen_Atlas_Templates/churchland_outlines.npy"
convert_mat_atlas_to_numpy_array(input_file, output_file)
get_atlas_outlines(output_file, output_file_edges)