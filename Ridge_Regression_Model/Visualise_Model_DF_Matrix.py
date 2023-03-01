import numpy as np
import matplotlib.pyplot as plt
import os

from Widefield_Utils import widefield_utils


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def visualise_model_df(base_directory):

    # Load Delta F Matrix
    delta_f_matrix = np.load(os.path.join(base_directory, "Full_Model_Delta_F_Matrix.npy"))

    colourmap = widefield_utils.get_musall_cmap()
    plt.imshow(np.transpose(delta_f_matrix), cmap=colourmap, vmin=-0.05, vmax=0.05)
    forceAspect(plt.gca())
    plt.show()


def visualise_design_matrix(base_directory):

    # Load Delta F Matrix
    design_matrix = np.load(os.path.join(base_directory, "Full_Model_Design_Matrix.npy"))
    design_matrix_magnitude = np.percentile(np.abs(design_matrix), 99)
    colourmap = widefield_utils.get_musall_cmap()
    plt.imshow(np.transpose(design_matrix), cmap=colourmap, vmin=-design_matrix_magnitude, vmax=design_matrix_magnitude)
    plt.colorbar()
    forceAspect(plt.gca())
    plt.show()


base_directory = r"/media/matthew/External_Harddrive_2/Control_Learning_Analysis/Full_Model/NXAK22.1A/2021_09_29_Discrimination_Imaging"
visualise_model_df(base_directory)
visualise_design_matrix(base_directory)