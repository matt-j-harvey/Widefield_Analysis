import nipype
import numpy as np
from nipype.interfaces import fsl
import nibabel
import h5py
import tables
import numpy as np
import os
from tqdm import tqdm
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
from skimage.transform import resize
import resource
import sys

from Widefield_Utils import widefield_utils
from Files import Session_List


def create_nifti_file(data_array):

    # Expand Dims
    data_array = np.expand_dims(data_array, 2)

    # Convert To Nifti
    nifty_image = nibabel.Nifti1Image(data_array, affine=np.eye(4))

    return nifty_image


def load_mask():

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()

    template = np.zeros(image_height * image_width)
    template[indicies] = 1
    template = np.reshape(template, (image_height, image_width))

    template = create_nifti_file(template)

    return template




def load_mask_downsampled():

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()

    indicies, image_height, image_width = downsample_mask_further(indicies, image_height, image_width )

    template = np.zeros(image_height * image_width)
    template[indicies] = 1
    template = np.reshape(template, (image_height, image_width))

    template = create_nifti_file(template)

    return template




def get_cluster_statistics(data_array):

    # Convert To Nifty
    data_array = create_nifti_file(data_array)

    # Load Mask
    mask_array = load_mask()

    # Save These
    nibabel.save(data_array, 'data_array.nii')
    nibabel.save(mask_array, 'mask_array.nii')

    # Estimate smoothness
    smoothest = fsl.SmoothEstimate()
    smoothest.inputs.mask_file = 'mask_array.nii'
    smoothest.inputs.zstat_file = 'data_array.nii'
    smooth = smoothest.run()

    print("Smooth", smooth)

    # Cluster correct
    cluster = fsl.Cluster(in_file=smoothest.inputs.zstat_file,
                          threshold=2.5,
                          pthreshold=0.95,
                          dlh=smooth.outputs.dlh,
                          volume=smooth.outputs.volume,
                          out_threshold_file= 'threshold.nii.gz',
                          out_index_file= 'index.nii.gz',
                          out_localmax_txt_file= 'localmax.txt',
                          )
    clusters = cluster.run()

    thresholded_data = nibabel.load('threshold.nii.gz')
    thresholded_data = thresholded_data.get_data()

    plt.imshow(thresholded_data)
    plt.show()

    print("Thresolded Data", np.shape(thresholded_data))

    print(clusters)



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



def get_cluster_statistic_downsampled(data_array):


    # Convert To Nifty
    data_array = create_nifti_file(data_array)

    # Load Mask
    mask_array = load_mask_downsampled()

    # Save These
    nibabel.save(data_array, 'data_array.nii')
    nibabel.save(mask_array, 'mask_array.nii')

    # Estimate smoothness
    smoothest = fsl.SmoothEstimate()
    smoothest.inputs.mask_file = 'mask_array.nii'
    smoothest.inputs.zstat_file = 'data_array.nii'
    smooth = smoothest.run()

    print("Smooth", smooth)

    # Cluster correct
    cluster = fsl.Cluster(in_file=smoothest.inputs.zstat_file,
                          threshold=2.5,
                          pthreshold=0.95,
                          dlh=smooth.outputs.dlh,
                          volume=smooth.outputs.volume,
                          out_threshold_file= 'threshold.nii.gz',
                          out_index_file= 'index.nii.gz',
                          out_localmax_txt_file= 'localmax.txt',
                          )
    clusters = cluster.run()

    thresholded_data = nibabel.load('threshold.nii.gz')
    thresholded_data = thresholded_data.get_data()

    plt.imshow(thresholded_data)
    plt.show()

    print("Thresolded Data", np.shape(thresholded_data))

    print(clusters)



# Load Z Image
#t_stat_file = r"/media/matthew/29D46574463D2856/Significance_Testing/Control_Learning/Hits_Pre_Post_Learning_responsep_value_tensor.npy"
#t_stat_image = np.load()


#t_stat_array = np.load(r"/media/matthew/29D46574463D2856/Significance_Testing/Control_Learning/Hits_Pre_Post_Learning_response_t_stat_tensor.npy")
#t_stat_array = np.load(r"/media/matthew/29D46574463D2856/Significance_Testing/Mutant_Learning/Hits_Pre_Post_Learning_response_t_stat_tensor.npy")
t_stat_array = np.load(r"/media/matthew/29D46574463D2856/Significance_Testing/Control_Learning_Downsampled/Hits_Pre_Post_Learning_response_t_stat_tensor.npy")

t_stat_array = np.load(r"/home/matthew/Documents/Adil_Meeting_2022_12_01/Control_Contextual_Modulation/n_mouse/Window/t_stats.npy")
t_stat_array = np.load(r"/home/matthew/Documents/Adil_Meeting_2022_12_01/Control_Contextual_Modulation/n_session/Window/t_stats.npy")

"""
# Reshape T Stat Array
indicies, image_height, image_width = widefield_utils.load_tight_mask()
indicies, image_height, image_width = downsample_mask_further(indicies, image_height, image_width)
template = np.zeros(image_height * image_width)
template[indicies] = t_stat_array
t_stat_array = np.reshape(template, (image_height, image_width))

t_stat_array = np.nan_to_num(t_stat_array)

plt.imshow(t_stat_array)
plt.show()

get_cluster_statistic_downsampled(t_stat_array)
"""

# Reshape T Stat Array
indicies, image_height, image_width = widefield_utils.load_tight_mask()
#indicies, image_height, image_width = downsample_mask_further(indicies, image_height, image_width)
template = np.zeros(image_height * image_width)
template[indicies] = t_stat_array
t_stat_array = np.reshape(template, (image_height, image_width))

t_stat_array = np.nan_to_num(t_stat_array)
print("t stat array", np.shape(t_stat_array))
t_stat_array = np.nan_to_num(t_stat_array)





plt.imshow(np.abs(t_stat_array), cmap='jet', vmin=0, vmax=5)
plt.colorbar()
plt.show()
get_cluster_statistics(t_stat_array)
