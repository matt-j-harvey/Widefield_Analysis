import numpy as np


def get_region_mean_tensor(tensor, pixel_assignments, selected_regions):

    # Get Selected Pixels
    selected_pixels = []
    for region in selected_regions:
        region_mask = np.where(pixel_assignments == region, 1, 0)
        region_indicies = np.nonzero(region_mask)[0]
        for index in region_indicies:
            selected_pixels.append(index)
    selected_pixels.sort()

    # Get Tensor Structure
    number_of_trials = np.shape(tensor)[0]
    number_of_timepoints = np.shape(tensor)[1]
    number_of_pixels = np.shape(tensor)[2]

    # Get Mean Response For Region For Each Trial
    region_trace = tensor[:, :, selected_pixels]
    region_mean = np.mean(region_trace, axis=2)

    region_mean = np.expand_dims(region_mean, 2)

    return region_mean
