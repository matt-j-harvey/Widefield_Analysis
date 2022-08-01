import numpy as np
import tables
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression, Ridge
from datetime import datetime
from skimage.transform import downscale_local_mean


def remove_early_onsets(onsets_list):

    thresholded_onsets = []
    for onset in onsets_list:
        if onset > 3000:
            thresholded_onsets.append(onset)

    return thresholded_onsets


def transform_clusters(clusters, variable_dictionary, invert=False):

    # Unpack Dict
    angle = variable_dictionary['rotation']
    x_shift = variable_dictionary['x_shift']
    y_shift = variable_dictionary['y_shift']

    # Invert
    if invert == True:
        angle = -1 * angle
        x_shift = -1 * x_shift
        y_shift = -1 * y_shift

    transformed_clusters = np.zeros(np.shape(clusters))

    unique_clusters = list(np.unique(clusters))
    for cluster in unique_clusters:
        cluster_mask = np.where(clusters == cluster, 1, 0)
        cluster_mask = ndimage.rotate(cluster_mask, angle, reshape=False, prefilter=True)
        cluster_mask = np.roll(a=cluster_mask, axis=0, shift=y_shift)
        cluster_mask = np.roll(a=cluster_mask, axis=1, shift=x_shift)
        cluster_indicies = np.nonzero(cluster_mask)
        transformed_clusters[cluster_indicies] = cluster

    return transformed_clusters


def load_generous_mask(base_directory):

    # Loads the mask for a video, returns a list of which pixels are included, as well as the original image height and width
    mask = np.load(os.path.join(base_directory, "Generous_Mask.npy"))

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask>0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width


def downsample_generous_mask(base_directory):

    # Loads the mask for a video, returns a list of which pixels are included, as well as the original image height and width
    mask = np.load(os.path.join(base_directory, "Generous_Mask.npy"))
    mask = downscale_local_mean(mask, (4,4))

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask>0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width



def get_selected_region_indicies(cluster_assignments, selected_regions, indicies):

    region_indexes = []
    region_pixel_positions = []

    external_indexes = []
    external_pixel_positions = []

    # Flatten Cluster Assigmnets
    cluster_assignments = np.reshape(cluster_assignments, (np.shape(cluster_assignments)[0] * np.shape(cluster_assignments)[1]))

    for index in range(len(indicies)):
        pixel_position = indicies[index]

        if cluster_assignments[pixel_position] in selected_regions:
            region_pixel_positions.append(pixel_position)
            region_indexes.append(index)

        else:
            external_pixel_positions.append(pixel_position)
            external_indexes.append(index)

    return region_indexes, region_pixel_positions, external_indexes, external_pixel_positions


def load_lagged_trial_tensor(delta_f_matrix, onset_list, start_window, stop_window, internal_indicies, external_indicies, order):

    prediction_tensor = []
    lagged_external_tensor = []
    lagged_region_tensor = []

    count = 0
    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window
        trial_data = delta_f_matrix[trial_start:trial_stop, external_indicies]
        prediction_tensor.append(trial_data)

        onset_external_lags = []
        onset_region_lags = []
        for lag in range(1, order):
            trial_start = trial_start - lag
            trial_stop = trial_stop - lag

            lagged_data = delta_f_matrix[trial_start:trial_stop]
            lagged_external_data = lagged_data[:, external_indicies]
            lagged_region_data = lagged_data[:, internal_indicies]

            onset_external_lags.append(lagged_external_data)
            onset_region_lags.append(lagged_region_data)

        lagged_external_tensor.append(onset_external_lags)
        lagged_region_tensor.append(onset_region_lags)

        count += 1

    prediction_tensor = np.array(prediction_tensor)
    lagged_external_tensor = np.array(lagged_external_tensor)
    lagged_region_tensor = np.array(lagged_region_tensor)

    print("Prediction Tenor", np.shape(prediction_tensor))
    print("Lagged External Tnesor", np.shape(lagged_external_tensor))
    print("Lagged Region Tensor", np.shape(lagged_region_tensor))

    # Get Region Mean
    lagged_region_tensor = np.mean(lagged_region_tensor, axis=3)
    print("Lagged Region Tensor", np.shape(lagged_region_tensor))

    lagged_external_tensor = np.moveaxis(lagged_external_tensor, (0, 1, 2, 3), (0, 2, 1, 3))
    lagged_region_tensor = np.moveaxis(lagged_region_tensor, (0, 1, 2), (0, 2, 1))

    # Reshape Tensor
    p1, p2, p3 = np.shape(prediction_tensor)
    l1, l2, l3, l4 = np.shape(lagged_external_tensor)
    lr1, lr2, lr3 = np.shape(lagged_region_tensor)

    prediction_tensor = np.reshape(prediction_tensor, (p1 * p2, p3))
    lagged_external_tensor = np.reshape(lagged_external_tensor, (l1 * l2, l3 * l4))
    lagged_region_tensor = np.reshape(lagged_region_tensor, (lr1 * lr2, lr3))

    print("Prediction Tenor", np.shape(prediction_tensor))
    print("Lagged External Tnesor", np.shape(lagged_external_tensor))
    print("Lagged Region Tensor", np.shape(lagged_region_tensor))

    return prediction_tensor, lagged_external_tensor, lagged_region_tensor




def get_chunk_structure(chunk_size, array_size):
    number_of_chunks = int(np.ceil(array_size / chunk_size))
    remainder = array_size % chunk_size

    # Get Chunk Sizes
    chunk_sizes = []
    if remainder == 0:
        for x in range(number_of_chunks):
            chunk_sizes.append(chunk_size)

    else:
        for x in range(number_of_chunks - 1):
            chunk_sizes.append(chunk_size)
        chunk_sizes.append(remainder)

    # Get Chunk Starts
    chunk_starts = []
    chunk_start = 0
    for chunk_index in range(number_of_chunks):
        chunk_starts.append(chunk_size * chunk_index)

    # Get Chunk Stops
    chunk_stops = []
    chunk_stop = 0
    for chunk_index in range(number_of_chunks):
        chunk_stop += chunk_sizes[chunk_index]
        chunk_stops.append(chunk_stop)

    return number_of_chunks, chunk_sizes, chunk_starts, chunk_stops



def get_granger_causality(prediction_tensor, lagged_external_tensor, lagged_region_tensor, number_of_pixels):
    print("Performing Granger Causality")
    print("Prediction Tenor", np.shape(prediction_tensor))
    print("Lagged External Tnesor", np.shape(lagged_external_tensor))
    print("Lagged Region Tensor", np.shape(lagged_region_tensor))

    chunk_size = 5000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = get_chunk_structure(chunk_size, number_of_pixels)

    granger_causality_vector = []
    for chunk_index in range(number_of_chunks):

        print("Chunk ", chunk_index, " of ", number_of_chunks, " at ", datetime.now())

        # Get Chunk Data
        chunk_start = chunk_starts[chunk_index]
        chunk_stop = chunk_stops[chunk_index]
        fit_tensor = prediction_tensor[:, chunk_start:chunk_stop]
        print("Fit tensor", np.shape(fit_tensor), fit_tensor.size)

        partial_model = Ridge()
        partial_model.fit(X=lagged_external_tensor, y=fit_tensor)
        partial_prediction = partial_model.predict(X=lagged_external_tensor)
        partial_error = np.subtract(fit_tensor, partial_prediction)
        partial_error = np.var(partial_error, axis=0)
        print("Partial error", np.shape(partial_error))

        full_model = Ridge()
        full_design_matrix = np.hstack([lagged_external_tensor, lagged_region_tensor])
        print("Full Design Matrix Shape", np.shape(full_design_matrix))
        full_model.fit(X=full_design_matrix, y=fit_tensor)
        full_prediction = full_model.predict(X=full_design_matrix)
        full_error = np.subtract(fit_tensor, full_prediction)
        full_error = np.var(full_error, axis=0)
        print("full error", np.shape(full_error))

        granger_causality = np.log(partial_error / full_error)
        print("granger Causality", np.shape(granger_causality))


        granger_causality_vector.append(granger_causality)

    granger_causality_vector = np.concatenate(granger_causality_vector)
    return granger_causality_vector


def view_granger_causality(granger_vector, external_indicies, image_height, image_width):

    gc_map = np.zeros(image_height * image_width)
    for index in range(len(external_indicies)):
        pixel_pos = external_indicies[index]
        gc_value = granger_vector[index]
        gc_map[pixel_pos] = gc_value

    gc_map = np.reshape(gc_map, (image_height, image_width))
    gc_map = ndimage.gaussian_filter(gc_map, sigma=1)
    plt.imshow(gc_map)
    plt.show()



def downsample_data(data_tensor, full_height, full_width, full_indicies, downsampled_height, downsampled_width, downsampled_indicies):

    downsampled_data = []

    for frame in data_tensor:
        template = np.zeros((full_height * full_width))
        template[full_indicies] = frame
        template = np.reshape(template, (full_height, full_width))
        template = downscale_local_mean(template, (4,4))
        template = np.ndarray.reshape(template, (downsampled_height * downsampled_width))
        frame_data = template[downsampled_indicies]
        downsampled_data.append(frame_data)

    return downsampled_data





def seed_based_granger_causality(base_directory, start_window, stop_window, order=20):

    # Load Delta F Matrix
    delta_f_matrix = np.load(os.path.join(base_directory, "Downsampled_Delta_F.npy"))
    number_of_frames, number_of_pixels = np.shape(delta_f_matrix)
    print("Frames:", number_of_frames, "Pixels:", number_of_pixels)

    # Load Mask
    downsampled_indicies, downsampled_image_height, downsampled_image_width = downsample_generous_mask(base_directory)

    # Load Onsets
    onsets_list = np.load(os.path.join(base_directory, "Stimuli_Onsets", "visual_1_correct_onsets.npy"))
    onsets_list = remove_early_onsets(onsets_list)
    print("Numver of onsets", len(onsets_list))
    print("Total Timepoints: ", len(onsets_list) * (stop_window - start_window))

    # Load Cluster Assignments
    cluster_assignments = np.load(r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Final_Consensus_Clusters.npy")
    cluster_alignment_dictionary = np.load(os.path.join(base_directory, "Cluster_Alignment_Dictionary.npy"), allow_pickle=True)[()]

    # Transform Them
    cluster_assignments = transform_clusters(cluster_assignments, cluster_alignment_dictionary, invert=True)
    cluster_assignments = downscale_local_mean(cluster_assignments, (4, 4))
    cluster_assignments = np.ndarray.astype(cluster_assignments, int)
    plt.imshow(cluster_assignments)
    plt.show()

    # Get Selected Region Indicies
    region_indexes, region_pixel_positions, external_indexes, external_pixel_positions = get_selected_region_indicies(cluster_assignments, selected_regions, downsampled_indicies)
    print("External pixels", len(external_indexes))
    #view_granger_causality(delta_f_matrix[4001, external_indexes], external_pixel_positions, downsampled_image_height, downsampled_image_width)

    # Get Trial Tensor
    order = 5
    prediction_tensor, lagged_external_tensor, lagged_region_tensor = load_lagged_trial_tensor(delta_f_matrix, onsets_list, start_window, stop_window, region_indexes, external_indexes, order)

    # Get Granger Causality
    granger_vector = get_granger_causality(prediction_tensor, lagged_external_tensor, lagged_region_tensor, len(external_indexes))
    print("Granger Vector ", np.shape(granger_vector))

    view_granger_causality(granger_vector, external_pixel_positions, downsampled_image_height, downsampled_image_width)


start_window = -20
stop_window = 20
#selected_regions = [56, 2] #v1
#selected_regions = [10, 48] #ant rsc
selected_regions = [4, 54]
selected_regions = [24 , 36] #M2
base_directory = r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_04_08_Transition_Imaging"
seed_based_granger_causality(base_directory, start_window, stop_window)