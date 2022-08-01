import numpy as np
import tables
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression, Ridge
from datetime import datetime
from skimage.transform import downscale_local_mean
from tqdm import tqdm

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
        trial_data = delta_f_matrix[trial_start:trial_stop, internal_indicies]
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
    prediction_tensor = np.mean(prediction_tensor, axis=2)
    print("Lagged Region Tensor", np.shape(lagged_region_tensor))

    lagged_external_tensor = np.moveaxis(lagged_external_tensor, (0, 1, 2, 3), (0, 2, 1, 3))
    lagged_region_tensor = np.moveaxis(lagged_region_tensor, (0, 1, 2), (0, 2, 1))

    # Reshape Tensor
    p1, p2 = np.shape(prediction_tensor)
    l1, l2, l3, l4 = np.shape(lagged_external_tensor)
    lr1, lr2, lr3 = np.shape(lagged_region_tensor)

    prediction_tensor = np.reshape(prediction_tensor, (p1 * p2, 1))
    lagged_external_tensor = np.reshape(lagged_external_tensor, (l1 * l2, l3, l4))
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


def create_stimuli_dictionary():

    channel_index_dictionary = {
        "Photodiode"        :0,
        "Reward"            :1,
        "Lick"              :2,
        "Visual 1"          :3,
        "Visual 2"          :4,
        "Odour 1"           :5,
        "Odour 2"           :6,
        "Irrelevance"       :7,
        "Running"           :8,
        "Trial End"         :9,
        "Camera Trigger"    :10,
        "Camera Frames"     :11,
        "LED 1"             :12,
        "LED 2"             :13,
        "Mousecam"          :14,
        "Optogenetics"      :15,
        }

    return channel_index_dictionary


def get_granger_causality(prediction_tensor, behavioural_tensor, lagged_external_tensor, lagged_region_tensor, number_of_pixels):

    print("Performing Granger Causality")
    print("Prediction Tenor", np.shape(prediction_tensor))
    print("Lagged External Tnesor", np.shape(lagged_external_tensor))
    print("Lagged Region Tensor", np.shape(lagged_region_tensor))

    # Get Partial Fit
    partial_design_matrix = np.hstack([behavioural_tensor, lagged_region_tensor])
    partial_model = Ridge()
    partial_model.fit(X=partial_design_matrix, y=prediction_tensor)
    partial_prediction = partial_model.predict(X=partial_design_matrix)
    partial_error = np.subtract(prediction_tensor, partial_prediction)
    partial_error = np.var(partial_error, axis=0)

    granger_causality_vector = []
    for pixel_index in tqdm(range(number_of_pixels)):

        lagged_pixel_trace = lagged_external_tensor[:, :, pixel_index]
        full_design_matrix = np.hstack([behavioural_tensor, lagged_region_tensor, lagged_pixel_trace])
        full_model = Ridge()
        full_model.fit(X=full_design_matrix, y=prediction_tensor)
        full_prediction = full_model.predict(X=full_design_matrix)
        full_error = np.subtract(prediction_tensor, full_prediction)
        full_error = np.var(full_error, axis=0)


        granger_causality = np.log(partial_error / full_error)
        granger_causality_vector.append(granger_causality)


    return granger_causality_vector


def view_granger_causality(granger_vector, external_indicies, image_height, image_width):

    gc_map = np.zeros(image_height * image_width)
    for index in range(len(external_indicies)):
        pixel_pos = external_indicies[index]
        gc_value = granger_vector[index]
        gc_map[pixel_pos] = gc_value

    gc_map = np.reshape(gc_map, (image_height, image_width))
    gc_map = ndimage.gaussian_filter(gc_map, sigma=1)

    return gc_map



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



def get_behavioural_tensor(onsets_list, start_window, stop_window, downsampled_ai_data):

    # Get Lick and Running Traces
    stimuli_dictionary = create_stimuli_dictionary()
    lick_trace = downsampled_ai_data[stimuli_dictionary["Lick"]]
    running_trace = downsampled_ai_data[stimuli_dictionary["Running"]]

    # Get Lick and Running Tensor
    lick_tensor = []
    running_tensor = []

    for onset in onsets_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window
        trial_running_data = running_trace[trial_start:trial_stop]
        trial_lick_data = lick_trace[trial_start:trial_stop]

        running_tensor.append(trial_running_data)
        lick_tensor.append(trial_lick_data)

    lick_tensor = np.array(lick_tensor)
    running_tensor = np.array(running_tensor)

    # Create Stimuli Tensor
    number_of_stimuli = len(onsets_list)
    trial_length = stop_window - start_window
    stimuli_regressor = np.zeros((number_of_stimuli * trial_length, trial_length))
    for trial in range(number_of_stimuli):
        trial_start = trial * trial_length
        trial_stop = trial_start + trial_length
        stimuli_regressor[trial_start:trial_stop, :] = np.eye(trial_length)

    lick_tensor = np.reshape(lick_tensor, (number_of_stimuli * trial_length, 1))
    running_tensor = np.reshape(running_tensor, (number_of_stimuli * trial_length, 1))

    behaviour_tensor = np.hstack([stimuli_regressor, running_tensor, lick_tensor])

    return behaviour_tensor



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

    # Get Selected Region Indicies
    region_indexes, region_pixel_positions, external_indexes, external_pixel_positions = get_selected_region_indicies(cluster_assignments, selected_regions, downsampled_indicies)
    print("External pixels", len(external_indexes))
    #view_granger_causality(delta_f_matrix[4001, external_indexes], external_pixel_positions, downsampled_image_height, downsampled_image_width)

    # Get Trial Tensor
    order = 10
    prediction_tensor, lagged_external_tensor, lagged_region_tensor = load_lagged_trial_tensor(delta_f_matrix, onsets_list, start_window, stop_window, region_indexes, external_indexes, order)

    # Get Behavioural Tensor
    downsampled_ai_data = np.load(os.path.join(base_directory, "Downsampled_AI_Matrix_Framewise.npy"))
    behaviour_tensor = get_behavioural_tensor(onsets_list, start_window, stop_window, downsampled_ai_data)

    # Get Granger Causality
    granger_vector = get_granger_causality(prediction_tensor, behaviour_tensor, lagged_external_tensor, lagged_region_tensor, len(external_indexes))
    print("Granger Vector ", np.shape(granger_vector))

    gc_map = view_granger_causality(granger_vector, external_pixel_positions, downsampled_image_height, downsampled_image_width)

    np.save(os.path.join(base_directory, "GC_Map.npy"), gc_map)






session_list = [

    # Controls 46 sessions

    # 78.1A - 6
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_16_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_17_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_19_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_21_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_24_Discrimination_Imaging",

    # 78.1D - 8
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_14_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_15_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_16_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_17_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_19_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_21_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_23_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_25_Discrimination_Imaging",

    # 4.1B - 7
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
    #r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging",

    # 22.1A - 7
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_09_25_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_09_29_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_01_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_03_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_05_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_07_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging",

    # 14.1A - 6
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_04_29_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_01_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_03_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_05_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_07_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_09_Discrimination_Imaging",

    # 7.1B - 12
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_01_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_03_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_05_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_07_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_09_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_11_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_13_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_15_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_17_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_19_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_22_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_24_Discrimination_Imaging",


    # Mutants

    # 4.1A - 15
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_08_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_10_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_12_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_14_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_16_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_18_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_23_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_25_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_27_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_03_01_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_03_03_Discrimination_Imaging",\
]
session_list = [
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging",

    # 20.1B - 11
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_09_28_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_09_30_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_09_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_11_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_13_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_15_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_17_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging",

    # 24.1C - 10
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_20_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_22_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_24_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_26_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_28_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_30_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_10_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_10_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_10_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_10_08_Discrimination_Imaging",

    # NXAK16.1B - 16
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_04_30_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_08_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_10_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_12_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_14_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_16_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_18_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_20_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_22_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_24_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_26_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_06_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_06_15_Discrimination_Imaging",

    # 10.1A - 8
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_04_30_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_08_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_10_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_12_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_14_Discrimination_Imaging"


    # 71.2A - 16

]


start_window = -42
stop_window = 56
#selected_regions = [56, 2] #v1
#selected_regions = [10, 48] #ant rsc
#selected_regions = [4, 54]
selected_regions = [24 , 36] #M2

for base_directory in session_list:
    seed_based_granger_causality(base_directory, start_window, stop_window)