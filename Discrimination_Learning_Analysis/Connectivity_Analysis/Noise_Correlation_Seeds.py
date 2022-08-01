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
from scipy.spatial.distance import cdist

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


def get_activity_tensor(delta_f_matrix, onset_list, start_window, stop_window):

    activity_tensor = []
    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window
        trial_data = delta_f_matrix[trial_start:trial_stop]
        activity_tensor.append(trial_data)

    return activity_tensor


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




def get_lowcut_coefs(w=0.0033, fs=28.):
    b, a = signal.butter(2, w/(fs/2.), btype='highpass');
    return b, a

def perform_lowcut_filter(data, b, a):
    filtered_data = signal.filtfilt(b, a, data, padlen=10000, axis=0)
    return filtered_data

def normalise_delta_f(activity_matrix):

    #activity_matrix = np.transpose(activity_matrix)

    # Subtract Min
    min_vector = np.min(activity_matrix, axis=0)
    activity_matrix = np.subtract(activity_matrix, min_vector)

    # Divide By New Max
    max_vector = np.max(activity_matrix, axis=0)
    activity_matrix = np.divide(activity_matrix, max_vector)

    # Remove Nans and Transpose
    activity_matrix = np.nan_to_num(activity_matrix)
    #activity_matrix = np.transpose(activity_matrix)

    return activity_matrix

def preprocess_activity_matrix(activity_matrix, early_cutoff=3000):

    # Lowcut Filter
    b, a = get_lowcut_coefs()

    # Get Sample Data
    usefull_data = activity_matrix[early_cutoff:]

    # Lowcut Filter
    usefull_data = perform_lowcut_filter(usefull_data, b, a)

    # Normalise
    usefull_data = normalise_delta_f(usefull_data)

    # Remove Early Frames
    activity_matrix[0:early_cutoff] = 0
    activity_matrix[early_cutoff:] = usefull_data

    return activity_matrix


def correlate_one_with_many(one, many):
    c = 1 - cdist(one, many, metric='correlation')[0]
    return c


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

    # Get Activity Tensor
    activity_tensor = get_activity_tensor(delta_f_matrix, onsets_list, start_window, stop_window)

    # Get Noise Tensor
    activity_mean = np.mean(activity_tensor, axis=0)
    activity_tensor = np.subtract(activity_tensor, activity_mean)
    number_of_trials, trial_length, number_of_pixels = np.shape(activity_tensor)
    activity_tensor = np.reshape(activity_tensor, (number_of_trials * trial_length, number_of_pixels))

    # Get Internal and External Tensors
    region_tensor = activity_tensor[:, region_indexes]
    region_tensor = np.mean(region_tensor, axis=1)
    region_tensor = np.reshape(region_tensor, (np.shape(region_tensor)[0], 1))
    external_tensor = activity_tensor[:, external_indexes]

    region_tensor = np.transpose(region_tensor)
    external_tensor = np.transpose(external_tensor)

    # Get Correlation Matrix
    print("REgion Tensor", np.shape(region_tensor))
    print("External Teensor", np.shape(external_tensor))
    correlation_matrix = correlate_one_with_many(region_tensor, external_tensor)

    correlation_map = view_granger_causality(correlation_matrix, external_pixel_positions, downsampled_image_height, downsampled_image_width)

    np.save(os.path.join(base_directory, "Seed_Correlation_Map.npy"), correlation_map)





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
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_03_03_Discrimination_Imaging",
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



pixel_assignments = np.load("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Final_Consensus_Clusters.npy")
plt.imshow(pixel_assignments)
plt.show()


start_window = -42
stop_window = 56
#selected_regions = [56, 2] #v1
#selected_regions = [10, 48] #ant rsc
#selected_regions = [4, 54]
selected_regions = [24 , 36] #M2
#selected_regions = [27 , 34] #M2

number_of_sessions = len(session_list)
for session_index in tqdm(range(number_of_sessions)):
    base_directory = session_list[session_index]
    seed_based_granger_causality(base_directory, start_window, stop_window)