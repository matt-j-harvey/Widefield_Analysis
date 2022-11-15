import numpy as np
import tables
import os
import pandas as pd
from scipy import ndimage
from skimage.transform import resize, warp, rescale
from matplotlib.colors import LinearSegmentedColormap

def flatten(l):
    return [item for sublist in l for item in sublist]


def load_downsampled_mask(base_directory):
    mask_dict = np.load(os.path.join(base_directory, "Downsampled_mask_dict.npy"), allow_pickle=True)[()]
    indicies = mask_dict["indicies"]
    image_height = mask_dict["image_height"]
    image_width = mask_dict["image_width"]
    return indicies, image_height, image_width


def load_mouse_sessions(mouse_name, session_type):

    # This Is The Location of My Experimental Logbook
    logbook_file_location = r"/home/matthew/Documents/Experiment_Logbook.ods"

    #  Read Logbook As A Dataframe
    logbook_dataframe = pd.read_excel(logbook_file_location, engine="odf")

    # Return A List Of the File Directories Of Sessions Matching The Mouse Name and Session Type
    selected_sessions = logbook_dataframe.loc[(logbook_dataframe["Mouse"] == mouse_name) & (logbook_dataframe["Session Type"] == session_type), ["Filepath"]].values.tolist()

    # Flatten The Subsequent Nested List
    selected_sessions = flatten(selected_sessions)

    return selected_sessions

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

def invert_dictionary(dictionary):
    inv_map = {v: k for k, v in dictionary.items()}
    return inv_map


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

def create_image_from_data(data, indicies, image_height, image_width):
    template = np.zeros((image_height, image_width))
    data = np.nan_to_num(data)
    np.put(template, indicies, data)
    image = np.ndarray.reshape(template, (image_height, image_width))

    return image

def get_musall_cmap():
    cmap = LinearSegmentedColormap.from_list('mycmap', [

        (0, 0.87, 0.9, 1),
        (0, 0, 1, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 1, 0, 1),

    ])

    return cmap


def zoom_transform(image, zoom_scale_factor, u, v):
        v = v * zoom_scale_factor
        u = u * zoom_scale_factor
        nr, nc = image.shape
        row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
        image_warp = warp(image, np.array([row_coords + v, col_coords + u]), mode='edge', preserve_range=True)
        return image_warp

"""
def transform_image(image, variable_dictionary, invert=False):

    # Unpack Dict
    angle = variable_dictionary['rotation']
    x_shift = variable_dictionary['x_shift']
    y_shift = variable_dictionary['y_shift']
    zoom = variable_dictionary['zoom']

    # Inverse
    if invert == True:
        angle = -1 * angle
        x_shift = -1 * x_shift
        y_shift = -1 * y_shift

    transformed_image = np.copy(image)
    transformed_image = np.nan_to_num(transformed_image)
    transformed_image = ndimage.rotate(transformed_image, angle, reshape=False, prefilter=True, order=1)

    transformed_image = np.roll(a=transformed_image, axis=0, shift=y_shift)
    transformed_image = np.roll(a=transformed_image, axis=1, shift=x_shift)

    # Zoom
    u = np.load("/home/matthew/Documents/Github_Code_Clean/Preprocessing/Brain_Alignment/zoom_optic_flow_u.npy")
    v = np.load("/home/matthew/Documents/Github_Code_Clean/Preprocessing/Brain_Alignment/zoom_optic_flow_v.npy")
    transformed_image = zoom_transform(transformed_image, zoom, u, v)

    return transformed_image
"""


def transform_image(image, variable_dictionary, invert=False):

    # Settings
    background_size = 1000
    background_offset = 200
    origional_height, origional_width = np.shape(image)
    window_y_start = background_offset
    window_y_stop = window_y_start + origional_height
    window_x_start = background_offset
    window_x_stop = window_x_start + origional_width

    # Unpack Transformation Details
    angle = variable_dictionary['rotation']
    x_shift = variable_dictionary['x_shift']
    y_shift = variable_dictionary['y_shift']
    scale_factor = variable_dictionary['zoom']

    # Inverse
    if invert == True:
        angle = -1 * angle
        x_shift = -1 * x_shift
        y_shift = -1 * y_shift
        scale_factor = 1 - scale_factor
    else:
        scale_factor = 1 + scale_factor

    # Copy
    transformed_image = np.copy(image)

    # Scale
    transformed_image = rescale(transformed_image, scale_factor, anti_aliasing=False, preserve_range=True)

    # Rotate
    transformed_image = ndimage.rotate(transformed_image, angle, reshape=False, prefilter=True)

    # Translate
    background = np.zeros((background_size, background_size))
    new_height, new_width = np.shape(transformed_image)

    y_start = background_offset + y_shift
    y_stop = y_start + new_height

    x_start = background_offset + x_shift
    x_stop = x_start + new_width

    background[y_start:y_stop, x_start:x_stop] = transformed_image

    # Get Normal Sized Window
    transformed_image = background[window_y_start:window_y_stop, window_x_start:window_x_stop]

    return transformed_image


def get_session_name(base_directory):
    # Create Output Directory
    split_base_directory = os.path.normpath(base_directory)
    split_base_directory = split_base_directory.split(os.sep)
    mouse_name = split_base_directory[-2]
    session_name = split_base_directory[-1]
    return mouse_name + "_" + session_name

def check_save_directory(base_directory, save_directory_root):

    # Create Output Directory
    split_base_directory = os.path.normpath(base_directory)
    split_base_directory = split_base_directory.split(os.sep)
    mouse_name = split_base_directory[-2]
    session_name = split_base_directory[-1]

    # Check Mouse Directory Exists
    mouse_directory = os.path.join(save_directory_root, mouse_name)
    if not os.path.exists(mouse_directory):
        os.mkdir(mouse_directory)

    # Check Session Save Directory Exists
    output_directory = os.path.join(mouse_directory, session_name)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    return output_directory


# Load Generous Mask
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



def transform_mask_or_atlas(image, variable_dictionary):

    image_height = 600
    image_width = 608

    # Unpack Dictionary
    angle = variable_dictionary['rotation']
    x_shift = variable_dictionary['x_shift']
    y_shift = variable_dictionary['y_shift']
    x_scale = variable_dictionary['x_scale']
    y_scale = variable_dictionary['y_scale']

    # Copy
    transformed_image = np.copy(image)

    # Scale
    original_height, original_width = np.shape(transformed_image)
    new_height = int(original_height * y_scale)
    new_width = int(original_width * x_scale)
    transformed_image = resize(transformed_image, (new_height, new_width), preserve_range=True)

    # Rotate
    transformed_image = ndimage.rotate(transformed_image, angle, reshape=False, prefilter=True)

    # Insert Into Background
    mask_height, mask_width = np.shape(transformed_image)
    centre_x = 200
    centre_y = 200
    background_array = np.zeros((1000, 1000))
    x_start = centre_x + x_shift
    x_stop = x_start + mask_width

    y_start = centre_y + y_shift
    y_stop = y_start + mask_height

    background_array[y_start:y_stop, x_start:x_stop] = transformed_image

    # Take Chunk
    transformed_image = background_array[centre_y:centre_y + image_height, centre_x:centre_x + image_width]

    # Rebinarize
    transformed_image = np.where(transformed_image > 0.5, 1, 0)

    return transformed_image


def get_activity_tensor(base_directory, onset_file, start_window, stop_window, save_directory_root, start_cutoff=3000):

    # Load Activity Matrix
    delta_f_file = os.path.join(base_directory, "Delta_F.h5")
    delta_f_container = tables.open_file(delta_f_file, "r")
    activity_matrix = delta_f_container.root.Data
    number_of_timepoints, number_of_pixels = np.shape(activity_matrix)

    # Open Onset File
    onset_list = np.load(os.path.join(base_directory, "Stimuli_Onsets", onset_file))
    number_of_trials = len(onset_list)

    # Create Activity Tensor
    activity_tensor = []
    for trial_index in range(number_of_trials):

        # Get Trial Activity
        trial_start = onset_list[trial_index] + start_window
        trial_stop = onset_list[trial_index] + stop_window
        print("Start", trial_start,  "stop", trial_stop)


        if trial_start > start_cutoff and trial_stop < number_of_timepoints:
            trial_activity = activity_matrix[trial_start:trial_stop]
            activity_tensor.append(trial_activity)

    activity_tensor = np.array(activity_tensor)

    # Save Activity Tensor
    output_directory = check_save_directory(base_directory, save_directory_root)
    save_file = os.path.join(output_directory, onset_file.replace("_onsets.npy", "") + "_" + "Activity_Tensor.npy")
    np.save(save_file, activity_tensor)

    # Close Delta F File
    delta_f_container.close()

    return activity_tensor


def load_tight_mask():
    tight_mask_dict = np.load("/home/matthew/Documents/Github_Code_Clean/Transition_Analysis/Brain_Registration/Tight_Mask_Dict.npy", allow_pickle=True)[()]
    indicies = tight_mask_dict["indicies"]
    image_height = tight_mask_dict["image_height"]
    image_width = tight_mask_dict["image_width"]
    return indicies, image_height, image_width


def load_tight_mask_downsized():

    mask = np.load("/home/matthew/Documents/Allen_Atlas_Templates/Mask_Array.npy")
    mask_alignment_dictionary = np.load(r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Tight_Mask_Alignment_Dictionary.npy", allow_pickle=True)[()]

    # Transform Mask
    mask = transform_mask_or_atlas(mask, mask_alignment_dictionary)
    mask = resize(mask, (100, 100), preserve_range=True, order=0, anti_aliasing=False)

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask>0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width



def get_ai_filename(base_directory):

    #Get List of all files
    file_list = os.listdir(base_directory)
    ai_filename = None

    #Get .h5 files
    h5_file_list = []
    for file in file_list:
        if file[-3:] == ".h5":
            h5_file_list.append(file)

    #File the H5 file which is two dates seperated by a dash
    for h5_file in h5_file_list:
        original_filename = h5_file

        #Remove Ending
        h5_file = h5_file[0:-3]

        #Split By Dashes
        h5_file = h5_file.split("-")

        if len(h5_file) == 2 and h5_file[0].isnumeric() and h5_file[1].isnumeric():
            return original_filename

def load_ai_recorder_file(base_directory):

    ai_filename = get_ai_filename(base_directory)
    ai_recorder_file_location = os.path.join(base_directory, ai_filename)

    table = tables.open_file(ai_recorder_file_location, mode='r')
    data = table.root.Data

    number_of_seconds = np.shape(data)[0]
    number_of_channels = np.shape(data)[1]
    sampling_rate = np.shape(data)[2]

    data_matrix = np.zeros((number_of_channels, number_of_seconds * sampling_rate))

    for second in range(number_of_seconds):
        data_window = data[second]
        start_point = second * sampling_rate

        for channel in range(number_of_channels):
            data_matrix[channel, start_point:start_point + sampling_rate] = data_window[channel]

    data_matrix = np.clip(data_matrix, a_min=0, a_max=None)
    return data_matrix


def load_analysis_container(analysis_name):

    # This Is The Location of My Analysis Logbook
    logbook_file_location = r"/home/matthew/Documents/Github_Code/Workflows/Analysis_Containers.ods"

    #  Read Logbook As A Dataframe
    logbook_dataframe = pd.read_excel(logbook_file_location, engine="odf")

    # Return First Analysis Container With This Name
    selected_analysis = logbook_dataframe.loc[(logbook_dataframe["Analysis_Name"] == analysis_name)].values[0]

    # Unpack Container
    start_window = selected_analysis[1]
    stop_window = selected_analysis[2]
    onset_files = selected_analysis[3].replace(' ', '')
    tensor_names = selected_analysis[4].replace(' ', '')
    behaviour_traces = selected_analysis[5] #.replace(' ', '')
    difference_conditions = selected_analysis[6].replace(' ', '')

    onset_files = onset_files.split(',')
    tensor_names = tensor_names.split(',')
    behaviour_traces = behaviour_traces.split(',')
    difference_conditions = difference_conditions.split(',')

    if difference_conditions[0] == 'None':
        difference_conditions = None

    else:
        for x in range(2):
            difference_conditions[x] = int(difference_conditions[x])

    # Return Container
    return [start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions]
