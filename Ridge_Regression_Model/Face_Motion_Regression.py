import matplotlib.pyplot as plt
import numpy as np
import os
import tables
from bisect import bisect_left
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

def create_image_from_data(data, indicies, image_height, image_width):
    template = np.zeros((image_height, image_width))
    data = np.nan_to_num(data)
    np.put(template, indicies, data)
    image = np.ndarray.reshape(template, (image_height, image_width))

    return image


def load_mask(home_directory):

    # Loads the mask for a video, returns a list of which pixels are included, as well as the original image height and width
    mask = np.load(home_directory + "/Generous_Mask.npy")

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask>0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width

def invert_dictionary(dictionary):
    inv_map = {v: k for k, v in dictionary.items()}
    return inv_map


def take_closest(myList, myNumber):

    """
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.
    """

    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before

def match_mousecam_to_widefield_frames(base_directory):

    # Load Frame Times
    widefield_frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    mousecam_frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Mousecam_Frame_Times.npy"), allow_pickle=True)[()]

    widefield_frame_times = invert_dictionary(widefield_frame_times)
    widefield_frame_time_keys = list(widefield_frame_times.keys())
    mousecam_frame_times_keys = list(mousecam_frame_times.keys())
    mousecam_frame_times_keys.sort()

    # Get Number of Frames
    number_of_widefield_frames = len(widefield_frame_time_keys)

    # Dictionary - Keys are Widefield Frame Indexes, Values are Closest Mousecam Frame Indexes
    widfield_to_mousecam_frame_dict = {}

    for widefield_frame in range(number_of_widefield_frames):
        frame_time = widefield_frame_times[widefield_frame]
        closest_mousecam_time = take_closest(mousecam_frame_times_keys, frame_time)
        closest_mousecam_frame = mousecam_frame_times[closest_mousecam_time]
        widfield_to_mousecam_frame_dict[widefield_frame] = closest_mousecam_frame
        #print("Widefield Frame: ", widefield_frame, " Closest Mousecam Frame: ", closest_mousecam_frame)

    # Save Directory
    save_directoy = os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy")
    np.save(save_directoy, widfield_to_mousecam_frame_dict)


def view_prediction(real_data, predicited_data):

    figure_1 = plt.figure()

def regress_face_motion(base_directory):

    # Load Face Motion Activity
    face_motion = np.load(os.path.join(base_directory, "Mousecam_analysis", "Face_Motion_Data.npy"))
    print("Face motion", np.shape(face_motion))

    # Load Delta F
    delta_f_file = os.path.join(base_directory, "Delta_F.h5")
    delta_f_container = tables.open_file(delta_f_file, "r")
    delta_f_matrix = delta_f_container.root.Data
    print("Delta F Matrix SHape", np.shape(delta_f_matrix))

    # Load Widefield To Mousecam Frame Dict
    widefield_to_mousecam_frame_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy"), allow_pickle=True)[()]

    # Select Widefield Frames
    sample_size = 17000
    train_start = 20000
    train_stop = train_start + sample_size
    test_start = train_stop
    test_stop = test_start + sample_size

    train_mousecam_frames = []
    for widefield_index in range(train_start, train_stop):
        mousecam_index = widefield_to_mousecam_frame_dict[widefield_index]
        train_mousecam_frames.append(mousecam_index)

    test_mousecam_frames = []
    for widefield_index in range(test_start, test_stop):
        mousecam_index = widefield_to_mousecam_frame_dict[widefield_index]
        test_mousecam_frames.append(mousecam_index)

    train_delta_f = delta_f_matrix[train_start:train_stop]
    test_delta_f = delta_f_matrix[test_start:test_stop]
    
    train_mousecam = face_motion[train_mousecam_frames]
    test_mousecam = face_motion[test_mousecam_frames]

    # Create Model
    print("Performing Regression")
    model = Ridge()
    model.fit(X=train_mousecam, y=train_delta_f)

    # Load Mask
    indicies, image_height, image_width = load_mask(base_directory)

    coefs = model.coef_
    coefs = np.transpose(coefs)
    coefs = np.mean(coefs, axis=0)
    coef_image = create_image_from_data(coefs, indicies, image_height, image_width)
    plt.imshow(coef_image)
    plt.show()

    # Predicted Delta F
    predicted_data = model.predict(test_mousecam)

    model_score = r2_score(y_true=test_delta_f, y_pred=predicted_data, multioutput='raw_values')
    print("Mean Score", np.mean(model_score))

    r2_map = create_image_from_data(model_score, indicies, image_height, image_width)
    plt.imshow(r2_map)
    plt.show()

    # View Predicited Data
    vmin = np.percentile(predicited_data, q=1)
    vmax = np.percentile(predicited_data, q=99)

    plt.ion()
    for frame in predicited_data:
        frame_image = create_image_from_data(frame, indicies, image_height, image_width)
        plt.clf()
        plt.imshow(frame_image, vmin=vmin, vmax=vmax, cmap='inferno')
        plt.draw()
        plt.pause(0.1)


    # Close Delta F File When Finished
    delta_f_container.close()


base_directory = r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging"
#match_mousecam_to_widefield_frames(base_directory)
regress_face_motion(base_directory)

