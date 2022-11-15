import cv2
import tables
import os
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.measure import find_contours
from scipy import ndimage
import numpy as np


import numpy as np
import matplotlib.pyplot as plt
import tables
from scipy import ndimage
from skimage.feature import canny
from skimage.morphology import binary_closing
from skimage.segmentation import morphological_chan_vese
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
from skimage.transform import warp, resize

import math

import os
import sys


def transform_image(image, variable_dictionary):

    # Rotate
    angle = variable_dictionary['rotation']
    x_shift = variable_dictionary['x_shift']
    y_shift = variable_dictionary['y_shift']

    transformed_image = np.copy(image)
    transformed_image = ndimage.rotate(transformed_image, angle, reshape=False, prefilter=True)
    transformed_image = np.roll(a=transformed_image, axis=0, shift=y_shift)
    transformed_image = np.roll(a=transformed_image, axis=1, shift=x_shift)

    return transformed_image

def transform_image_list(image_list, session_list):

    transformed_image_list = []
    number_of_sessions = len(session_list)
    for session_index in range(number_of_sessions):

        session = session_list[session_index]

        # Load Alignment Dictionary
        alignment_dictionary = np.load(os.path.join(session, "Cluster_Alignment_Dictionary.npy"), allow_pickle=True)[()]

        image = transform_image(image_list[session_index], alignment_dictionary)

        plt.title("Transformed image")
        plt.imshow(image)
        plt.show()
        transformed_image_list.append(image)

    return transformed_image_list

def load_images(file_list):

    array_list = []
    for filename in file_list:
        datacontainer = tables.open_file(filename, mode="r")
        data = datacontainer.root["blue"][0]
        array_list.append(data)

    return array_list


def transform_with_optical_flow(image, v, u):
    nr, nc = image.shape
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
    image_warp = warp(image, np.array([row_coords + v, col_coords + u]), mode='edge')
    return image_warp


def resize_images(image_list):
    resized_images = []
    for image in image_list:
        resized_image = resize(image, (300, 304))

        plt.title("resized")
        plt.imshow(resized_image)
        plt.show()
        resized_images.append(resized_image)

    return resized_images


def extract_holes(image_list):

    processed_image_list = []
    for image in image_list:
        edges = canny(image, sigma=2)
        edges = np.multiply(edges, 255)
        edges = np.ndarray.astype(edges, float)
        #edges = ndimage.gaussian_filter(edges, sigma=1)
        processed_image_list.append(edges)
        #plt.title("Extracting holes")
        #plt.imshow(edges)
        #plt.show()
    return processed_image_list


def create_combined_image(height, width, template, dst):

    combined_image = np.zeros((height, width, 3))
    for y in range(height):
        for x in range(width):

            if template[y, x] > 0:
                combined_image[y, x, 0] = 1

            if dst[y, x] > 0:
                combined_image[y, x, 1] = 1

    return combined_image


full_file_list =   ["/media/matthew/External_Harddrive_1/Zoom_Calibration/1_Grid_Infinity/1/Grid_Infinity_2_20221021-165643_widefield.h5",
                    "/media/matthew/External_Harddrive_1/Zoom_Calibration/2_Grid_12_4/1/Grid_4_2_4_20221021-170208_widefield.h5",
                    "/media/matthew/External_Harddrive_1/Zoom_Calibration/3_Grid_7_2/1/Grid_7_2_20221021-170824_widefield.h5",
                    "/media/matthew/External_Harddrive_1/Zoom_Calibration/4_Grid_4_1_2/1/Grid_4_1_2_20221021-171202_widefield.h5"]


full_session_list = ["/media/matthew/External_Harddrive_1/Zoom_Calibration/1_Grid_Infinity/1",
                    "/media/matthew/External_Harddrive_1/Zoom_Calibration/2_Grid_12_4/1",
                    "/media/matthew/External_Harddrive_1/Zoom_Calibration/3_Grid_7_2/1",
                    "/media/matthew/External_Harddrive_1/Zoom_Calibration/4_Grid_4_1_2/1"]

second_session = 3
file_list = [full_file_list[0], full_file_list[second_session]]
session_list = [full_session_list[0], full_session_list[second_session]]

image_list = load_images(file_list)
image_list = transform_image_list(image_list, session_list)
image_list = resize_images(image_list)
edges_list = extract_holes(image_list)

# --- Compute the optical flow
v, u = optical_flow_ilk(image_list[0], image_list[1], radius=15)

image_1 = edges_list[0]
image_2 = edges_list[1]
h, w = np.shape(image_1)
transformation_increments = np.linspace(start=0.1, stop=1, num=100)

plt.ion()
for increment in transformation_increments:
    u_increment = np.multiply(u, increment)
    v_increment = np.multiply(v, increment)
    new_image_2 = transform_with_optical_flow(image_2, v_increment, u_increment)
    combined_image = create_combined_image(h, w, image_1, new_image_2)
    plt.title(increment)
    plt.imshow(combined_image)
    plt.draw()
    plt.pause(0.1)
    plt.clf()


np.save(os.path.join("/home/matthew/Documents/Github_Code_Clean/Preprocessing/Brain_Alignment", "zoom_optic_flow_u.npy"), u)
np.save(os.path.join("/home/matthew/Documents/Github_Code_Clean/Preprocessing/Brain_Alignment", "zoom_optic_flow_v.npy"), v)


