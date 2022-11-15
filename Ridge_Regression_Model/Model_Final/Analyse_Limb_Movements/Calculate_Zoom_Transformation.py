import cv2
import tables
import os
import matplotlib.pyplot as plt
from skimage.feature import canny
from scipy import ndimage
import numpy as np


import numpy as np
import matplotlib.pyplot as plt
import tables
from scipy import ndimage
from skimage.feature import canny
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
        transformed_image_list.append(image)

    return transformed_image_list

def load_images(file_list):

    array_list = []
    for filename in file_list:
        datacontainer = tables.open_file(filename, mode="r")
        data = datacontainer.root["blue"][0]
        array_list.append(data)

    return array_list


def extract_holes(image_list):

    processed_image_list = []
    for image in image_list:
        edges = canny(image, sigma=2)
        processed_image_list.append(edges)
        #plt.imshow(edges)
        #plt.show()
    return processed_image_list




full_file_list =   ["/media/matthew/External_Harddrive_1/Zoom_Calibration/1_Grid_Infinity/1/Grid_Infinity_2_20221021-165643_widefield.h5",
                    "/media/matthew/External_Harddrive_1/Zoom_Calibration/2_Grid_12_4/1/Grid_4_2_4_20221021-170208_widefield.h5",
                    "/media/matthew/External_Harddrive_1/Zoom_Calibration/3_Grid_7_2/1/Grid_7_2_20221021-170824_widefield.h5",
                    "/media/matthew/External_Harddrive_1/Zoom_Calibration/4_Grid_4_1_2/1/Grid_4_1_2_20221021-171202_widefield.h5"]


full_session_list = ["/media/matthew/External_Harddrive_1/Zoom_Calibration/1_Grid_Infinity/1",
                    "/media/matthew/External_Harddrive_1/Zoom_Calibration/2_Grid_12_4/1",
                    "/media/matthew/External_Harddrive_1/Zoom_Calibration/3_Grid_7_2/1",
                    "/media/matthew/External_Harddrive_1/Zoom_Calibration/4_Grid_4_1_2/1"]

second_session = 1
file_list = [full_file_list[0], full_file_list[second_session]]
session_list = [full_session_list[0], full_session_list[second_session]]



image_list = load_images(file_list)
image_list = extract_holes(image_list)
image_list = transform_image_list(image_list, session_list)


plt.imshow(image_list[0], cmap="Reds", alpha=0.5)
plt.imshow(image_list[1], cmap="Greens", alpha=0.5)
plt.show()

template = image_list[0]
dst = image_list[1]

template = np.ndarray.astype(template, np.float32)
dst = np.ndarray.astype(dst, np.float32)

warp_mode = cv2.MOTION_HOMOGRAPHY
M = np.eye(3, 3, dtype=np.float32)
niter = 100
eps0 = 1e-3
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,niter, eps0)
h, w = template.shape
hann = cv2.createHanningWindow((w, h), cv2.CV_32FC1)
hann = (hann * 255).astype('uint8')
gaussian_filter = 1
(res, M) = cv2.findTransformECC(template, dst, M, warp_mode, criteria, inputMask=hann, gaussFiltSize=gaussian_filter)
dst = cv2.warpPerspective(dst, M, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

print("M")
print(M)


plt.imshow(template, cmap="Reds", alpha=0.5)
plt.imshow(dst, cmap="Greens", alpha=0.5)
plt.show()


