import numpy as np
import sklearn.svm
from sklearn.decomposition import NMF
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
import os
import matplotlib.pyplot as plt
import sys
from matplotlib import cm
import h5py

from scipy import ndimage

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")
import Widefield_General_Functions




def load_generous_mask(home_directory):

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


def visualise_coefficients(base_directory, coefficients):
    coefficients = np.nan_to_num(coefficients)

    indicies, image_height, image_width = load_generous_mask(base_directory)

    number_of_dimensions = np.ndim(coefficients)
    max_coefficient = np.percentile(coefficients, q=99.5)
    min_coefficient = np.min(coefficients)
    print("Max Coefficinets", max_coefficient)
    print("MIN Coeffi", min_coefficient)

    if number_of_dimensions == 1:
        image = Widefield_General_Functions.create_image_from_data(coefficients, indicies, image_height, image_width)
        plt.imshow(image)
        plt.show()


    elif number_of_dimensions == 2:

        dim_1, dim_2 = np.shape(coefficients)

        if dim_1 > dim_2:
            coefficients = np.transpose(coefficients)

        nuber_of_samples = np.shape(coefficients)[0]
        plt.ion()

        for x in range(nuber_of_samples):
            image = Widefield_General_Functions.create_image_from_data(coefficients[x], indicies, image_height, image_width)
            image = ndimage.gaussian_filter(image, sigma=1)
            plt.title(str(x))
            plt.imshow(image, cmap='viridis')
            #vmin=0, vmax=max_coefficient
            plt.draw()
            plt.pause(0.1)
            plt.clf()

        plt.ioff()
    plt.close()



session_list = [
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging"
]




for session in session_list:

    # Load Dictionary
    regression_dictionary = np.load(os.path.join(session, "Simple_Regression", "Simple_Regression_Model.npy"), allow_pickle=True)[()]
    print(regression_dictionary.keys())

    # Extract Items Of Interest
    regression_coefficients = regression_dictionary["Regression_Coefficients"]

    r2 = regression_dictionary["R2"]
    partial_determination_matrix = regression_dictionary["Coefficients_of_Partial_Determination"]
    condition_1 = regression_dictionary["Condition_1"]
    condition_2 = regression_dictionary["Condition_2"]
    start_window = regression_dictionary["Start_Window"]
    stop_window = regression_dictionary["Stop_Window"]
    error = regression_dictionary['Full_Sum_Sqaure_Error']
    #visualise_coefficients(session, regression_coefficients)
    #visualise_coefficients(session, partial_determination_matrix)

    visualise_coefficients(session, error)