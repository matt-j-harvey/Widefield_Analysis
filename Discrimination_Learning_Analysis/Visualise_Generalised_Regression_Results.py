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


def visualise_regression_results(session, model_name):

    # Load Dictionary
    regression_dictionary = np.load(os.path.join(session, "Simple_Regression", model_name + "_Regression_Model.npy"), allow_pickle=True)[()]
    print(regression_dictionary.keys())

    # Extract Items Of Interest
    r2 = regression_dictionary["R2"]
    error = regression_dictionary['Full_Sum_Sqaure_Error']
    regression_coefficients = regression_dictionary["Regression_Coefficients"]
    partial_determination_coefficients = regression_dictionary["Coefficients_of_Partial_Determination"]

    number_of_conditions = len(regression_coefficients)
    for condition_index in range(number_of_conditions):

        condition_regression_coefs = regression_coefficients[condition_index]
        condition_cpds = partial_determination_coefficients[condition_index]

        print("Coefs", np.shape(condition_regression_coefs))
        print("CPDs", np.shape(condition_cpds))

        visualise_coefficients(session, regression_coefficients[condition_index])
        visualise_coefficients(session, partial_determination_coefficients[condition_index])


session_list = [
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging"

    r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_08_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_10_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_12_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_14_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_16_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_18_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_23_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_25_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_27_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_03_01_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_03_03_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_03_05_Discrimination_Imaging",

    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_09_28_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_09_30_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_09_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_11_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_13_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_15_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_17_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging",

    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_16_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_17_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_19_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_21_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive1/Processed_Widefield_Data/NRXN78.1A/2020_11_24_Discrimination_Imaging",
]


session_list = [

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
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",

    # NXAK16.1B - 16
    "/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_04_30_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_02_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_04_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_06_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_08_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_10_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_12_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_14_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_16_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_18_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_20_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_22_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_24_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_26_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_06_04_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_06_15_Discrimination_Imaging",
    ]

model_name = "Lick_Onsets"

for session in session_list:
    visualise_regression_results(session, model_name)