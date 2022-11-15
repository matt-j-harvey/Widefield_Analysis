import numpy as np
import matplotlib.pyplot as plt
import os

import Regression_Utils


def reconstruct_face_coef(coef, base_directory):

    face_pixels = np.load(os.path.join(base_directory, "Mousecam_analysis", "Whisker_Pixels.npy"))
    print("Face Pixels", np.shape(face_pixels))

    # Get Face Extent
    face_y_min = np.min(face_pixels[0])
    face_y_max = np.max(face_pixels[0])
    face_x_min = np.min(face_pixels[1])
    face_x_max = np.max(face_pixels[1])
    print("y min", face_y_min, "y max", face_y_max, "x min", face_x_min, "x max", face_x_max)

    face_pixels = np.transpose(face_pixels)
    print("Face Pixels", np.shape(face_pixels))


    template = np.zeros((480, 640))
    count = 0
    for pixel in face_pixels:
        template[pixel[0], pixel[1]] = coef[count]
        count += 1

    template = template[face_y_min:face_y_max, face_x_min:face_x_max]
    return template


"""
def get_sample_face_frame(base_directory):

    # Get Video File Name
    video_name =

    # Open Face Pixels

    # Open Video File

    # Read Frame 1
"""


def view_face_pixel_loadings(face_coefs, base_directory):

    # Load Face Compoents
    face_components = np.load(os.path.join(base_directory, "Mousecam_Analysis", "Mousecam_Face_Components.npy"))
    print("Face Compoentns", np.shape(face_components))

    face_coefs = np.dot(np.transpose(face_components), face_coefs)
    print("Face coefs", np.shape(face_coefs))

    face_map = np.sum(np.abs(face_coefs), axis=1)
    magnitude = np.percentile(face_map, 99)
    print(np.shape(face_map))

    face_map = reconstruct_face_coef(face_map, base_directory)
    alpha_map = np.divide(face_map, magnitude)
    alpha_map = np.clip(alpha_map, a_min=0, a_max=1)

    # Load Example Frame


    plt.imshow(face_map, cmap='hot', vmax=magnitude, vmin=0, alpha=alpha_map)
    plt.show()



def view_coefficients(number_of_regressors, regression_coefs, regressor_names, indicies, image_height, image_width):

    # View Coefs
    difference_cmap = Regression_Utils.get_musall_cmap()
    for regressor_index in range(number_of_regressors):
        coef = regression_coefs[regressor_index]
        name = regressor_names[regressor_index]
        coef_map = Regression_Utils.create_image_from_data(coef, indicies, image_height, image_width)
        coef_magnitude = np.max(np.abs(coef_map))
        plt.title(name)
        plt.imshow(coef_map, cmap=difference_cmap, vmax=coef_magnitude, vmin=-1 * coef_magnitude)
        plt.show()

def view_coefficients_of_partial_determination(number_of_regressors, coef_list, regressor_names, indicies, image_height, image_width):

    # View Coefs
    print("coef list", np.shape(coef_list))
    for regressor_index in range(number_of_regressors):
        coef = coef_list[regressor_index]
        name = regressor_names[regressor_index]
        coef_map = Regression_Utils.create_image_from_data(coef, indicies, image_height, image_width)
        plt.title(name)
        plt.imshow(coef_map)
        plt.show()


def view_regression_model(base_directory):

    # Load Regression Dict
    regression_dictionary = np.load(os.path.join(base_directory, "Regression_Coefs",  "Regression_Dicionary_Bodycam.npy"), allow_pickle=True)[()]
    print("regression dictionary", regression_dictionary.keys())
    regression_coefs = regression_dictionary["Coefs"]
    coefficients_of_partial_determination = regression_dictionary["Coefficients_of_Partial_Determination"]
    print("CPDS", np.shape(coefficients_of_partial_determination))
    regression_coefs = np.transpose(regression_coefs)
    coefficients_of_partial_determination = np.transpose(coefficients_of_partial_determination)
    regressor_names = regression_dictionary["Regressor_Names"]
    number_of_regressors = len(regressor_names)


    # Load Mask
    indicies, image_height, image_width = Regression_Utils.load_downsampled_mask(base_directory)

    # View Face Pixel Loadings
    #face_coefs = regression_coefs[2:]
    #view_face_pixel_loadings(face_coefs, base_directory)

    view_coefficients_of_partial_determination(number_of_regressors, coefficients_of_partial_determination, regressor_names, indicies, image_height, image_width)

    # View Coefs
    view_coefficients(number_of_regressors, regression_coefs, regressor_names, indicies, image_height, image_width)



# Load Session List
# Fit Model
session_list = [
    r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_23_Transition_Imaging",
    #r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_31_Transition_Imaging",
    #r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_04_02_Transition_Imaging",

    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_02_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_08_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_10_Transition_Imaging",

    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_13_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_15_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_17_Transition_Imaging",

    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_10_29_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_11_03_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_11_05_Transition_Imaging"

]

for session in session_list:
    view_regression_model(session)
