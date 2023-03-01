import numpy as np
import matplotlib.pyplot as plt
import os

import Regression_Utils

def view_regression_coefs(base_directory):

    # Load Regression Dict
    regression_dictionary = np.load(os.path.join(base_directory, "Regression_Coefs",  "Regression_Dicionary_Bodycam.npy"), allow_pickle=True)[()]
    print("regression dictionary", regression_dictionary.keys())
    regression_coefs = regression_dictionary["Coefs"]
    regression_coefs = np.transpose(regression_coefs)
    regressor_names = regression_dictionary["Regressor_Names"]
    number_of_regressors = len(regressor_names)

    # Load Mask
    indicies, image_height, image_width = Regression_Utils.load_downsampled_mask(base_directory)

    # View Coefs
    difference_cmap = Regression_Utils.get_musall_cmap()

    for regressor_index in range(number_of_regressors):
        coef = regression_coefs[regressor_index]
        name = regressor_names[regressor_index]
        coef_map = Regression_Utils.create_image_from_data(coef,  indicies, image_height, image_width)
        coef_magnitude = np.max(np.abs(coef_map))
        plt.title(name)
        plt.imshow(coef_map, cmap=difference_cmap, vmax=coef_magnitude, vmin=-1*coef_magnitude)
        plt.show()



# Load Session List
# Fit Model
session_list = [
    #r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_23_Transition_Imaging",
    #r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_31_Transition_Imaging",
    #r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_04_02_Transition_Imaging",

    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_02_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_08_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_10_Transition_Imaging",

    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_13_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_15_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_17_Transition_Imaging",

    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_10_29_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_11_03_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_11_05_Transition_Imaging"

]

for session in session_list:
    view_regression_coefs(session)
