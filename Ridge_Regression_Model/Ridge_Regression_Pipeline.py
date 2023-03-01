from datetime import datetime
from tqdm import tqdm
import os
import numpy as np


#import Ridge_Regression_Manual_Input
#import Prepare_Data_For_Ridge_Regression
import Create_Design_Matrix
import Ridge_Regression_Model_Simple
import Visualise_Regression_Results


# Fit Model
session_list = [

        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_13_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_15_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_17_Switching_Imaging",

        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_31_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_02_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_04_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_08_Transition_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_10_Transition_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_12_Transition_Imaging",

        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_20_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_22_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_24_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_14_Transition_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_16_Transition_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_06_18_Transition_Imaging",

        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_17_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_19_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_23_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_30_Transition_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_06_Transition_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_08_Transition_Imaging",

        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_15_Switching_Imaging", - error
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_17_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_19_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_22_Transition_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_24_Transition_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_26_Transition_Imaging",

        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_14_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_20_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_26_Switching_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_05_Transition_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_08_Transition_Imaging",
        #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_11_10_Transition_Imaging",

        #r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_28_Switching_Imaging", - error

        ## Started Here ##

        r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_05_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_09_Switching_Imaging",

        r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_29_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_05_Switching_Imaging",

        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_02_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_04_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_06_Switching_Imaging",
        #r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_02_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_08_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_10_Transition_Imaging",

        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_26_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_28_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_02_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_23_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_31_Transition_Imaging",
        #r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_04_02_Transition_Imaging",

        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_21_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_23_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_11_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_13_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_15_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_17_Transition_Imaging",

        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_14_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_20_Switching_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_22_Switching_Imaging",

        ## Still Need To Do These ##
        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_29_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_03_Transition_Imaging",
        r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_05_Transition_Imaging",
    ]


# Settings
regression_type = "Ridge_Regression"


"""
# Ensure All Required Manual Input Has Been Performed
print("Loading Data For Manual Inspection")
Ridge_Regression_Manual_Input.ridge_regression_manual_input(session_list)


# Ensure Data Preprocessing Has Been Performed
print("Preparing Data For Ridge Regression")
for session in tqdm(session_list):
    Prepare_Data_For_Ridge_Regression.prepare_data_for_ridge_model(session)
"""

"""
# Create Design Matricies
print("Creating Design Matricies")
for base_directory in tqdm(session_list):
    print(base_directory)
    Create_Design_Matrix.create_design_matrix(base_directory)

"""

"""
# Fit Ridge Model
print("Fitting Ridge Models")
for session in tqdm(session_list):
    print("session", session, "at", datetime.now())
    Ridge_Regression_Model_Simple.fit_ridge_model(session, use_100=True)
    #Visualise_Regression_Results.visualise_regression_results(session)
"""



# Evaluate Model
print("Evaluating Models")
for session in tqdm(session_list):
    print("session", session, "at", datetime.now())
    Visualise_Regression_Results.visualise_regression_results(session, regression_type, use_100 = True)