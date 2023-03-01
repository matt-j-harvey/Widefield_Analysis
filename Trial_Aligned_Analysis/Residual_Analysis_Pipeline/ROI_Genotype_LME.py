import tables
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pymer4.models import Lmer

import ROI_Quantification_Functions


def load_analysis_data(tensor_directory, analysis_name):

    # Open Analysis Dataframe
    analysis_file = tables.open_file(os.path.join(tensor_directory, analysis_name + "_Trialwise_.h5"), mode="r")
    activity_dataset = analysis_file.root["Data"]
    metadata_dataset = analysis_file.root["Trial_Details"]
    metadata_dataset = np.array(metadata_dataset)
    activity_dataset = np.array(activity_dataset)
    print("metadata_dataset", np.shape(metadata_dataset))
    print("activity_dataset", np.shape(activity_dataset))

    return activity_dataset, metadata_dataset




def repackage_data_into_dataframe(activity_list, mouse_list, genotype_list, session_list):

    # Combine_Into Dataframe
    dataframe = pd.DataFrame(dtype=np.float64)
    dataframe["Data_Value"] = activity_list
    dataframe["Condition"] = genotype_list
    dataframe["Mouse"] = mouse_list
    dataframe["Session"] = session_list

    return dataframe


def mixed_effects_two_levels_random_slope_intercept(dataframe):

    #model = Lmer('Data_Value ~ Condition + (1 + Condition|Mouse)', data=dataframe)
    model = Lmer('Data_Value ~ Condition + (1 + Condition|Mouse) + (1 + Condition|Session)', data=dataframe)

    results = model.fit(verbose=False)
    results = np.array(results)
    slope = results[1, 0]
    t_statistic = results[1, 5]
    p_value = results[1, 6]

    #print("t stat", t_statistic)
    print("P value", p_value)

    return slope, t_statistic, p_value



def roi_genotype_lme(tensor_directory, analysis_name, roi_list, roi_name):

    # Load Data
    activity_dataset, metadata_dataset = load_analysis_data(tensor_directory, analysis_name)
    print("activity_dataset Shape", np.shape(activity_dataset))

    # Get Save Directory
    save_directory = os.path.join(tensor_directory, "Mixed_Effects_Model_Results")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Get ROI Mean
    roi_pixels = ROI_Quantification_Functions.get_pooled_roi_from_list(roi_list)

    # Get Trial Details
    genotype_list = metadata_dataset[:, 0]
    mouse_list = metadata_dataset[:, 1]
    session_list = metadata_dataset[:, 2]

    activity_dataset = activity_dataset[:, :, roi_pixels]
    activity_dataset = np.mean(activity_dataset, axis=2)
    print("Actiivty Shape", np.shape(activity_dataset))

    number_of_trials, number_of_timepoints = np.shape(activity_dataset)

    p_value_list = []
    for timepoint_index in range(number_of_timepoints):

        # Get Timepoint Activity
        timepoint_activity_list = activity_dataset[:, timepoint_index]

        # COmbine Into Dataframe
        dataframe = repackage_data_into_dataframe(timepoint_activity_list, mouse_list, genotype_list, session_list)

        # Perform MEM
        slope, t_statistic, p_value = mixed_effects_two_levels_random_slope_intercept(dataframe)
        p_value_list.append(p_value)

    p_value_list = np.array(p_value_list)
    plt.plot(1- p_value_list)
    plt.axhline(0.95)
    plt.show()
    np.save(os.path.join(save_directory, roi_name + "_LME_P_Values.npy"), p_value_list)


tensor_directory = r"/media/matthew/External_Harddrive_2/Regression_Modelling_Results/Genotype_Comparison_CR"
tensor_directory = r"/media/matthew/External_Harddrive_2/Regression_Modelling_Results/Genotype_Comparison_CR_All_Post"
analysis_name = "Correct_Rejections"

tensor_directory = r"//media/matthew/External_Harddrive_2/Regression_Modelling_Results/Genotype_RT_Matched_Post_Learning"
analysis_name = "Genotype_Vis_1_Comparison_Post"

analysis_name = "Hits_Vis_1_Aligned_Post"
tensor_directory = r"//media/matthew/External_Harddrive_2/Regression_Modelling_Results/Genotype_Comparisons_Hits_Post_Vis_Alinged"



roi_list = [ "m2_left",  "m2_right"]
roi_name = "M2"
roi_genotype_lme(tensor_directory, analysis_name, roi_list, roi_name)


roi_list = [ "primary_visual_left",  "primary_visual_right"]
roi_name = "V1"
roi_genotype_lme(tensor_directory, analysis_name, roi_list, roi_name)
