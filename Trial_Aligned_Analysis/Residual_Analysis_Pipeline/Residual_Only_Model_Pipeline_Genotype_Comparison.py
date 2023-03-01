import os

number_of_threads = 2
os.environ["OMP_NUM_THREADS"] = str(number_of_threads) # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = str(number_of_threads) # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = str(number_of_threads) # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(number_of_threads) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(number_of_threads) # export NUMEXPR_NUM_THREADS=1

import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from Widefield_Utils import widefield_utils


import Create_Trial_Tensors
import Create_Analysis_Dataset_Genotype_Comparison
import Extract_Averages_Different_Genotypes
import Visualise_Average_Activity

from Files import Session_List



def run_residual_only_model_pipeline(selected_session_list, analysis_name, data_root_diretory_list, tensor_save_directory, start_cutoff=3000):

    # Select Analysis Details
    # For 2 Seconds Pre
    # To 1.5 Seconds Post
    # -56 to 40

    [start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = widefield_utils.load_analysis_container(analysis_name)
    print("Start Window", start_window)
    print("Stop Window", stop_window)

    print("Onset Files", onset_files)

    # Create Trial Tensors
    """
    genotype_index = 0
    for genotype in selected_session_list:
        for mouse in tqdm(genotype, leave=True, position=0, desc="Genotype"):
            for base_directory in tqdm(mouse, leave=True, position=1, desc="Trial Tensors Session"):
                for onsets_file in onset_files:

                    print("BAse Directory", base_directory)
                    print("Data Root Directory", data_root_diretory_list[genotype_index])
                    Create_Trial_Tensors.create_trial_tensor(os.path.join(data_root_diretory_list[genotype_index], base_directory), onsets_file, start_window, stop_window, tensor_save_directory,
                                        start_cutoff=start_cutoff,
                                        ridge_regression_correct=True,
                                        gaussian_filter=False,
                                        baseline_correct=False,
                                        align_within_mice=False,
                                        align_across_mice=False,
                                        extended_tensor=False,
                                        mean_only=False,
                                        stop_stimuli=None,
                                        use_100_df=True)

        genotype_index += 1
    """

    # Create Activity Dataset
    #Create_Analysis_Dataset_Genotype_Comparison.create_analysis_dataset(tensor_save_directory, selected_session_list, onset_files, analysis_name, start_window, stop_window)


    # Extract Averages
    #baseline_window = list(range(0, 3))
    baseline_window = list(range(0, 14))
    #baseline_window= list(range(69-14, 69))
    #Extract_Averages_Different_Genotypes.extract_condition_averages(tensor_save_directory, analysis_name, baseline_correct=True, baseline_window=baseline_window)

    # Visualise Averages
    condition_1_index = 1
    condition_2_index = 0
    comparison_name = "Genotype_Correct_Rejections"
    Visualise_Average_Activity.view_average_difference(tensor_save_directory, condition_1_index, condition_2_index, comparison_name, start_window, stop_window, vmin=-0.0, vmax=0.015, diff_magnitude=0.006)

    # Plot Regions


    # Test Region Significance


control_switching_only_sessions_nested = [

    [r"NRXN78.1A/2020_11_28_Switching_Imaging",
     r"NRXN78.1A/2020_12_05_Switching_Imaging",
     r"NRXN78.1A/2020_12_09_Switching_Imaging"],

    [r"NRXN78.1D/2020_11_29_Switching_Imaging",
     r"NRXN78.1D/2020_12_05_Switching_Imaging",
     r"NRXN78.1D/2020_12_07_Switching_Imaging"],

    [r"NXAK14.1A/2021_05_21_Switching_Imaging",
     r"NXAK14.1A/2021_05_23_Switching_Imaging",
     r"NXAK14.1A/2021_06_11_Switching_Imaging"],

    [r"NXAK22.1A/2021_10_14_Switching_Imaging",
     r"NXAK22.1A/2021_10_20_Switching_Imaging",
     r"NXAK22.1A/2021_10_22_Switching_Imaging"],

    [r"NXAK4.1B/2021_03_02_Switching_Imaging",
     r"NXAK4.1B/2021_03_04_Switching_Imaging",
     r"NXAK4.1B/2021_03_06_Switching_Imaging"],

    [r"NXAK7.1B/2021_02_26_Switching_Imaging",
     r"NXAK7.1B/2021_02_28_Switching_Imaging",
     #r"NXAK7.1B/2021_03_02_Switching_Imaging"
     ],
    
]



mutant_switching_only_sessions_nested = [

    [r"NRXN71.2A/2020_12_13_Switching_Imaging",
     r"NRXN71.2A/2020_12_15_Switching_Imaging",
     r"NRXN71.2A/2020_12_17_Switching_Imaging"],

    [r"NXAK4.1A/2021_03_31_Switching_Imaging",
     r"NXAK4.1A/2021_04_02_Switching_Imaging",
     r"NXAK4.1A/2021_04_04_Switching_Imaging"],

    [r"NXAK10.1A/2021_05_20_Switching_Imaging",
     r"NXAK10.1A/2021_05_22_Switching_Imaging",
     r"NXAK10.1A/2021_05_24_Switching_Imaging"],

    [r"NXAK16.1B/2021_06_17_Switching_Imaging",
     r"NXAK16.1B/2021_06_19_Switching_Imaging",
     r"NXAK16.1B/2021_06_23_Switching_Imaging"],

    [r"NXAK20.1B/2021_11_15_Switching_Imaging",
     r"NXAK20.1B/2021_11_17_Switching_Imaging",
     r"NXAK20.1B/2021_11_19_Switching_Imaging"],

    [r"NXAK24.1C/2021_10_14_Switching_Imaging",
     r"NXAK24.1C/2021_10_20_Switching_Imaging",
     r"NXAK24.1C/2021_10_26_Switching_Imaging"],

]












session_list = [

    # Controls
    [[r"NRXN78.1A/2020_11_24_Discrimination_Imaging"],
     [r"NRXN78.1D/2020_11_25_Discrimination_Imaging"],
     [r"NXAK4.1B/2021_02_22_Discrimination_Imaging"],
     [r"NXAK7.1B/2021_02_24_Discrimination_Imaging"],
     [r"NXAK14.1A/2021_05_09_Discrimination_Imaging"],
     [r"NXAK22.1A/2021_10_08_Discrimination_Imaging"]],

    # Mutants
    [[r"NRXN71.2A/2020_12_09_Discrimination_Imaging"],
     [r"NXAK4.1A/2021_03_05_Discrimination_Imaging"],
     [r"NXAK10.1A/2021_05_14_Discrimination_Imaging"],
     [r"NXAK16.1B/2021_06_15_Discrimination_Imaging"],
     [r"NXAK20.1B/2021_10_19_Discrimination_Imaging"],
     [r"NXAK24.1C/2021_10_08_Discrimination_Imaging"]]

    ]



Control_Post_learning = [

    ["NRXN78.1A/2020_11_17_Discrimination_Imaging",
     "NRXN78.1A/2020_11_19_Discrimination_Imaging",
     "NRXN78.1A/2020_11_21_Discrimination_Imaging"],

    ["NRXN78.1D/2020_11_21_Discrimination_Imaging",
     "NRXN78.1D/2020_11_23_Discrimination_Imaging",
     "NRXN78.1D/2020_11_25_Discrimination_Imaging"],

    ["NXAK4.1B/2021_02_14_Discrimination_Imaging",
     "NXAK4.1B/2021_02_22_Discrimination_Imaging"],

    ["NXAK7.1B/2021_02_24_Discrimination_Imaging"],

    ["NXAK14.1A/2021_05_05_Discrimination_Imaging",
     "NXAK14.1A/2021_05_07_Discrimination_Imaging",
     "NXAK14.1A/2021_05_09_Discrimination_Imaging"],

    ["NXAK22.1A/2021_10_07_Discrimination_Imaging",
     "NXAK22.1A/2021_10_08_Discrimination_Imaging"]

]


Mutant_Post_Learning = [

    ["NRXN71.2A/2020_12_05_Discrimination_Imaging"],

    ["NXAK4.1A/2021_03_03_Discrimination_Imaging",
     "NXAK4.1A/2021_03_05_Discrimination_Imaging"],

    ["NXAK10.1A/2021_05_12_Discrimination_Imaging",
     "NXAK10.1A/2021_05_14_Discrimination_Imaging"],

    ["NXAK16.1B/2021_06_15_Discrimination_Imaging"],

    ["NXAK20.1B/2021_10_11_Discrimination_Imaging",
     "NXAK20.1B/2021_10_13_Discrimination_Imaging",
     "NXAK20.1B/2021_10_15_Discrimination_Imaging",
     "NXAK20.1B/2021_10_17_Discrimination_Imaging",
     "NXAK20.1B/2021_10_19_Discrimination_Imaging"],

    ["NXAK24.1C/2021_10_02_Discrimination_Imaging",
     "NXAK24.1C/2021_10_04_Discrimination_Imaging",
     "NXAK24.1C/2021_10_06_Discrimination_Imaging"]

]


"""


analysis_name = "Hits_Vis_1_Aligned_Post"
tensor_save_directory = r"//media/matthew/External_Harddrive_2/Regression_Modelling_Results/Genotype_Comparisons_Hits_Post_Vis_Alinged"

analysis_name = "Hits_Vis_1_Aligned_Post"
tensor_directory = r"//media/matthew/External_Harddrive_2/Regression_Modelling_Results/Genotype_Comparisons_Hits_Post_Vis_Alinged"
"""




"""
selected_session_list = [control_switching_only_sessions_nested, mutant_switching_only_sessions_nested]
data_root_directory_list = [r"/media/matthew/Expansion/Control_Data", r"/media/matthew/External_Harddrive_1/Neurexin_Data"]
analysis_name = r"Visual_Context_Vis_2_Genotype"
tensor_directory = r"/media/matthew/External_Harddrive_2/Regression_Modelling_Results/Genotype_Vis_Context_Vis_2"

analysis_name = r"Odour_Context_Vis_2_Genotype"
tensor_directory = r"/media/matthew/External_Harddrive_2/Regression_Modelling_Results/Genotype_Odour_Context_Vis_2"
"""


selected_session_list = [Control_Post_learning, Mutant_Post_Learning]
data_root_directory_list = [r"/media/matthew/Expansion/Control_Data", r"/media/matthew/External_Harddrive_1/Neurexin_Data"]
analysis_name = r"Correct_Rejections"
tensor_directory = r"/media/matthew/External_Harddrive_2/Regression_Modelling_Results/Genotype_Comparison_CR_All_Post"


run_residual_only_model_pipeline(selected_session_list, analysis_name, data_root_directory_list, tensor_directory)