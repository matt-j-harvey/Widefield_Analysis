import Check_Photodiode_Trace
import Extract_Trial_Aligned_Activity_Continous_Retinotopy
import Create_Sweep_Aligned_Movie
import Continous_Retinotopy_Fourier_Analysis


session_list =  ["/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_01_Continous_Retinotopy_Left",
                 "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_21_Continous_Retinotopy_Right",
                 "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_07_Continous_Retinotopy_Right",
                 "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_01_Continous_Retinotopy_Left",
                 "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_26_Continous_Retinotopy_Left",
                 "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_27_Continous_Retinotopy_Right",
                 "/media/matthew/Expansion/Control_Data/NXAK14.1A/Continous_Retinotopic_Mapping_Left",
                 "/media/matthew/Expansion/Control_Data/NXAK14.1A/Continous_Retinotopic_Mapping_Right",
                 "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_01_Continuous_Retinotopic_Mapping_Left",
                 "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_13_Continuous_Retinotopic_Mapping_Right"]


for base_directory in session_list:

    # Get Stimuli Onsets
    #Check_Photodiode_Trace.check_photodiode_times(base_directory)

    # Extract Trial Aligned Activity
    #Extract_Trial_Aligned_Activity_Continous_Retinotopy.extract_trial_aligned_activity(base_directory)

    # Create Trial Averaged Movie
    #Create_Sweep_Aligned_Movie.create_activity_video(base_directory, "Horizontal_Sweep")
    #Create_Sweep_Aligned_Movie.create_activity_video(base_directory, "Vertical_Sweep")

    # Perform Fourrier Analysis
    Continous_Retinotopy_Fourier_Analysis.perform_fourier_analysis(base_directory)