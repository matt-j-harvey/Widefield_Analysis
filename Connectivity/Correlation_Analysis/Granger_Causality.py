def calculate_granger_causality(base_directory, onset_files, tensor_names, start_window, stop_window, tensor_save_directory, aligned=True):

    # Get File Structure
    split_base_directory = Path(base_directory).parts
    mouse_name = split_base_directory[-2]
    session_name = split_base_directory[-1]

    colourmap = widefield_utils.get_musall_cmap()

    # Get Data Structure
    number_of_conditions = len(onset_files)

    # Load Combined Mask
    if aligned == False:
        indicies, image_height, image_width = widefield_utils.load_downsampled_mask(base_directory)
    else:
        indicies, image_height, image_width = widefield_utils.load_tight_mask()

    # Downsample Further
    downsample_indicies, downsample_height, downsample_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

    correlation_matrix_list = []
    for condition_index in range(number_of_conditions):

        # Create Activity Tensor
        if aligned == False:
            Create_Activity_Tensor.create_activity_tensor(base_directory, onset_files[condition_index], start_window, stop_window, tensor_save_directory, start_cutoff=3000, align_within_mice=False, align_across_mice=False)
            activity_tensor = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, tensor_names[condition_index] + "_Activity_Tensor.npy"))
            correlation_matrix_filename = "_Signal_Correlation_Matrix.npy"

        if aligned == 'Within_Mouse':
            Create_Activity_Tensor.create_activity_tensor(base_directory, onset_files[condition_index], start_window, stop_window, tensor_save_directory, start_cutoff=3000, align_within_mice=True, align_across_mice=False)
            activity_tensor = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, tensor_names[condition_index] + "_Activity_Tensor_Aligned_Within_Mouse.npy"))
            correlation_matrix_filename = "_Signal_Correlation_Matrix_Aligned_Within_Mouse.npy"

        elif aligned == 'Across_Mice':
            Create_Activity_Tensor.create_activity_tensor(base_directory, onset_files[condition_index], start_window, stop_window, tensor_save_directory, start_cutoff=3000, align_within_mice=True, align_across_mice=True)
            activity_tensor = np.load(os.path.join(tensor_save_directory, mouse_name, session_name, tensor_names[condition_index] + "_Activity_Tensor_Aligned_Across_Mice.npy"))
            correlation_matrix_filename = "_Signal_Correlation_Matrix_Aligned_Across_Mice.npy"

        print("Activity Tensor Shape", np.shape(activity_tensor))

        # Concatenate and Subtract Mean
        activity_tensor = downsample_tensor(activity_tensor, indicies, image_height, image_width, downsample_indicies, downsample_height, downsample_width)
        print("Noise Tensor Shape", np.shape(activity_tensor))
        activity_tensor = np.mean(activity_tensor, axis=0)
        print("Noise Tensor Shape", np.shape(activity_tensor))


control_mouse_list = [

                ["/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_28_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_05_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_09_Switching_Imaging"],

                ["/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_29_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_05_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging"],

                ["/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_02_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_04_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_06_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_02_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_08_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_10_Transition_Imaging"],

                ["/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_26_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_28_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_02_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_23_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_31_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_04_02_Transition_Imaging"],

                ["/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_21_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_23_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_11_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_13_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_15_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_17_Transition_Imaging"],

                ["/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_14_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_20_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_22_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_29_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_03_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_05_Transition_Imaging"],
                ]