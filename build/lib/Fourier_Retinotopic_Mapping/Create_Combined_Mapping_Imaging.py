import numpy as np
import os
import matplotlib.pyplot as plt

import Retinotopy_Utils



def load_sign_map(base_directory):

    # Load Sign Map
    map_directory = os.path.join(base_directory, "Stimuli_Evoked_Responses")
    sign_map = np.load(os.path.join(map_directory, "Thresholded_Sign_Map.npy"))

    return sign_map



def create_combined_mapping_image(left_directory, right_directory):

    # Load Maps
    left_sign_map = load_sign_map(left_directory)
    right_sign_map = load_sign_map(right_directory)

    # Load Alignment Dictionary
    left_within_mouse_alignment_dictionary = np.load(os.path.join(left_directory, "Within_Mouse_Alignment_Dictionary.npy"), allow_pickle=True)[()]
    right_within_mouse_alignment_dictionary = np.load(os.path.join(right_directory, "Within_Mouse_Alignment_Dictionary.npy"), allow_pickle=True)[()]

    # Transform Sign Maps
    left_sign_map = Retinotopy_Utils.transform_image(left_sign_map, left_within_mouse_alignment_dictionary)
    right_sign_map = Retinotopy_Utils.transform_image(right_sign_map, right_within_mouse_alignment_dictionary)

    # Take The Left Half One and The Right Half Of the Other
    combined_sign_map = np.zeros(np.shape(left_sign_map))
    combined_sign_map[:, 0:152] = right_sign_map[:, 0:152]
    combined_sign_map[:, 152:] = left_sign_map[:, 152:]

    # Load Example Images
    left_blue_example_image = np.load(os.path.join(left_directory, "Blue_Example_Image.npy"))
    right_blue_example_image = np.load(os.path.join(right_directory, "Blue_Example_Image.npy"))

    #plt.title("Left example image")
    #plt.imshow(left_blue_example_image)
    #plt.show()

    left_blue_example_image = Retinotopy_Utils.transform_image(left_blue_example_image, left_within_mouse_alignment_dictionary)
    right_blue_example_image = Retinotopy_Utils.transform_image(right_blue_example_image, right_within_mouse_alignment_dictionary)

    #plt.title("Left example image")
    #plt.imshow(left_blue_example_image)
    #plt.show()


    combined_greyscale_image =  np.zeros(np.shape(left_blue_example_image))
    combined_greyscale_image[:, 0:152] = left_blue_example_image[:, 0:152]
    combined_greyscale_image[:, 152:] = right_blue_example_image[:, 152:]


    plt.imshow(combined_greyscale_image, cmap="Greys_r")
    sign_map_magntidue = np.max(np.abs(combined_sign_map))
    plt.imshow(combined_sign_map, cmap='jet', alpha=0.8*np.abs(combined_sign_map), vmin=-sign_map_magntidue, vmax=sign_map_magntidue)
    plt.show()

    # Save Combined Image
    np.save(os.path.join(left_directory, "Combined_Sign_map.npy"), combined_sign_map)
    np.save(os.path.join(right_directory, "Combined_Sign_map.npy"), combined_sign_map)



session_list = [

    [r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_26_Continous_Retinotopy_Left",
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_07_27_Continous_Retinotopy_Right"],

    [r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_01_Continuous_Retinotopic_Mapping_Left",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_13_Continuous_Retinotopic_Mapping_Right"],

    ["/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_01_Continous_Retinotopy_Left",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_21_Continous_Retinotopy_Right"],

    ["/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_01_Continous_Retinotopy_Left",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_07_Continous_Retinotopy_Right"],

    ["/media/matthew/Expansion/Control_Data/NXAK14.1A/Continous_Retinotopic_Mapping_Left",
    "/media/matthew/Expansion/Control_Data/NXAK14.1A/Continous_Retinotopic_Mapping_Right"],

    ]


for session in session_list:
    create_combined_mapping_image(session[0], session[1])