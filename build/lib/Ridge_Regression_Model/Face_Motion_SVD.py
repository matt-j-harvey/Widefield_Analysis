import numpy as np
import os
import pickle
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD

def decompose_face_motion(base_directory):

    face_motion_data = np.load(os.path.join(base_directory, "Mousecam_Analysis", "Face_Motion_Energy.npy"))

    model = TruncatedSVD(n_components=20)
    transformed_data = model.fit_transform(face_motion_data)
    components = model.components_

    ## Save Model
    model_file = os.path.join(base_directory, "Mousecam_Analysis",  "Face_SVD_Model.sav")
    pickle.dump(model, open(model_file, 'wb'))

    np.save(os.path.join(base_directory, "Mousecam_Analysis", "Transformed_Mousecam_Face_Data.npy"), transformed_data)
    np.save(os.path.join(base_directory, "Mousecam_Analysis", "Mousecam_Face_Components.npy"), components)


def get_matched_face_motion(base_directory):

    bodycam_transformed_data = np.load(os.path.join(base_directory, "Mousecam_Analysis", "Transformed_Mousecam_Face_Data.npy"))
    print("Transformed Bodycam Data Shape", np.shape(bodycam_transformed_data))
    number_of_bodycam_frames = np.shape(bodycam_transformed_data)[0]

    # Load Widefied To Mousecam Frame Dict
    widefield_to_mousecam_frame_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy"), allow_pickle=True)[()]
    widefield_frame_keys = list(widefield_to_mousecam_frame_dict.keys())
    print("Widefield Frames", len(widefield_frame_keys))

    print("max bodycam Frame", np.max(list(widefield_to_mousecam_frame_dict.values())))
    matched_bodycam_data = []
    for widefield_index in widefield_frame_keys:
        mousecam_index = widefield_to_mousecam_frame_dict[widefield_index]
        if mousecam_index == number_of_bodycam_frames:
            matched_bodycam_data.append(bodycam_transformed_data[mousecam_index-1])
        else:
            matched_bodycam_data.append(bodycam_transformed_data[mousecam_index])

    matched_bodycam_data = np.array(matched_bodycam_data)
    np.save(os.path.join(base_directory, "Mousecam_Analysis", "Matched_Transformed_Mousecam_Face_Data.npy"), matched_bodycam_data)




session_list = [

        r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_23_Transition_Imaging",
        r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_31_Transition_Imaging",
        r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_04_02_Transition_Imaging",

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

for session in tqdm(session_list):
    decompose_face_motion(session)
    get_matched_face_motion(session)