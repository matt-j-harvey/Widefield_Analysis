import os
import numpy as np
import matplotlib.pyplot as plt

import Regression_Utils

session_list = [
    r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_23_Transition_Imaging",
    r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_31_Transition_Imaging",
    r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_04_02_Transition_Imaging",
]

colourmap = Regression_Utils.get_musall_cmap()
for session in session_list:
    heamo_coefs = np.load(os.path.join(session, "Heamodynamic_Regression_Coefs2.npy"))
    mask_dict = np.load(os.path.join(session, "Downsampled_mask_dict.npy"), allow_pickle=True)[()]

    indicies = mask_dict["indicies"]
    image_height = mask_dict["image_height"]
    image_width = mask_dict["image_width"]

    image = Regression_Utils.create_image_from_data(heamo_coefs, indicies, image_height, image_width)
    image_magntidue = np.percentile(np.abs(image), 99)
    plt.axis('off')
    plt.imshow(image, cmap=colourmap, vmin=-image_magntidue, vmax=image_magntidue)
    plt.colorbar()
    plt.show()


