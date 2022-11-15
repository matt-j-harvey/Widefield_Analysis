import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

base_directory = r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_10_29_Transition_Imaging"

svd_model = pickle.load(open(os.path.join(base_directory, "Mousecam_Analysis", "SVD Model.sav"), 'rb'))

components = svd_model.components_
for component in components:
    component = np.reshape(component, (480, 640))
    component_magnitude = np.abs(np.max(component))
    plt.imshow(component, cmap='bwr', vmin=-1 * component_magnitude, vmax = component_magnitude)
    plt.show()


