import numpy as np
import matplotlib.pyplot as plt
import os

base_directory = r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Mutant_Data/NXAK4.1A/2021_02_02_Discrimination_Imaging"
example_image = np.load(os.path.join(base_directory, "Blue_Example_Image.npy"))