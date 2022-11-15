import numpy as np
import matplotlib.pyplot as plt


image_file = r"/media/matthew/External_Harddrive_1/Opto_Test/KPGC2.2G/2022_10_23_Opto_Test_No_Filter/Blue_Example_Image.npy"
image = np.load(image_file)
plt.imshow(image, cmap='Greys_r')
plt.show()