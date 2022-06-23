import numpy as np
import matplotlib.pyplot as plt
import os

base_directory = r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging"
clustered_delta_f = np.load(os.path.join(base_directory, "Cluster_Activity_Matrix.npy"))
print("Clustered Delta F shape", np.shape(clustered_delta_f))

plt.imshow(np.transpose(clustered_delta_f[0:500]))
plt.show()