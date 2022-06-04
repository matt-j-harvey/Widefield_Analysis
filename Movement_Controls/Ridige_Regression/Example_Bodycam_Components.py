import numpy as np
import matplotlib.pyplot as plt

file_location = "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_08_Transition_Imaging/Functional Connectivity Analysis/Beta Series Correlations/Bodycam_Components.npy"
bodycam_components = np.load(file_location)

print(np.shape(bodycam_components))

number_of_components = np.shape(bodycam_components)[0]
for x in range(number_of_components):
    data = bodycam_components[x]
    data = np.reshape(data, (480, 640))
    data_magnitude = np.max(np.abs(data))
    plt.imshow(data, cmap='bwr', vmin=-1*data_magnitude, vmax=data_magnitude)
    plt.show()