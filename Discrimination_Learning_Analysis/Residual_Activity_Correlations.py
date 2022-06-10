import numpy as np
import matplotlib.pyplot as plt
import os

session_list = [
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    #"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging",
]



number_of_sessions  = len(session_list)

figure_1 = plt.figure()
rows = 1
columns = number_of_sessions


for session_index in range(number_of_sessions):
    print("Session: ", session_index, " of ", number_of_sessions)

    session = session_list[session_index]
    residual_activity = np.load(os.path.join(session, "Movement_Correction", "Unexplained_Activity.npy"))
    residual_activity = np.transpose(residual_activity)
    print("Residual Activity Shape", np.shape(residual_activity))

    residual_correlation = np.corrcoef(residual_activity)

    axis = figure_1.add_subplot(rows, columns, session_index + 1)
    axis.imshow(residual_correlation, cmap='jet', vmin=0, vmax=1)
plt.show()