from pyEDM import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime



def get_CCM_matrix(base_directory):

    # Load Cluster Activity
    cluster_activity_matrix = np.load(os.path.join(base_directory, "Cluster_Activity_Matrix.npy"))
    cluster_activity_matrix = cluster_activity_matrix[3000:3500, 1:]
    print("Cluster Activity MAtrix Shape", np.shape(cluster_activity_matrix))

    number_of_regions = np.shape(cluster_activity_matrix)[1]
    connectivity_matrix = np.zeros((number_of_regions, number_of_regions))

    for region_1_index in range(number_of_regions):
        for region_2_index in range(region_1_index, number_of_regions):
            print("Region: ", region_1_index, " To ", region_2_index)

            region_1_trace = cluster_activity_matrix[:, region_1_index]
            region_2_trace = cluster_activity_matrix[:, region_2_index]

            dataframe_dict = {"region_0":region_1_trace, "region_1":region_1_trace, "region_2":region_2_trace}
            dataframe = pd.DataFrame(dataframe_dict)

            result_dataframe = CCM( dataFrame = dataframe, E = 3, columns = "region_1", target = "region_2", libSizes = "10 75 5", sample = 100, showPlot = False );
            region_1_to_2 = np.mean(result_dataframe["region_1:region_2"])
            region_2_to_1 = np.mean(result_dataframe["region_2:region_1"])

            connectivity_matrix[region_1_index, region_2_index] = region_1_to_2
            connectivity_matrix[region_2_index, region_1_index] = region_2_to_1

    plt.imshow(connectivity_matrix)
    plt.show()
base_directory =     r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_03_23_Transition_Imaging"
get_CCM_matrix(base_directory)
