from scipy.io import savemat
import numpy as np
import glob

import os

base_directory = r"/media/matthew/29D46574463D2856/Nature_Transition_Analysis_Results/ROI_Trace_Raw_Values"

file_list = os.listdir(base_directory)
for numpy_file in file_list:
    if numpy_file[-4:] == ".npy":
        file_data = np.load(os.path.join(base_directory, numpy_file))
        data_dictionary = {"file_data":file_data}
        print(file_data)
        matlab_filename = numpy_file.replace(".npy", ".mat")
        savemat(file_name=os.path.join(base_directory, matlab_filename), mdict=data_dictionary)
