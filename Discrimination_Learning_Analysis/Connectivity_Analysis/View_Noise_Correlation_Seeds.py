import numpy as np
import matplotlib.pyplot as plt
import os





control_session_list = [

    # Controls 46 sessions

    # 78.1A - 6
    [
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_16_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_17_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_19_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_21_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1A/2020_11_24_Discrimination_Imaging",
    ],

    # 78.1D - 8
    [
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_14_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_15_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_16_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_17_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_19_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_21_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_23_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_11_25_Discrimination_Imaging",
    ],

    # 4.1B - 7
    [
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
    #r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging",
    ],

    # 22.1A - 7
    [
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_09_25_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_09_29_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_01_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_03_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_05_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_07_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging",
    ],

    # 14.1A - 6
    [
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_04_29_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_01_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_03_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_05_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_07_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_09_Discrimination_Imaging",
    ],

    # 7.1B - 12
    [
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_01_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_03_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_05_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_07_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_09_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_11_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_13_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_15_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_17_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_19_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_22_Discrimination_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_24_Discrimination_Imaging",
    ],
]
    # Mutants

mutant_session_list = [
    [
    # 4.1A - 15
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_08_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_10_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_12_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_14_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_16_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_18_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_23_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_25_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_02_27_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_03_01_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_03_03_Discrimination_Imaging",
    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging",
],
    # 20.1B - 11
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_09_28_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_09_30_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_09_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_11_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_13_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_15_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_17_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging",

    # 24.1C - 10
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_20_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_22_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_24_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_26_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_28_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_09_30_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_10_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_10_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_10_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK24.1C/2021_10_08_Discrimination_Imaging",

    # NXAK16.1B - 16
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_04_30_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_08_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_10_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_12_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_14_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_16_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_18_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_20_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_22_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_24_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_05_26_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_06_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK16.1B/2021_06_15_Discrimination_Imaging",

    # 10.1A - 8
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_04_30_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_02_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_04_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_06_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_08_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_10_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_12_Discrimination_Imaging",
    r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NXAK10.1A/2021_05_14_Discrimination_Imaging"


    # 71.2A - 16

]


for mouse in control_session_list:

    figure_1 = plt.figure()
    rows = 1
    columns = len(mouse)

    count = 1
    for base_directory in mouse:
        gc_map = np.load(os.path.join(base_directory, "Seed_Correlation_Map.npy"))
        axis = figure_1.add_subplot(rows, columns, count)
        plt.imshow(gc_map, cmap='jet', vmin=0, vmax=1)
        count += 1
    plt.show()


for mouse in mutant_session_list:

    figure_1 = plt.figure()
    rows = 1
    columns = len(mouse)

    count = 1
    for base_directory in mouse:
        gc_map = np.load(os.path.join(base_directory, "GC_Map.npy"))
        axis = figure_1.add_subplot(rows, columns, count)
        plt.imshow(gc_map, cmap='jet', vmin=0, vmax=0.1)
        count += 1
    plt.show()