import os

import tables
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn

import Learning_Utils

# Remove This Later
import warnings
warnings.filterwarnings("ignore")


def plot_swarmplot_multi_mice(dataframe):

    # Plot As Swarmplot
    axis = seaborn.swarmplot(y="Data_Value", hue="Mouse", x="Condition", data=dataframe, size=2, alpha=0.5)

    return axis


def mixed_effects_random_slope_and_intercept(dataframe):

    model = sm.MixedLM.from_formula("Data_Value ~ Condition", dataframe, re_formula="Condition", groups=dataframe["Mouse"])
    model_fit = model.fit()
    parameters = model_fit.params
    group_slope = parameters[1]
    p_value = model_fit.pvalues["Condition"]

    return p_value, group_slope


def repackage_data_into_dataframe(file_container, pixel_index, condition_names):

    # Combine_Into Dataframe
    dataframe = pd.DataFrame(dtype=np.float64)
    datapoints_list = []
    mouse_list = []
    condition_list = []

    mouse_index = 0
    for array in file_container.list_nodes(where="/" + condition_names[0]):
        array = np.array(array, dtype=np.float64)
        for trial in array:
            datapoints_list.append(trial[pixel_index])
            mouse_list.append(mouse_index)
            condition_list.append(0)
        mouse_index += 1

    mouse_index = 0
    for array in file_container.list_nodes(where="/" + condition_names[1]):
        array = np.array(array, dtype=np.float64)
        for trial in array:
            datapoints_list.append(trial[pixel_index])
            mouse_list.append(mouse_index)
            condition_list.append(1)
        mouse_index += 1

    #print("Datapoint list", datapoints_list)
    dataframe["Data_Value"] = datapoints_list
    dataframe["Condition"] = condition_list
    dataframe["Mouse"] = mouse_list

    return dataframe


def get_effect_map(data_directory, save_directory, condition_names):

    # Check Save Directory
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # load Mask
    indicies, image_height, image_width = Learning_Utils.load_tight_mask_downsized()
    number_of_pixels = len(indicies)

    file_list = os.listdir(data_directory)

    colourmap = Learning_Utils.get_mussall_cmap()

    count = 19
    data_file = file_list[count]
    #for data_file in file_list[count:]:
    file_container = tables.open_file(os.path.join(data_directory, data_file), "r")
    slope_list = np.zeros(number_of_pixels)
    for pixel_index in tqdm(range(number_of_pixels)):
        dataframe = repackage_data_into_dataframe(file_container, pixel_index, condition_names)

        try:
            p_value, slope = mixed_effects_random_slope_and_intercept(dataframe)
        except:
            p_value = 1
            slope = 0

        if p_value < 0.05:
            slope_list[pixel_index] = slope

    image = Learning_Utils.create_image_from_data(slope_list, indicies, image_height, image_width)
    image_magnitude = np.max(np.abs(image))
    plt.imshow(image, cmap=colourmap, vmax=image_magnitude, vmin=-1 * image_magnitude)
    plt.savefig(os.path.join(save_directory, str(count).zfill(4) + ".png"))
    plt.close()
    count += 1
    file_container.close()




# Control Learning
"""
filename = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Control_Combined_Tensor_Response_Baseline_Corrected"
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Mixed_Effects_Control_Response"
get_effect_map(filename, save_directory)

# Mutant Learning
filename = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Mutant_Combined_Tensor_Response_Baseline_Corrected"
save_directory = r"//media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Mixed_Effects_Mutants_Response"
#get_effect_map(filename, save_directory)
"""


# Genotype Pre Learning Vis 1
"""
filename = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Genotype_RT_Matched_Pre"
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Mixed_Effects_Genotype_Pre_Learning_Response"
condition_names = ["Controls", "Mutants"]
get_effect_map(filename, save_directory, condition_names)

# Genotype Post Learning Vis 1
filename = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Genotype_RT_Matched_Post"
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Mixed_Effects_Genotype_Post_Learning_Response"
condition_names = ["Controls", "Mutants"]
get_effect_map(filename, save_directory, condition_names)
"""
# Genotype Post Learning Vis 2
filename = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Genotype_Correct_Rejections_Post_Learning"
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Mixed_Effects_Genotype_Correct_Rejections_Response"
condition_names = ["Controls", "Mutants"]
get_effect_map(filename, save_directory, condition_names)

