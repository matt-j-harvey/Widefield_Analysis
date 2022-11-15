import os

import tables
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn
from matplotlib import cm
from matplotlib.pyplot import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import Learning_Utils

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


def repackage_data_into_dataframe(file_container, pixel_index):

    # Combine_Into Dataframe
    dataframe = pd.DataFrame(dtype=np.float64)
    datapoints_list = []
    mouse_list = []
    condition_list = []

    mouse_index = 0
    for array in file_container.list_nodes(where="/Pre_Learning"):
        array = np.array(array, dtype=np.float64)
        for trial in array:
            datapoints_list.append(trial[pixel_index])
            mouse_list.append(mouse_index)
            condition_list.append(0)
        mouse_index += 1

    mouse_index = 0
    for array in file_container.list_nodes(where="/Post_Learning"):
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


def repackage_into_numpy_arrays(file_container, condition_names):

    pre_learning_activity_list = []
    post_learning_activity_list = []

    for array in file_container.list_nodes(where="/" + condition_names[0]):
        array = np.array(array)
        for trial in array:
            pre_learning_activity_list.append(trial)

    for array in file_container.list_nodes(where="/" + condition_names[1]):
        array = np.array(array)
        for trial in array:
            post_learning_activity_list.append(trial)

    pre_learning_activity_list = np.array(pre_learning_activity_list)
    post_learning_activity_list = np.array(post_learning_activity_list)

    return pre_learning_activity_list, post_learning_activity_list



def get_effect_map(data_directory, save_directory, downsized=False):

    # Create Save Directory
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Get Baseline


    # load Mask
    if downsized == True:
        indicies, image_height, image_width = Learning_Utils.load_tight_mask_downsized()
    else:
        indicies, image_height, image_width = Learning_Utils.load_tight_mask()


    file_list = os.listdir(data_directory)
    colourmap = Learning_Utils.get_mussall_cmap()

    count = 0
    for data_file in file_list[count:]:

        # Extract Data
        file_container = tables.open_file(os.path.join(data_directory, data_file), "r")
        pre_learning_array, post_learning_array = repackage_into_numpy_arrays(file_container)
        file_container.close()

        #print("Pre learning shape", np.shape(pre_learning_array))
        #print("Post learning shape", np.shape(post_learning_array))

        t_stats, p_values = stats.ttest_ind(post_learning_array, pre_learning_array, axis=0)
        #print("t stats shape", np.shape(t_stats))

        t_stats = np.where(p_values < 0.05, t_stats, 0)

        image = Learning_Utils.create_image_from_data(t_stats, indicies, image_height, image_width)
        image_magnitude = np.max(np.abs(image))
        plt.imshow(image, cmap=colourmap, vmax=image_magnitude, vmin=-1 * image_magnitude)
        plt.savefig(os.path.join(save_directory, str(count).zfill(4) + ".png"))
        plt.close()
        count += 1


def create_effect_figure(data_directory, save_directory, condition_names, start_window, downsized=False):


    # Create Save Directory
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # load Mask
    if downsized == True:
        indicies, image_height, image_width = Learning_Utils.load_tight_mask_downsized()
    else:
        indicies, image_height, image_width = Learning_Utils.load_tight_mask()

    # Get Background Pixels
    background_pixels = Learning_Utils.get_background_pixels(indicies, image_height, image_width)

    # Load Colourmaps
    positive_magnitude = 10000
    difference_magntidue = 3000
    positive_colourmap = ScalarMappable(cmap=cm.get_cmap("inferno"), norm=Normalize(vmin=0, vmax=positive_magnitude))
    difference_colourmap = ScalarMappable(cmap=Learning_Utils.get_mussall_cmap(), norm=Normalize(vmin=-difference_magntidue, vmax=difference_magntidue))

    # Load File List
    file_list = os.listdir(data_directory)

    # Create Figure
    figure_1 = plt.figure(figsize=(15, 4))
    gridspec_1 = GridSpec(nrows=1, ncols=3, figure=figure_1)

    # Iterate Though Timepoints
    count = 0
    time_values = list(range(len(file_list)))
    time_values = np.add(time_values, start_window)
    time_values = np.multiply(time_values, 36)

    for data_file in file_list[count:]:

        condition_1_axis = figure_1.add_subplot(gridspec_1[0, 0])
        condition_2_axis = figure_1.add_subplot(gridspec_1[0, 1])
        difference_axis = figure_1.add_subplot(gridspec_1[0, 2])

        # Extract Data
        file_container = tables.open_file(os.path.join(data_directory, data_file), "r")
        pre_learning_array, post_learning_array = repackage_into_numpy_arrays(file_container, condition_names)
        file_container.close()

        # Get Mean Pre and Post Matricies
        pre_learning_mean = np.mean(pre_learning_array, axis=0)
        post_learning_mean = np.mean(post_learning_array, axis=0)

        difference = np.subtract(pre_learning_mean, post_learning_mean)

        t_stats, p_values = stats.ttest_ind(post_learning_array, pre_learning_array, axis=0)
        t_stats = np.where(p_values < 0.05, t_stats, 0)
        difference = np.where(p_values < 0.05, difference, 0)

        # Create Images
        difference_image = Learning_Utils.create_image_from_data(difference, indicies, image_height, image_width)
        pre_learning_image = Learning_Utils.create_image_from_data(pre_learning_mean, indicies, image_height, image_width)
        post_learning_image = Learning_Utils.create_image_from_data(post_learning_mean, indicies, image_height, image_width)

        # Apply Colourmaps
        difference_image = difference_colourmap.to_rgba(difference_image)
        pre_learning_image = positive_colourmap.to_rgba(pre_learning_image)
        post_learning_image = positive_colourmap.to_rgba(post_learning_image)

        # Colour Background White
        difference_image[background_pixels] = [1,1,1,1]
        pre_learning_image[background_pixels] = [1, 1, 1, 1]
        post_learning_image[background_pixels] = [1, 1, 1, 1]

        # Plot Images
        condition_1_axis.imshow(pre_learning_image)
        condition_2_axis.imshow(post_learning_image)
        difference_axis.imshow(difference_image)

        # Remove Axis
        condition_1_axis.axis('off')
        condition_2_axis.axis('off')
        difference_axis.axis('off')

        # Set Titles
        condition_1_axis.set_title(condition_names[0])
        condition_2_axis.set_title(condition_names[1])
        difference_axis.set_title("Differences")

        # Save Figure
        figure_1.suptitle(str(time_values[count]) + "ms")

        # Add Colourbars
        plt.colorbar(mappable=positive_colourmap, ax=condition_1_axis, orientation='vertical', fraction=0.1, aspect=8)
        plt.colorbar(mappable=positive_colourmap, ax=condition_2_axis, orientation='vertical', fraction=0.1, aspect=8)
        plt.colorbar(mappable=difference_colourmap, ax=difference_axis, orientation='vertical', fraction=0.1, aspect=8)

        plt.savefig(os.path.join(save_directory, str(count).zfill(4) + ".png"))
        plt.clf()
        count += 1

"""
# Control Learning
filename = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Control_Combined_Tensor_Response_Baseline_Corrected"
condition_names = ["Pre_Learning", "Post_Learning"]
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Images/Control_Pre_Post_Learning_Baseline_Corrected"
analysis_name = "Hits_Pre_Post_Learning_response"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Learning_Utils.load_analysis_container(analysis_name)
create_effect_figure(filename, save_directory, condition_names, start_window, downsized=True)

# Mutant Learning
filename = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Mutant_Combined_Tensor_Response_Baseline_Corrected"
condition_names = ["Pre_Learning", "Post_Learning"]
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Images/Mutant_Pre_Post_Learning_Baseline_Corrected"
analysis_name = "Hits_Pre_Post_Learning_response"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Learning_Utils.load_analysis_container(analysis_name)
create_effect_figure(filename, save_directory, condition_names, start_window, downsized=True)

# Genotype Pre Learning Vis 1
filename = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Genotype_RT_Matched_Pre"
condition_names = ["Controls", "Mutants"]
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Images/Genotype_Pre_Learning_Response"
analysis_name = "Matched_Correct_Vis_1"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Learning_Utils.load_analysis_container(analysis_name)
create_effect_figure(filename, save_directory, condition_names, start_window, downsized=True)

# Genotype Post Learning Vis 1
filename = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Genotype_RT_Matched_Post"
condition_names = ["Controls", "Mutants"]
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Images/Genotype_Post_Learning_Response"
analysis_name = "Matched_Correct_Vis_1"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Learning_Utils.load_analysis_container(analysis_name)
create_effect_figure(filename, save_directory, condition_names, start_window, downsized=True)

# Genotype Post Learning Vis 2
filename = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Genotype_Correct_Rejections_Post_Learning"
condition_names = ["Controls", "Mutants"]
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Images/Genotype_Correct_Rejections_Response"
analysis_name = "Correct_Rejections_Response"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Learning_Utils.load_analysis_container(analysis_name)
create_effect_figure(filename, save_directory, condition_names, start_window, downsized=True)
"""


# Genotype Post Learning Vis 2
analysis_name = "Unrewarded_Contextual_Modulation"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Learning_Utils.load_analysis_container(analysis_name)
filename = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Genotype_Context_Vis_2"
condition_names = ["Controls", "Mutants"]
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Images/Genotype_Visual_Context_Vis_2"
create_effect_figure(filename, save_directory, condition_names, start_window, downsized=True)

# Control Contextual Modulation Vis 2
analysis_name = "Unrewarded_Contextual_Modulation"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Learning_Utils.load_analysis_container(analysis_name)
filename = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Control_Contextual_Vis_2"
condition_names = ["Visual Context", "Odour Context"]
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Images/Control_Contextual_Modulation_Vis_2"
create_effect_figure(filename, save_directory, condition_names, start_window, downsized=True)

analysis_name = "Unrewarded_Contextual_Modulation"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Learning_Utils.load_analysis_container(analysis_name)
filename = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Mutant_Contextual_Vis_2"
condition_names = ["Visual Context", "Odour Context"]
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Images/Mutant_Contextual_Modulation_Vis_2"
create_effect_figure(filename, save_directory, condition_names, start_window, downsized=True)

# Genotype Odour Context Vis 2
analysis_name = "Unrewarded_Contextual_Modulation"
[start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions] = Learning_Utils.load_analysis_container(analysis_name)
filename = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Genotype_Context_Vis_2_Odour_Context"
condition_names = ["Controls", "Mutants"]
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Learning_Analysis/Mixed_Effects_Modelling/Images/Genotype_Odour_Context_Vis_2"
create_effect_figure(filename, save_directory, condition_names, start_window, downsized=True)
