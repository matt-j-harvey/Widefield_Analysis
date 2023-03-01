import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm

# Remove This Later
import warnings
warnings.filterwarnings("ignore")


def generate_pseudodata(image_width=30, image_height=30, noise_level=0.2, signal_level=1, n_mice=6):

    # Create Lists To Hold Trial Details
    mouse_list = []
    genotype_list = []
    condition_list = []
    data_list = []


    common_signal = np.zeros((image_height, image_width))
    common_signal[20:30, 10:20] = signal_level

    mutant_signal = np.zeros((image_height, image_width))
    mutant_signal[0:10, 10:20] = signal_level

    plt.title("Common Signal")
    plt.imshow(common_signal)
    plt.show()

    plt.title("Genotype Signal")
    plt.imshow(mutant_signal)
    plt.show()

    # Create Pre Learning Data
    mouse_index = 0

    # Controls
    for x in range(n_mice):
        pre_learning_data = np.random.normal(loc=0, scale=noise_level, size=(image_height,image_width))
        mouse_list.append(mouse_index)
        genotype_list.append(0)
        condition_list.append(0)
        data_list.append(pre_learning_data)
        mouse_index += 1

    # Mutants
    for x in range(n_mice):
        pre_learning_data = np.random.normal(loc=0, scale=noise_level, size=(image_height, image_width))
        mouse_list.append(mouse_index)
        genotype_list.append(1)
        condition_list.append(0)
        data_list.append(pre_learning_data)
        mouse_index += 1



    # Create Post Learning Data
    mouse_index = 0

    # Controls
    for x in range(n_mice):
        post_learning_data = np.random.normal(loc=0, scale=noise_level, size=(image_height, image_width))
        post_learning_data = np.add(post_learning_data, common_signal)
        mouse_list.append(mouse_index)
        genotype_list.append(0)
        condition_list.append(1)
        data_list.append(post_learning_data)
        mouse_index += 1

    # Mutants
    for x in range(n_mice):
        post_learning_data = np.random.normal(loc=0, scale=noise_level, size=(image_height, image_width))
        post_learning_data = np.add(post_learning_data, common_signal)
        post_learning_data = np.add(post_learning_data, mutant_signal)

        mouse_list.append(mouse_index)
        genotype_list.append(1)
        condition_list.append(1)
        data_list.append(post_learning_data)
        mouse_index += 1



    data_list = np.array(data_list)
    return mouse_list, genotype_list, condition_list, data_list


def repackage_into_dataframe(pixel_trace, mouse_list, genotype_list, condition_list):

    # Combine_Into Dataframe
    dataframe = pd.DataFrame(dtype=np.float64)
    dataframe["Data_Value"] = pixel_trace
    dataframe["Mouse"] = mouse_list
    dataframe["Genotype"] = genotype_list
    dataframe["Condition"] = condition_list

    return dataframe


def mixed_effects_random_slope_and_intercept(dataframe):

    model = sm.MixedLM.from_formula("Data_Value ~ Condition + Genotype", dataframe, re_formula="Condition", groups=dataframe["Mouse"])
    model_fit = model.fit(method='lbfgs')
    parameters = model_fit.params

    #print(model_fit.summary())

    condition_slope = parameters[1]
    genotype_slope = parameters[2]

    condition_p_value = model_fit.pvalues["Condition"]
    condition_z_stat = model_fit.tvalues["Condition"]

    genotype_p_value = model_fit.pvalues["Genotype"]
    genotype_z_stat = model_fit.tvalues["Genotype"]

    return condition_p_value, condition_z_stat, condition_slope, genotype_p_value, genotype_z_stat, genotype_slope


image_height = 30
image_width = 30

condition_t_tensor = np.zeros((image_height, image_width))
genotype_t_tensor = np.zeros((image_height, image_width))

mouse_list, genotype_list, condition_list, data_list = generate_pseudodata(image_height=image_height, image_width=image_width)
print("mouse_list", mouse_list)
print("genotype_list", genotype_list)
print("condition_list", condition_list)
print("data_list", np.shape(data_list))

for y in tqdm(range(image_height)):
    for x in range(image_width):

        pixel_trace = data_list[:, y, x]
        pixel_dataframe = repackage_into_dataframe(pixel_trace, mouse_list, genotype_list, condition_list)
        condition_p_value, condition_z_stat, condition_slope, genotype_p_value, genotype_z_stat, genotype_slope = mixed_effects_random_slope_and_intercept(pixel_dataframe)

        condition_t_tensor[y, x] = condition_z_stat
        genotype_t_tensor[y, x] = genotype_z_stat

plt.title("Condition t tensor")
plt.imshow(condition_t_tensor)
plt.show()


plt.title("Genotype t tensor")
plt.imshow(genotype_t_tensor)
plt.show()
