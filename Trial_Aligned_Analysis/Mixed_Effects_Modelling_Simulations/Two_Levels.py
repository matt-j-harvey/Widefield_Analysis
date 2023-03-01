import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from matplotlib import cm
from pymer4.models import Lmer

 # Create Pseudodata
def generate_pseudodata(n_mice, n_trials, fixed_effect, mouse_effects, mouse_intercepts):

    activity_list = []
    condition_list = []
    mouse_list = []

    # Metadata Structure = [Mouse_index, session_index, condition_index]
    session_counter = 0

    for mouse_index in range(n_mice):
        for trial_index in range(n_trials):
            for condition_index in range(2):
                trial_baseline = np.random.normal(loc=0, size=1, scale=0.01)[0]
                datapoint = trial_baseline + (condition_index * fixed_effect) + (condition_index * mouse_effects[mouse_index]) + mouse_intercepts[mouse_index]

                activity_list.append(datapoint)
                condition_list.append(condition_index)
                mouse_list.append("M" + str(mouse_index))


            session_counter += 1

    return activity_list, condition_list, mouse_list


def repackage_data_into_dataframe(activity_list, condition_list, mouse_list):

    # Metadata Structure = [Mouse_index, session_index, condition_index]
    dataframe = pd.DataFrame()
    dataframe["Data_Value"] = activity_list
    dataframe["Mouse"] = mouse_list
    dataframe["Condition"] = condition_list

    return dataframe


def model_random_intercept(dataframe):
    # Random Intercept
    print("Random Intercept")
    model = smf.mixedlm('Data_Value ~ Condition  + C(Mouse)', dataframe, groups=dataframe['Mouse'])
    results = model.fit()
    summary = results.summary()
    print("Results")
    print(summary)


def plot_two_level_data(dataframe, model, results):

    # Colour By Mouse
    mouse_list = dataframe["Mouse"]


    # Create Figure
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    swarm_dispersion = 0.1
    colourmap = cm.get_cmap('hsv')

    # Extract Random Effect Intercepts and Coefs
    random_effects = model.ranef
    random_effects = np.array(random_effects)

    # Extract Fixed Effect
    results = np.array(results)
    fixed_intercept = results[0, 0]
    fixed_slope = results[1, 0]

    unique_mice = np.unique(mouse_list)
    number_of_mice = len(unique_mice)
    mouse_index = 0

    for mouse_name in unique_mice:

        # Extract Mouse Data
        mouse_condition_1_data = dataframe[(dataframe['Mouse'] == mouse_name) & (dataframe['Condition'] == 0)]["Data_Value"]
        mouse_condition_2_data = dataframe[(dataframe['Mouse'] == mouse_name) & (dataframe['Condition'] == 1)]["Data_Value"]

        # Scatter Datapoints
        mouse_colour = colourmap(float(mouse_index) / number_of_mice)

        n_condition_1_points = len(mouse_condition_1_data)
        n_condition_2_points = len(mouse_condition_2_data)

        condition_1_x_values = np.add(np.zeros(n_condition_1_points), np.random.uniform(low=-swarm_dispersion, high=swarm_dispersion, size=n_condition_1_points))
        condition_2_x_values = np.add(np.ones(n_condition_2_points), np.random.uniform(low=-swarm_dispersion, high=swarm_dispersion, size=n_condition_2_points))

        axis_1.scatter(x=condition_1_x_values, y=mouse_condition_1_data, color=mouse_colour, alpha=0.5)
        axis_1.scatter(x=condition_2_x_values, y=mouse_condition_2_data, color=mouse_colour, alpha=0.5)

        # Plot Fitted Random Effect Line
        random_intercept = random_effects[mouse_index, 0]
        random_slope = random_effects[mouse_index, 1]

        full_intercept = random_intercept + fixed_intercept
        full_slope = random_slope + fixed_slope

        mouse_line_y = np.multiply([0, 1], full_slope) + full_intercept
        axis_1.plot([0, 1], mouse_line_y, c=mouse_colour)

        mouse_index += 1
    plt.show()




# 1 Fixed Effect - Context
# 2 Levels of Random Effect - Mouse + Session

# Simulation Settings
n_mice = 5
n_trials = 20

fixed_effect = 2
mouse_effect = 0.3

# Generate Random Effects
mouse_effects = np.around(np.random.normal(loc=0, scale=mouse_effect, size=n_mice), decimals=2)
mouse_intercepts = np.around(np.random.normal(loc=0, scale=mouse_effect, size=n_mice), decimals=2)

# Generate Data
activity_list, condition_list, mouse_list = generate_pseudodata(n_mice, n_trials, fixed_effect, mouse_effects, mouse_intercepts)

# Convert To Dataframe
dataframe = repackage_data_into_dataframe(activity_list, condition_list, mouse_list)

# Perform Mixed Effects Modelling
model = Lmer('Data_Value ~ Condition + (Condition|Mouse)', data=dataframe)
results = model.fit(verbose=False, summarize=False)

# Plot Results
#plot_two_level_data(dataframe, model, results)