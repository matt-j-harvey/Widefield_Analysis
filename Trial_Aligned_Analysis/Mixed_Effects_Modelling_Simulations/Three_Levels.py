import numpy as np
import pandas as pd
from pymer4.models import Lmer
import matplotlib.pyplot as plt
from matplotlib import cm

import statsmodels.api as sm
import statsmodels.formula.api as smf


 # Create Pseudodata
def generate_pseudodata(n_mice, n_sessions, n_trials, fixed_effect, mouse_effect, mouse_intercept, session_effect, session_intercept):

    activity_list = []
    mouse_list = []
    session_list = []
    condition_list = []

    # Metadata Structure = [Mouse_index, session_index, condition_index]
    session_counter = 0

    for mouse_index in range(n_mice):
        for session_index in range(n_sessions):
            for trial_index in range(n_trials):
                for condition_index in range(2):
                    trial_baseline = np.random.normal(loc=0, size=1, scale=0.1)[0]
                    datapoint = trial_baseline + (condition_index * fixed_effect) + (condition_index * mouse_effect[mouse_index]) + (condition_index * session_effect[session_index]) + mouse_intercept[mouse_index] + session_intercept[session_index]

                    activity_list.append(datapoint)
                    mouse_list.append("M_" + str(mouse_index).zfill(3))
                    session_list.append("S_" + str(session_counter).zfill(3))
                    condition_list.append(condition_index)

            session_counter += 1

    return activity_list, mouse_list, session_list, condition_list



def repackage_data_into_dataframe(activity_list, mouse_list, session_list, condition_list):

    # Metadata Structure = [Mouse_index, session_index, condition_index]
    dataframe = pd.DataFrame(dtype=np.float64)
    dataframe["Data_Value"] = activity_list
    dataframe["Mouse"] = mouse_list
    dataframe["Session"] = session_list
    dataframe["Condition"] = condition_list

    return dataframe




def plot_three_level_data(dataframe, model, results):

    # Colour By Mouse
    mouse_list = dataframe["Mouse"]

    # Create Figure
    figure_1 = plt.figure()
    swarm_dispersion = 0.1
    colourmap = cm.get_cmap('tab20c')

    # Extract Random Effect Intercepts and Coefs
    random_effects = model.ranef
    random_effects = np.array(random_effects)

    session_random_effects = np.array(random_effects[0])
    mouse_random_effects = np.array(random_effects[1])
    print("Random Effects", random_effects)
    print("Random Effects Shape", np.shape(random_effects))

    print("Mouse Random Effects", mouse_random_effects)
    print("Session Random Effects", session_random_effects)

    # Extract Fixed Effect
    results = np.array(results)
    fixed_intercept = results[0, 0]
    fixed_slope = results[1, 0]

    unique_mice = np.unique(mouse_list)
    number_of_mice = len(unique_mice)
    number_of_sessions = len(np.unique(dataframe["Session"]))

    mouse_index = 0
    session_index = 0
    for mouse_name in unique_mice:

        # Create Mouse Axis
        mouse_axis = figure_1.add_subplot(1, number_of_mice, mouse_index + 1)
        mouse_axis.set_ylim([-2, 5])

        # Extract Mouse Data
        mouse_data = dataframe[(dataframe['Mouse'] == mouse_name)]



        # Extract Sessions
        mouse_sessions = mouse_data["Session"]
        unique_sessions = np.unique(mouse_sessions)
        print("Mouse ", mouse_name, "Sessios: ", unique_sessions)

        for session_name in unique_sessions:

            # Extract Session Data
            session_condition_1_data = dataframe[(dataframe['Mouse'] == mouse_name) & (dataframe['Session'] == session_name) & (dataframe['Condition'] == 0)]["Data_Value"]
            session_condition_2_data = dataframe[(dataframe['Mouse'] == mouse_name) & (dataframe['Session'] == session_name) * (dataframe['Condition'] == 1)]["Data_Value"]

            # Scatter Datapoints
            session_colour = colourmap(float(session_index) / number_of_sessions)

            n_condition_1_points = len(session_condition_1_data)
            n_condition_2_points = len(session_condition_2_data)

            # Create X Values
            condition_1_x_values = np.add(np.zeros(n_condition_1_points), np.random.uniform(low=-swarm_dispersion, high=swarm_dispersion, size=n_condition_1_points))
            condition_2_x_values = np.add(np.ones(n_condition_2_points), np.random.uniform(low=-swarm_dispersion, high=swarm_dispersion, size=n_condition_2_points))

            mouse_axis.scatter(x=condition_1_x_values, y=session_condition_1_data, color=session_colour, alpha=0.3)
            mouse_axis.scatter(x=condition_2_x_values, y=session_condition_2_data, color=session_colour, alpha=0.3)

            # Plot Fitted Random Effect Line
            full_intercept = fixed_intercept + mouse_random_effects[mouse_index][0] + session_random_effects[session_index][0]
            full_slope = fixed_slope + mouse_random_effects[mouse_index][1] + session_random_effects[session_index][1]
    
            mouse_line_y = np.multiply([0, 1], full_slope) + full_intercept
            mouse_axis.plot([0, 1], mouse_line_y, c=session_colour)

            session_index += 1
        mouse_index += 1
    plt.show()





# 1 Fixed Effect - Context
# 2 Levels of Random Effect - Mouse + Session

# Simulation Settings
n_mice = 5
n_sessions = 4
n_trials = 20


fixed_effect = 2
mouse_effect = 0.4
session_effect = 0.4

# Generate Random Effects
mouse_effects = np.around(np.random.normal(loc=0, scale=mouse_effect, size=n_mice), decimals=2)
mouse_intercepts = np.around(np.random.normal(loc=0, scale=mouse_effect, size=n_mice), decimals=2)

session_effects = np.around(np.random.normal(loc=0, scale=session_effect, size=n_sessions), decimals=2)
session_intercepts = np.around(np.random.normal(loc=0, scale=session_effect, size=n_sessions), decimals=2)

print("")
print("Mouse Effects", mouse_effects)
print("Mouse Intercepts", mouse_intercepts)
print("Session Effects", session_effects)
print("Session Intercepts", session_intercepts)
print("")

# Generate Data
activity_list, mouse_list, session_list, condition_list = generate_pseudodata(n_mice, n_sessions, n_trials, fixed_effect, mouse_effects, mouse_intercepts, session_effects, session_intercepts)

# Convert To Dataframe
dataframe = repackage_data_into_dataframe(activity_list, mouse_list, session_list, condition_list)

# Perform Mixed Effects Modelling
model = Lmer('Data_Value ~ Condition + (1 + Condition|Mouse) + (1 + Condition|Session)', data=dataframe)

results = model.fit()
print("Results")
print(results)
print("")

print("Condition results")
results_array = np.array(results)
print("Results shape", np.shape(results_array))
slope = results_array[1, 0]
t_statistic = results_array[1, 5]
p_value = results_array[1, 6]

print("Slope", slope, "t statistic", t_statistic, "p value", p_value)


# Plot Results
plot_three_level_data(dataframe, model, results)

