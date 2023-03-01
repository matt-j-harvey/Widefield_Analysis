import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.pyplot import Normalize
from pathlib import Path
import numpy as np
import os


from Files import Session_List
import Analyse_Discrimination_Session


def get_longest_session_list(nested_session_list):

    number_of_sessions_list = []
    for mouse in nested_session_list:
        number_of_sessions = len(mouse)
        number_of_sessions_list.append(number_of_sessions)

    max_number_of_sessions = np.max(number_of_sessions_list)
    print("max number of sessions", max_number_of_sessions)
    return max_number_of_sessions



def calculate_percentage_correct(performance_dict):
    visual_hits = performance_dict["visual_hits.npy"]
    visual_misses = performance_dict["visual_misses.npy"]
    visual_false_alarms = performance_dict["visual_false_alarms.npy"]
    visual_correct_rejections = performance_dict["visual_correct_rejections"]
    percentage_correct = float(visual_hits + visual_correct_rejections) / (visual_hits + visual_misses + visual_false_alarms + visual_correct_rejections)
    return percentage_correct


def get_mice_names(nested_session_list):

    mouse_name_list = []

    for mouse in nested_session_list:
        session_path = Path(mouse[0])
        session_parts = session_path.parts
        mouse_name = session_parts[-2]
        mouse_name_list.append(mouse_name)

    return mouse_name_list



def plot_performance_over_learning(nested_session_list, metric="d_prime"):

    # Get Max Number of Sessions
    max_sessions = get_longest_session_list(nested_session_list)
    number_of_mice = len(nested_session_list)

    # Get Mouse Names
    mouse_name_list = get_mice_names(nested_session_list)
    print("Mouse Names", mouse_name_list)

    # Create Figure
    performance_matrix = np.zeros((max_sessions, number_of_mice))
    performance_colour_matrix = np.zeros((max_sessions, number_of_mice, 4))

    # Create Colourmap
    if metric == "d_prime":
        performance_colourmap = cm.ScalarMappable(cmap="viridis", norm=Normalize(vmin=0, vmax=3))

    elif metric == "percentage_correct":
        performance_colourmap = cm.ScalarMappable(cmap="viridis", norm=Normalize(vmin=0.4, vmax=1))



    for mouse_index in range(len(nested_session_list)):
        for session_index in range(len(nested_session_list[mouse_index])):

            session = nested_session_list[mouse_index][session_index]

            # Load Performance Dict
            performance_dictionary = np.load(os.path.join(session, "Behavioural_Measures", "Performance_Dictionary.npy"), allow_pickle=True)[()]

            if metric == "d_prime":
                session_performance = performance_dictionary["visual_d_prime"]

            elif metric == "percentage_correct":
                session_performance = calculate_percentage_correct(performance_dictionary)

            # Add Performance To Performance Matrix
            performance_matrix[session_index, mouse_index] = session_performance

            # Add Colour To Performance Colour Matrix
            session_colour = performance_colourmap.to_rgba(session_performance)
            performance_colour_matrix[session_index, mouse_index] = session_colour


    figure_1 = plt.figure(figsize=(7,20))
    axis_1 = figure_1.add_subplot(1,1,1)
    print("Performance Matrix", np.shape(performance_matrix))
    axis_1.imshow(performance_colour_matrix)

    axis_1.set_ylabel("Session")
    axis_1.set_xlabel("Mouse")
    axis_1.set_xticks(list(range(len(mouse_name_list))))
    axis_1.set_xticklabels(mouse_name_list)
    axis_1.xaxis.set_label_position('top')
    axis_1.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    #axis_1.tick_params(axis='xaxis', which='both', length=0)
    plt.xticks(rotation=90)

    # Add Text
    for i in range(max_sessions):
        for j in range(number_of_mice):
            text = axis_1.text(j, i, np.around(performance_matrix[i, j], 2), ha="center", va="center", color="w")
            print("Tet", text)

    plt.colorbar(mappable=performance_colourmap)
    plt.show()






# Load Session List
#selected_session_list = Session_List.nested_mutant_discrimination_sessions
selected_session_list = Session_List.nested_control_discrimination_sessions


# Analyse Discrimination Sessions
"""
for mouse in selected_session_list:
    for session in mouse:
        print("analysing session", session)
        Analyse_Discrimination_Session.analyse_discrimination_session(session)
"""

# Create Figure
#
#
#plot_performance_over_learning(selected_session_list, metric="d_prime")
plot_performance_over_learning(selected_session_list, metric="percentage_correct")
# Plot D Prime Maps and Percent Correct Maps