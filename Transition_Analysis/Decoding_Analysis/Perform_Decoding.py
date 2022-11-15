import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
import mat73
from scipy.io import loadmat
import matplotlib.ticker as mtick

import Load_Decoding_Data

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def load_matlab_sessions(base_directory):
    """Given A Directory - Return all .Mat Files"""

    matlab_file_list = []
    all_files = os.listdir(base_directory)
    for file in all_files:
        if file[-3:] == "mat":
            matlab_file_list.append(os.path.join(base_directory, file))

    return matlab_file_list


def get_trial_tensor(delta_f_matrix, onset_list, start_window, stop_window):

    number_of_timepoints = np.shape(delta_f_matrix)[0]

    trial_tensor = []
    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window

        if trial_start >= 0 and trial_stop < number_of_timepoints - 1:
            trial_data = delta_f_matrix[trial_start:trial_stop]
            trial_tensor.append(trial_data)

    trial_tensor = np.array(trial_tensor)

    return trial_tensor



def balance_classes(condition_1_onsets, condition_2_onsets):

    number_of_condition_1_trials = len(condition_1_onsets)
    number_of_condition_2_trials = len(condition_2_onsets)
    print("Condition 1 trials pre balance: ", number_of_condition_1_trials)
    print("Condition 2 trials pre balance: ", number_of_condition_2_trials)

    # If There Are Balanced, Great! Dont change a thing
    if number_of_condition_1_trials == number_of_condition_2_trials:
        return condition_1_onsets, condition_2_onsets

    # Else Remove Random Samples From The Larger Class
    else:
        smallest_class = np.min([number_of_condition_1_trials, number_of_condition_2_trials])
        largest_class = np.max([number_of_condition_1_trials, number_of_condition_2_trials])
        trials_to_remove = largest_class - smallest_class

        if number_of_condition_1_trials > number_of_condition_2_trials:
            for x in range(trials_to_remove):
                random_index = int(np.random.uniform(low=0, high=len(condition_1_onsets)))
                del condition_1_onsets[random_index]
            return condition_1_onsets, condition_2_onsets


        else:
            for x in range(trials_to_remove):
                random_index = int(np.random.uniform(low=0, high=len(condition_2_onsets)))
                del condition_2_onsets[random_index]
            return condition_1_onsets, condition_2_onsets




def perform_k_fold_cross_validation(data, labels, number_of_folds=5):

    score_list = []
    weight_list = []

    # Get Indicies To Split Data Into N Train Test Splits
    k_fold_object = StratifiedKFold(n_splits=number_of_folds, shuffle=True) #random_state=42

    # Iterate Through Each Split
    for train_index, test_index in k_fold_object.split(data, y=labels):

        # Split Data Into Train and Test Sets
        data_train, data_test = data[train_index], data[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

        # Train Model
        model = LogisticRegression(penalty='l2', max_iter=500)
        #model = LinearDiscriminantAnalysis()
        model.fit(data_train, labels_train)

        # Test Model
        model_score = model.score(data_test, labels_test)

        # Add Score To Score List
        score_list.append(model_score)

        # Get Model Weights
        model_weights = model.coef_
        weight_list.append(model_weights)

    # Return Mean Score and Mean Model Weights
    mean_score = np.mean(score_list)

    weight_list = np.array(weight_list)
    mean_weights = np.mean(weight_list, axis=0)
    return mean_score, mean_weights





def perform_decoding(delta_f_matrix, condition_1_onsets, condition_2_onsets, start_window, stop_window):

    # Balanace Trial Numbers
    condition_1_onsets, condition_2_onsets = balance_classes(condition_1_onsets, condition_2_onsets)
    print("Condition 1 onsets", condition_1_onsets)
    print("Number Of Condition 1 Onsets Post Balance", len(condition_1_onsets))
    print("Number of condition 2 onsets post balance", len(condition_2_onsets))

    # Get Trial Tensors
    condition_1_data = get_trial_tensor(delta_f_matrix, condition_1_onsets, start_window, stop_window)
    print("Condition 1 data",  np.shape(condition_1_data))
    condition_2_data = get_trial_tensor(delta_f_matrix, condition_2_onsets, start_window, stop_window)
    print("Condition 2 data", np.shape(condition_2_data))

    combined_data = np.concatenate([condition_1_data, condition_2_data], axis=0)

    # Create Labels
    condition_1_labels = np.ones(np.shape(condition_1_data)[0])
    condition_2_labels = np.zeros(np.shape(condition_2_data)[0])
    data_labels = np.concatenate([condition_1_labels, condition_2_labels], axis=0)

    # Perform Decoding With K Fold Cross Validation
    temporal_score_list = []
    number_of_timepoints = np.shape(combined_data)[1]
    for timepoint_index in range(number_of_timepoints):
        mean_score, mean_weights = perform_k_fold_cross_validation(combined_data[:, timepoint_index], data_labels)
        temporal_score_list.append(mean_score)

    return temporal_score_list


def plot_group_performance(group_visual_temporal_score_list, group_odour_temporal_score_list, group_context_temporal_score_list, start_window, frame_timestep, plot_save_directory):

    # Convert To Arrays for Subsequent Computation
    group_visual_temporal_score_list = np.array(group_visual_temporal_score_list)
    group_odour_temporal_score_list = np.array(group_odour_temporal_score_list)
    group_context_temporal_score_list = np.array(group_context_temporal_score_list)

    # Get Group Means
    group_visual_temporal_score_mean = np.mean(group_visual_temporal_score_list, axis=0)
    group_odour_temporal_score_mean = np.mean(group_odour_temporal_score_list, axis=0)
    group_context_temporal_score_mean = np.mean(group_context_temporal_score_list, axis=0)

    # Get Group Standard Deviations
    group_visual_temporal_score_std = np.std(group_visual_temporal_score_list, axis=0)
    group_odour_temporal_score_std = np.std(group_odour_temporal_score_list, axis=0)
    group_context_temporal_score_std = np.std(group_context_temporal_score_list, axis=0)

    # Get Shading Upper and Lower Bounds
    group_visual_upper_bound = np.add(group_visual_temporal_score_mean, group_visual_temporal_score_std)
    group_visual_lower_bound = np.subtract(group_visual_temporal_score_mean, group_visual_temporal_score_std)
    group_odour_upper_bound = np.add(group_odour_temporal_score_mean, group_odour_temporal_score_std)
    group_odour_lower_bound = np.subtract(group_odour_temporal_score_mean, group_odour_temporal_score_std)
    group_context_upper_bound = np.add(group_context_temporal_score_mean, group_context_temporal_score_std)
    group_context_lower_bound = np.subtract(group_context_temporal_score_mean, group_context_temporal_score_std)

    # Plot Means
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, frame_timestep)

    plt.plot(x_values, group_visual_temporal_score_mean, c='b', label='Relevant Visual Stimuli')
    plt.plot(x_values, group_odour_temporal_score_mean, c='g', label='Odour')
    plt.plot(x_values, group_context_temporal_score_mean, c='m', label='Context')

    # Shade Standard Deviations
    plt.fill_between(x=x_values, y1=group_visual_lower_bound, y2=group_visual_upper_bound, color='b', alpha=0.2)
    plt.fill_between(x=x_values, y1=group_odour_lower_bound, y2=group_odour_upper_bound, color='g', alpha=0.2)
    plt.fill_between(x=x_values, y1=group_context_lower_bound, y2=group_context_upper_bound, color='m', alpha=0.2)

    # Add Legengds
    plt.legend(loc="lower right")

    # Add Line At Stimulus Onset
    plt.axvline(x=0, color='k', linestyle='--')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    plt.title("Decoding Performance In ACC Around Stimuli Onsets")
    plt.xlabel("Time (ms)")
    plt.ylabel("Accuracy")

    plt.savefig(os.path.join(plot_save_directory, "Group_Average.svg"))
    plt.show()


def view_trial_averaged_ai_traces(onsets, matlab_data, start_window=-20, stop_window=20):

    # Extract AI Data
    downsampled_ai = matlab_data['downsampled_AI']

    # Extract Vis 1 Trace
    # Extract Downsampled AI
    downsampled_ai = matlab_data['downsampled_AI']
    channel_names = downsampled_ai['chanNames']
    ai_data = downsampled_ai['data']

    # Extract Relevant Traces
    lick_trace = ai_data[channel_names.index('lick')]
    vis_1_trace = ai_data[channel_names.index('vis1')]
    vis_2_trace = ai_data[channel_names.index('vis2')]
    irrel_trace = ai_data[channel_names.index('irrel')]
    odour_1_trace = ai_data[channel_names.index('odr1')]
    odour_2_trace = ai_data[channel_names.index('odr2')]


def plot_group_average(group_visual_temporal_score_list, group_odour_temporal_score_list, group_context_temporal_score_list, start_window, stop_window, timestep, plot_save_directory):
    
    # Convert To Arrays
    group_visual_temporal_score_list = np.array(group_visual_temporal_score_list)
    group_odour_temporal_score_list = np.array(group_odour_temporal_score_list)
    group_context_temporal_score_list = np.array(group_context_temporal_score_list)

    # Get Means
    mean_visual_score = np.mean(group_visual_temporal_score_list, axis=0)
    mean_odour_score = np.mean(group_odour_temporal_score_list, axis=0)
    mean_context_score = np.mean(group_context_temporal_score_list, axis=0)

    # Get Standard Deviation
    visual_score_sd = np.std(group_visual_temporal_score_list, axis=0)
    odour_score_sd = np.std(group_odour_temporal_score_list, axis=0)
    context_score_sd = np.std(group_context_temporal_score_list, axis=0)

    # Get Upper and Lower Bounds
    visual_upper_limit = np.add(mean_visual_score, visual_score_sd)
    visual_lower_limit = np.subtract(mean_visual_score, visual_score_sd)
    
    odour_upper_limit = np.add(mean_odour_score, odour_score_sd)
    odour_lower_limit = np.subtract(mean_odour_score, odour_score_sd)

    context_upper_limit = np.add(mean_context_score, context_score_sd)
    context_lower_limit = np.subtract(mean_context_score, context_score_sd)

    # Plot These
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, timestep)

    # Plot Means
    axis_1.plot(x_values, mean_visual_score, c='b')
    axis_1.plot(x_values, mean_odour_score, c='g')
    axis_1.plot(x_values, mean_context_score, c='m')

    # Shade SDs
    axis_1.fill_between(x=x_values, y1=visual_lower_limit, y2=visual_upper_limit, color='b', alpha=0.2, label='Visual Stimuli')
    axis_1.fill_between(x=x_values, y1=odour_lower_limit, y2=odour_upper_limit, color='g', alpha=0.2, label='Odour Stimuli')
    axis_1.fill_between(x=x_values, y1=context_lower_limit, y2=context_upper_limit, color='m', alpha=0.2, label='Context')

    axis_1.axvline(0, c='k', linestyle='--')
    axis_1.set_title("Group Average")
    axis_1.set_ylim(0.4, 1.0)
    axis_1.legend()

    axis_1.set_xlabel("Time (ms)")
    axis_1.set_ylabel("Decoding Performance")

    if plot_save_directory != None:
        plt.savefig(os.path.join(plot_save_directory, "Group_Mean" + ".svg"))
    else:
        plt.show()
    plt.close()

def run_decoding_analysis(matlab_sessions,  start_window, stop_window, timestep, plot_save_directory=None):

    # Iterate Through Each Session
    group_visual_temporal_score_list = []
    group_odour_temporal_score_list = []
    group_context_temporal_score_list = []

    for session_dict in matlab_sessions:

        # Load Delta F Matrix
        delta_f_matrix = session_dict['delta_f']
        delta_f_matrix = np.nan_to_num(delta_f_matrix)
        delta_f_matrix = np.transpose(delta_f_matrix)

        # Load Required Onsets
        visual_context_hits = session_dict['relVis1Onsets']
        visual_context_correct_rejections = session_dict['relVis2Onsets']
        odour_context_hits = session_dict['odr1Onsets']
        odour_context_correct_rejections = session_dict['odr2Onsets']
        irrel_vis_2_ignored =  session_dict['irrelVis2Onsets']

        # Decode Visual Stimuli
        visual_temporal_score_list = perform_decoding(delta_f_matrix,
                                                      visual_context_hits,
                                                      visual_context_correct_rejections,
                                                      start_window, stop_window)

        odour_temporal_score_list = perform_decoding(delta_f_matrix,
                                                     odour_context_hits,
                                                     odour_context_correct_rejections,
                                                     start_window, stop_window)

        context_temporal_score_list = perform_decoding(delta_f_matrix,
                                                       visual_context_correct_rejections,
                                                       irrel_vis_2_ignored,
                                                       start_window, stop_window)

        # View Trial Averaged Traces
        session_name = session_dict['session_name']
        figure_1 = plt.figure()
        axis_1 = figure_1.add_subplot(1,1,1)
        x_values = list(range(start_window, stop_window))
        x_values = np.multiply(x_values, timestep)
        axis_1.plot(x_values, visual_temporal_score_list, c='b', label='Visual Stimuli')
        axis_1.plot(x_values, odour_temporal_score_list, c='g', label='Odour Stimuli')
        axis_1.plot(x_values, context_temporal_score_list, c='m', label='Context')
        axis_1.axvline(0, c='k', linestyle='--')
        axis_1.set_title(session_name)
        axis_1.set_ylim(0.4, 1.0)
        axis_1.legend()

        axis_1.set_xlabel("Time (ms)")
        axis_1.set_ylabel("Decoding Performance")

        if plot_save_directory != None:
            plt.savefig(os.path.join(plot_save_directory, session_name + ".svg"))
        else:
            plt.show()
        plt.close()


        group_visual_temporal_score_list.append(visual_temporal_score_list)
        group_odour_temporal_score_list.append(odour_temporal_score_list)
        group_context_temporal_score_list.append(context_temporal_score_list)


    plot_group_average(group_visual_temporal_score_list, group_odour_temporal_score_list, group_context_temporal_score_list, start_window, stop_window, timestep, plot_save_directory)

    return group_visual_temporal_score_list, group_odour_temporal_score_list, group_context_temporal_score_list





# Load Matlab Data
#base_directory = r"C:\\Users\\matth\\Documents\\Nick_Analysis_First_Draft\\Best_switching_sessions_all_sites"
#base_directory = r"C:\Users\matth\Documents\Nick_Analysis_First_Draft\Best_Switching_Sessions_Same_Imaging_Rate"
#lick_threshold_directory = r"C:\Users\matth\Documents\Nick_Analysis_First_Draft\Best_switching_sessions_all_sites\Lick_Thresholds"
#plot_save_directory = r"C:\Users\matth\Documents\Nick_Analysis_First_Draft\Decoding_Analysis\Plots"
"""
matlab_sessions = load_matlab_sessions(base_directory)
print("Nubmer of sessions: ", len(matlab_sessions))

# Decoding Parameters
start_window = -20
stop_window = 20
frame_timestep = 120

# Run Decoding
group_visual_temporal_score_list, group_odour_temporal_score_list, group_context_temporal_score_list = run_decoding_analysis(matlab_sessions, lick_threshold_directory, plot_save_directory)

# Plot Decoding Performance
plot_group_performance(group_visual_temporal_score_list, group_odour_temporal_score_list, group_context_temporal_score_list, start_window, frame_timestep, plot_save_directory)

"""

"""
data_directory = r"/home/matthew/Documents/Github_Code_Clean/Transition_Analysis/Decoding_Analysis/Data/ITI_OP"
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Transition_Figure/Decoding/ITI"

"""
data_directory = r"/home/matthew/Documents/Github_Code_Clean/Transition_Analysis/Decoding_Analysis/Data/Peristim_OP"
save_directory = r"/media/matthew/Expansion/Widefield_Analysis/Transition_Figure/Decoding/OP"

start_window = -20
stop_window = 20
timestep = 202

# Load Data
session_data_list = Load_Decoding_Data.load_decoding_data(data_directory)

# Run Decoding
run_decoding_analysis(session_data_list, start_window, stop_window, timestep, plot_save_directory=save_directory)

