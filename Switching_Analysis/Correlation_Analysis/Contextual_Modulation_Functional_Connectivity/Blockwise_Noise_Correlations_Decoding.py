import numpy as np
import matplotlib.pyplot as plt
import os
import numbers
from matplotlib.pyplot import GridSpec
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.manifold import SpectralEmbedding
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist
from statsmodels.stats.multitest import fdrcorrection
from matplotlib.pyplot import cm
from sklearn.neural_network import MLPClassifier

def remove_early_onsets(onsets_list):

    thresholded_onsets = []
    for onset in onsets_list:
        if onset > 3000:
            thresholded_onsets.append(onset)

    return thresholded_onsets


def load_onsets_blockwise(base_directory, onset_file, onset_behaviour_matrix_field, min_block_size=5, early_onset_cutoff = 3000):

    # Load onsets
    onset_list = np.load(os.path.join(base_directory, "Stimuli_Onsets", onset_file))
    onset_list = remove_early_onsets(onset_list)

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

    window_size = 12

    # Create Block Onset Dict
    block_onset_dict = {}
    for trial in behaviour_matrix:
        trial_onset = trial[onset_behaviour_matrix_field]
        if isinstance(trial_onset, numbers.Number):
            for selected_onset in onset_list:
                if trial_onset > (selected_onset - window_size) and trial_onset < selected_onset + window_size:
                    block = trial[8]
                    block_onset_dict[selected_onset] = block

    # Split Up BLock Onset Dict
    onset_block_meta_list = []
    block_numbers = list(block_onset_dict.values())
    block_numbers = list(set(block_numbers))


    for block in block_numbers:
        block_onset_list = []
        for onset in onset_list:
            onset_block = block_onset_dict[onset]
            if onset_block == block:
                block_onset_list.append(onset)

        if len(block_onset_list) >= min_block_size:
            onset_block_meta_list.append(block_onset_list)

    return onset_block_meta_list


def load_trial_tensor(delta_f_matrix, onset_list, start_window, stop_window, subtract_baseline):

    trial_tensor = []
    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window
        trial_data = delta_f_matrix[trial_start:trial_stop]

        if subtract_baseline == True:
            trial_baseline = trial_data[0:np.abs(start_window)]
            trial_baseline = np.mean(trial_baseline, axis=0)
            trial_data = np.subtract(trial_data, trial_baseline)

        trial_tensor.append(trial_data)

    trial_tensor = np.array(trial_tensor)

    return trial_tensor


def get_noise_tensor(condition_tensor):

    condition_mean = np.mean(condition_tensor, axis=0)

    noise_tensor = []
    for trial in condition_tensor:
        trial = np.subtract(trial, condition_mean)
        noise_tensor.append(trial)

    noise_tensor = np.array(noise_tensor)
    return noise_tensor


def get_mi_matrix(noise_tensor):
    print("MI Tensor Shape", np.shape(noise_tensor))
    number_of_traces = np.shape(noise_tensor)[1]

    mi_matrix = np.zeros((number_of_traces, number_of_traces))
    for region_1 in range(number_of_traces):
        for region_2 in range(number_of_traces):

            mi = mutual_info_regression(X=noise_tensor[:, region_1].reshape(-1, 1), y=noise_tensor[:, region_2])
            mi_matrix[region_1, region_2] = mi

    mi_matrix = np.array(mi_matrix)
    np.fill_diagonal(mi_matrix, 0)
    return mi_matrix


def get_gc_value(from_trace, to_trace, order):

    trace_to_predict = to_trace[order:]
    number_of_timepoints = len(from_trace)

    autoregressive_tensor = []
    bivariate_tensor = []

    for x in range(1, order):
        signal_start = order - x
        signal_stop = number_of_timepoints - x

        autoregressive_trace = to_trace[signal_start:signal_stop]
        autoregressive_tensor.append(autoregressive_trace)

        bivariate_trace = from_trace[signal_start:signal_stop]
        bivariate_tensor.append(bivariate_trace)

    autoregressive_tensor = np.array(autoregressive_tensor)
    bivariate_tensor = np.array(bivariate_tensor)

    autoregressive_tensor = np.transpose(autoregressive_tensor)
    bivariate_tensor = np.transpose(bivariate_tensor)

    #print("Autoregressive tensor", np.shape(autoregressive_tensor))
    #print("Bivariate Tensor", np.shape(bivariate_tensor))

    full_tensor = np.hstack([autoregressive_tensor, bivariate_tensor])

    partial_model = LinearRegression()
    full_model = LinearRegression()

    partial_model.fit(X=autoregressive_tensor, y=trace_to_predict)
    full_model.fit(X=full_tensor, y=trace_to_predict)

    autogressive_prediction = partial_model.predict(X=autoregressive_tensor)
    autogressive_error = np.subtract(trace_to_predict, autogressive_prediction)
    autogressive_error = np.var(autogressive_error)

    bivariate_prediction = full_model.predict(X=full_tensor)
    bivariate_error = np.subtract(trace_to_predict, bivariate_prediction)
    bivariate_error = np.var(bivariate_error)

    granger_causality = np.log(autogressive_error / bivariate_error)

    return granger_causality



def get_gc_matrix(noise_tensor, order=5):

    number_of_traces = np.shape(noise_tensor)[1]
    gc_matrix = np.zeros((number_of_traces, number_of_traces))

    for region_1 in range(number_of_traces):
        for region_2 in range(number_of_traces):
            if region_1 != region_2:
                gc_value = get_gc_value(noise_tensor[region_1], noise_tensor[region_2], order)
                gc_matrix[region_1, region_2] = gc_value

    return gc_matrix


def perform_blockwise_correlations(base_directory, onset_file, onset_behaviour_matrix_field,  start_window, stop_window, subtract_baseline):

    # Load Blockwise Onsets
    blockwise_onset_list = load_onsets_blockwise(base_directory, onset_file, onset_behaviour_matrix_field)

    # Load Delta F Matrix
    cluster_activity_matrix = np.load(os.path.join(base_directory, "Cluster_Activity_Matrix.npy"))


    cluster_activity_matrix[:, 33] = 0
    cluster_activity_matrix[:, 29] = 0
    cluster_activity_matrix[:, 26] = 0
    cluster_activity_matrix[:, 27] = 0
    cluster_activity_matrix[:, 28] = 0
    cluster_activity_matrix[:, 34] = 0

    cluster_activity_matrix = cluster_activity_matrix[:, 1:]

    correlation_matrix_list = []
    for block in blockwise_onset_list:

        # Load Trial Tensor
        trial_tensor = load_trial_tensor(cluster_activity_matrix, block, start_window, stop_window, subtract_baseline)

        # Convert To Noise Tensor
        trial_tensor = get_noise_tensor(trial_tensor)

        # Flatten Noise Tensor
        number_of_trials, trial_length, number_of_regions = np.shape(trial_tensor)
        trial_tensor = np.reshape(trial_tensor, (number_of_trials * trial_length, number_of_regions))

        # Create Correlarion Matrix
        correlation_matrix = np.corrcoef(np.transpose(trial_tensor))

        correlation_matrix_list.append(correlation_matrix)

    return correlation_matrix_list


def plot_correlation_matrix_lists(condition_1_list, condition_2_list):

    number_of_condition_1_matricies = len(condition_1_list)
    number_of_condition_2_matricies = len(condition_2_list)

    figure_1 = plt.figure()
    number_of_rows = 2
    number_of_columns = np.max([number_of_condition_1_matricies, number_of_condition_2_matricies])
    gridspec_1 = GridSpec(nrows=number_of_rows, ncols=number_of_columns, figure=figure_1)

    for matrix_index in range(number_of_condition_1_matricies):
        correlation_matrix = condition_1_list[matrix_index]
        axis = figure_1.add_subplot(gridspec_1[0, matrix_index])
        axis.imshow(correlation_matrix, vmin=0, vmax=1, cmap='jet')

    for matrix_index in range(number_of_condition_2_matricies):
        correlation_matrix = condition_2_list[matrix_index]
        axis = figure_1.add_subplot(gridspec_1[1, matrix_index])
        axis.imshow(correlation_matrix, vmin=0, vmax=1, cmap='jet')

    plt.show()


def plot_means(condition_1_mean_list, condition_2_mean_list, difference_list):

    number_of_sessions = len(condition_1_mean_list)
    figure_1 = plt.figure()
    number_of_rows = 3
    number_of_columns = number_of_sessions
    gridspec_1 = GridSpec(nrows=number_of_rows, ncols=number_of_columns, figure=figure_1)


    for session_index in range(number_of_sessions):

        condition_1_axis = figure_1.add_subplot(gridspec_1[0, session_index])
        condition_2_axis = figure_1.add_subplot(gridspec_1[1, session_index])
        difference_axis = figure_1.add_subplot(gridspec_1[2, session_index])

        condition_1_axis.imshow(condition_1_mean_list[session_index], vmin=0, vmax=1, cmap='jet')
        condition_2_axis.imshow(condition_2_mean_list[session_index], vmin=0, vmax=1, cmap='jet')
        difference_axis.imshow(difference_list[session_index], vmin=-0.5, vmax=0.5, cmap='bwr')

    plt.show()

def sort_matrix(matrix):

    # Cluster Matrix
    Z = ward(pdist(matrix))

    # Get Dendogram Leaf Order
    new_order = leaves_list(Z)

    # Sorted Matrix
    sorted_matrix = matrix[:, new_order][new_order]

    return sorted_matrix




def balance_classes(condition_1_onsets, condition_2_onsets):

    condition_1_onsets = list(condition_1_onsets)
    condition_2_onsets = list(condition_2_onsets)

    number_of_condition_1_trials = len(condition_1_onsets)
    number_of_condition_2_trials = len(condition_2_onsets)

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
        #model = LogisticRegression(penalty='elasticnet', max_iter=5000, solver='saga', l1_ratio=0.5)
        model = LogisticRegression(penalty='l2', max_iter=5000, solver='saga', C=1)

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



def decode_context_connectivity_matricies(condition_1_matrix_list, condition_2_matrix_list):

    condition_1_matrix_list = np.nan_to_num(condition_1_matrix_list)
    condition_2_matrix_list = np.nan_to_num(condition_2_matrix_list)

    condition_1_labels = np.ones(np.shape(condition_1_matrix_list)[0])
    condition_2_labels = np.zeros(np.shape(condition_2_matrix_list)[0])

    combined_labels = np.concatenate([condition_1_labels, condition_2_labels])
    combined_data = np.vstack([condition_1_matrix_list, condition_2_matrix_list])

    mean_score, decoder_weights = perform_k_fold_cross_validation(combined_data, combined_labels)
    decoder_weights = np.reshape(decoder_weights, (58, 58))

    #np.save("/media/matthew/Expansion/Widefield_Analysis/Switching_Analysis/Decoding/Decoder_weights.npy", decoder_weights)

    """
    weight_magnitude = np.max(np.abs(decoder_weights))
    plt.imshow(decoder_weights, cmap='bwr', vmin=-weight_magnitude, vmax=weight_magnitude)
    plt.show()

    sorted_wights = sort_matrix(decoder_weights)
    plt.imshow(sorted_wights, cmap='bwr', vmin=-weight_magnitude, vmax=weight_magnitude)
    plt.imshow(sorted_wights)
    plt.show()
    """
    return mean_score, decoder_weights


def run_decoding_analysis(session_list,condition_1, condition_2, start_window, stop_window, subtract_baseline):

    all_condition_1_matricies = []
    all_condition_2_matricies = []

    for base_directory in session_list:

        condition_1_matricies = perform_blockwise_correlations(base_directory, condition_1, condition_1_behaviour_matrix_field, start_window, stop_window, subtract_baseline)
        condition_2_matricies = perform_blockwise_correlations(base_directory, condition_2, condition_2_behaviour_matrix_field, start_window, stop_window, subtract_baseline)

        #plot_correlation_matrix_lists(condition_1_matricies, condition_2_matricies)

        for matrix in condition_1_matricies:
            all_condition_1_matricies.append(np.ndarray.flatten(matrix))

        for matrix in condition_2_matricies:
            all_condition_2_matricies.append(np.ndarray.flatten(matrix))

    all_condition_1_matricies = np.nan_to_num(all_condition_1_matricies)
    all_condition_2_matricies = np.nan_to_num(all_condition_2_matricies)
    all_condition_1_matricies, all_condition_2_matricies = balance_classes(all_condition_1_matricies, all_condition_2_matricies)

    mean_score, mean_weights = decode_context_connectivity_matricies(all_condition_1_matricies, all_condition_2_matricies)

    return mean_score, mean_weights


def view_mean_weight_vector(mean_weight_vector):

    # Load Consensus Clusters
    consensus_clusters = np.load(r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Final_Consensus_Clusters.npy")

    reconstructed_vector = np.zeros(np.shape(consensus_clusters))

    unique_clusters = list(np.unique(consensus_clusters))
    unique_clusters.remove(0)

    colourmap = cm.get_cmap('bwr')

    number_of_clusters = len(unique_clusters)
    for cluster_index in range(number_of_clusters):
        cluster = unique_clusters[cluster_index]

        cluster_mask = np.where(consensus_clusters == cluster, 1, 0)
        cluster_pixels = np.nonzero(cluster_mask)

        cluster_value = mean_weight_vector[cluster_index]
        reconstructed_vector[cluster_pixels] = cluster_value

    plt.imshow(reconstructed_vector)
    plt.show()

"""
18 Onset closest Frame
19 Offset Closest Frame
20 Irrel Onset Closest Frame
21 Irrel Offset Closest Frame
"""

session_list = [

    #r"/media/matthew/Seagate Expansion Drive2/Processed_Widefield_Data/NRXN78.1D/2020_12_05_Switching_Imaging",

    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_04_08_Transition_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_04_02_Transition_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging",

    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_03_23_Transition_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_03_31_Transition_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging",

    "/media/matthew/Expansion/Widefield_Analysis/NXAK22.1A/2021_10_29_Transition_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK22.1A/2021_11_03_Transition_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK22.1A/2021_11_05_Transition_Imaging",

    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_06_13_Transition_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_06_15_Transition_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",
]

consensus_clusters = np.load(r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Final_Consensus_Clusters.npy")
plt.imshow(consensus_clusters)
plt.show()


condition_1 = "visual_context_stable_vis_2_onsets.npy"
condition_1_behaviour_matrix_field = 18

condition_2 = "odour_context_stable_vis_2_onsets.npy"
condition_2_behaviour_matrix_field = 20


start_window = -55
stop_window = 0
subtract_baseline = False
mean_score_list = []
for x in range(10):
    mean_score, mean_weights = run_decoding_analysis(session_list, condition_1, condition_2, start_window, stop_window, subtract_baseline)
    print("Mean score", mean_score)
    mean_score_list.append(mean_score)
grand_mean = np.mean(mean_score_list)
print("Grand Mean", grand_mean)
plt.imshow(mean_weights)
plt.show()

mean_vector = np.mean(np.abs(mean_weights), axis=0)
view_mean_weight_vector(mean_vector)
np.save(r"/media/matthew/Expansion/Widefield_Analysis/Switching_Analysis/Decoding/Decoder_weights.npy", mean_weights)
"""
mean_score_list = []
for x in range(-69, 56):
    start_window = x
    stop_window = x + 42
    mean_score, mean_weights = run_decoding_analysis(session_list,condition_1, condition_2, start_window, stop_window, subtract_baseline)
    mean_score_list.append(mean_score)
    print("TIme: ", x, "MEan Score", mean_score)

plt.plot(mean_score_list)
plt.show()
"""