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
from pyEDM import *
import pandas as pd


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


def load_trial_tensor(delta_f_matrix, onset_list, start_window, stop_window):

    trial_tensor = []
    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window
        trial_data = delta_f_matrix[trial_start:trial_stop]

        """
        trial_baseline = trial_data[0:np.abs(start_window)]
        trial_baseline = np.mean(trial_baseline, axis=0)
        trial_data = np.subtract(trial_data, trial_baseline)
        """

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


def get_ccm_matrix(cluster_activity_matrix):
    number_of_regions = np.shape(cluster_activity_matrix)[1]
    connectivity_matrix = np.zeros((number_of_regions, number_of_regions))

    for region_1_index in range(number_of_regions):
        for region_2_index in range(region_1_index, number_of_regions):
            print("Region: ", region_1_index, " To ", region_2_index)

            region_1_trace = cluster_activity_matrix[:, region_1_index]
            region_2_trace = cluster_activity_matrix[:, region_2_index]

            dataframe_dict = {"region_0": region_1_trace, "region_1": region_1_trace, "region_2": region_2_trace}
            dataframe = pd.DataFrame(dataframe_dict)

            result_dataframe = CCM(dataFrame=dataframe, E=3, columns="region_1", target="region_2", libSizes="10 75 5", sample=100, showPlot=False);
            region_1_to_2 = np.mean(result_dataframe["region_1:region_2"])
            region_2_to_1 = np.mean(result_dataframe["region_2:region_1"])

            connectivity_matrix[region_1_index, region_2_index] = region_1_to_2
            connectivity_matrix[region_2_index, region_1_index] = region_2_to_1

    return connectivity_matrix


def perform_blockwise_correlations(base_directory, onset_file, onset_behaviour_matrix_field,  start_window, stop_window):

    # Load Blockwise Onsets
    blockwise_onset_list = load_onsets_blockwise(base_directory, onset_file, onset_behaviour_matrix_field)
    print("Onset List", blockwise_onset_list)
    # Load Delta F Matrix
    cluster_activity_matrix = np.load(os.path.join(base_directory, "Cluster_Activity_Matrix.npy"))
    cluster_activity_matrix = cluster_activity_matrix[:, 1:]
    print("Cluster Activity MAtrix Shape", np.shape(cluster_activity_matrix))

    correlation_matrix_list = []
    for block in blockwise_onset_list:

        # Load Trial Tensor
        trial_tensor = load_trial_tensor(cluster_activity_matrix, block, start_window, stop_window)

        # Convert To Noise Tensor
        trial_tensor = get_noise_tensor(trial_tensor)

        # Flatten Noise Tensor
        number_of_trials, trial_length, number_of_regions = np.shape(trial_tensor)
        trial_tensor = np.reshape(trial_tensor, (number_of_trials * trial_length, number_of_regions))

        #correlation_matrix = get_mi_matrix(trial_tensor)
        #plt.imshow(correlation_matrix)
        #plt.show()

        #correlation_matrix = get_gc_matrix(trial_tensor, order=5)
        correlation_matrix = get_ccm_matrix(trial_tensor)
        plt.imshow(correlation_matrix)
        plt.show()
        # Create Correlarion Matrix
        #correlation_matrix = np.corrcoef(np.transpose(trial_tensor))

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
        axis.imshow(correlation_matrix, vmin=0, vmax=0.2)

    for matrix_index in range(number_of_condition_2_matricies):
        correlation_matrix = condition_2_list[matrix_index]
        axis = figure_1.add_subplot(gridspec_1[1, matrix_index])
        axis.imshow(correlation_matrix, vmin=0, vmax=0.2)

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

        condition_1_axis.imshow(condition_1_mean_list[session_index], vmin=0, vmax=0.2)
        condition_2_axis.imshow(condition_2_mean_list[session_index], vmin=0, vmax=0.2)
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

def decode_context_connectivity_matricies(condition_1_matrix_list, condition_2_matrix_list):

    condition_1_matrix_list = np.nan_to_num(condition_1_matrix_list)
    condition_2_matrix_list = np.nan_to_num(condition_2_matrix_list)

    condition_1_labels = np.zeros(np.shape(condition_1_matrix_list)[0])
    condition_2_labels = np.ones(np.shape(condition_2_matrix_list)[0])
    combined_labels = np.concatenate([condition_1_labels, condition_2_labels])

    combined_data = np.vstack([condition_1_matrix_list, condition_2_matrix_list])

    print("Combined Labels Shape", np.shape(combined_labels))
    print("Combined Data Shape", np.shape(combined_data))

    decoder = LogisticRegression(max_iter=500)

    for n in range(5):
        strat_k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=n)
        scores = cross_val_score(decoder, combined_data, combined_labels, cv=strat_k_fold)
        mean_score = np.mean(scores)
        print("Score", mean_score)

    decoder.fit(X=combined_data, y=combined_labels)
    decoder_weights = decoder.coef_
    print("Decoder weights", np.shape(decoder_weights))

    decoder_weights = np.reshape(decoder_weights, (58, 58))

    np.save("/media/matthew/Expansion/Widefield_Analysis/Switching_Analysis/Decoding/Decoder_weights.npy", decoder_weights)

    weight_magnitude = np.max(np.abs(decoder_weights))
    plt.imshow(decoder_weights, cmap='bwr', vmin=-weight_magnitude, vmax=weight_magnitude)
    plt.show()

def decompose_connectivity_matricies():
    all_matricies = all_condition_1_matricies + all_condition_2_matricies
    all_matricies = np.array(all_matricies)
    all_matricies = np.nan_to_num(all_matricies)
    print("All Matricies Shape", np.shape(all_matricies))

    pca_model = FastICA(n_components=3).fit(all_matricies)

    for component in pca_model.components_:
        plt.imshow(np.reshape(component, (59, 59)))
        plt.show()

    all_condition_1_matricies = np.nan_to_num(all_condition_1_matricies)
    all_condition_2_matricies = np.nan_to_num(all_condition_2_matricies)
    all_condition_1_matricies_transformed = pca_model.transform(all_condition_1_matricies)
    all_condition_2_matricies_transformed = pca_model.transform(all_condition_2_matricies)
    print("All condition 1 transformed", np.shape(all_condition_1_matricies_transformed))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(all_condition_1_matricies_transformed[:, 0], all_condition_1_matricies_transformed[:, 1], all_condition_1_matricies_transformed[:, 2], c='b')
    ax.scatter(all_condition_2_matricies_transformed[:, 0], all_condition_2_matricies_transformed[:, 1], all_condition_2_matricies_transformed[:, 2], c='g')
    plt.show()


"""
18 Onset closest Frame
19 Offset Closest Frame
20 Irrel Onset Closest Frame
21 Irrel Offset Closest Frame
"""

session_list = [

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


condition_1 = "visual_context_stable_vis_2_onsets.npy"
condition_1_behaviour_matrix_field = 18

condition_2 = "odour_context_stable_vis_2_onsets.npy"
condition_2_behaviour_matrix_field = 20

start_window = -14
stop_window = 28

condition_1_mean_list = []
condition_2_mean_list = []
difference_matrix_list = []

all_matrix_list = []


all_condition_1_matricies = []
all_condition_2_matricies = []


for base_directory in session_list:

    print("Blokwise Correlations For Session: ", base_directory)
    condition_1_matricies = perform_blockwise_correlations(base_directory, condition_1, condition_1_behaviour_matrix_field, start_window, stop_window)
    condition_2_matricies = perform_blockwise_correlations(base_directory, condition_2, condition_2_behaviour_matrix_field, start_window, stop_window)

    plot_correlation_matrix_lists(condition_1_matricies, condition_2_matricies)

    condition_1_matrix_mean = np.mean(np.array(condition_1_matricies), axis=0)
    condition_2_matrix_mean = np.mean(np.array(condition_2_matricies), axis=0)
    difference = np.subtract(condition_1_matrix_mean, condition_2_matrix_mean)

    condition_1_mean_list.append(condition_1_matrix_mean)
    condition_2_mean_list.append(condition_2_matrix_mean)
    difference_matrix_list.append(difference)

    for matrix in condition_1_matricies:
        all_condition_1_matricies.append(np.ndarray.flatten(matrix))

    for matrix in condition_2_matricies:
        all_condition_2_matricies.append(np.ndarray.flatten(matrix))

plot_means(condition_1_mean_list, condition_2_mean_list, difference_matrix_list)
#decode_context_connectivity_matricies(all_condition_1_matricies, all_condition_2_matricies)
condition_1_mean_list = np.nan_to_num(condition_1_mean_list)
condition_2_mean_list = np.nan_to_num(condition_2_mean_list)

t_stats, p_values = stats.ttest_rel(condition_1_mean_list, condition_2_mean_list, axis=0, nan_policy="omit")
thresholded_t_stats = np.where(p_values < 0.05, t_stats, 0)
plt.imshow(thresholded_t_stats)
plt.show()

upper_triangle_indicies = np.triu_indices_from(p_values)
p_values = p_values[upper_triangle_indicies]
print("Upper trianlge p value shape", np.shape(p_values))

p_values = np.nan_to_num(p_values)
# N tests = 1682

p_values = np.ndarray.flatten(p_values)
rejected, adjusted_p_values = fdrcorrection(p_values, alpha=0.05)

print("p values", p_values)
print("Adjusted P values", np.sort(adjusted_p_values))
for p_value in np.sort(adjusted_p_values):
    print(p_value)
