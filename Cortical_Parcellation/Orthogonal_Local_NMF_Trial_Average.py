import os

import scipy.ndimage
import tensorflow as tf
from tensorflow import keras
import tables
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow.experimental.numpy as tnp
import tensorflow_probability as tfp
from scipy.spatial import ConvexHull
import datetime

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")
sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Analysis/Trial_Aligned_Analysis")

import Widefield_General_Functions
import Create_Activity_Tensor




def create_coordinate_vectors(indicies, image_height, image_width):

    # Create X and Y Coordinate Vectors
    y_coordinate_vector = []
    x_coordinate_vector = []
    for index in indicies:
        y_coord = np.floor_divide(index, image_width)
        x_coord = index % image_width

        y_coordinate_vector.append(y_coord)
        x_coordinate_vector.append(x_coord)

    y_coordinate_vector = tf.convert_to_tensor(y_coordinate_vector, dtype=tf.float32)
    x_coordinate_vector = tf.convert_to_tensor(x_coordinate_vector, dtype=tf.float32)

    return y_coordinate_vector, x_coordinate_vector


def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[1]
        y0 = center[0]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


def seed_initial_regions_grid(base_directory):

    # Load Mask Details
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    # Height = 600
    # Width = 608
    x_offset = 4

    grid_size = 600
    density = 15
    gaussian_size = 25
    step_size = int(grid_size / density)

    # Get Gaussian Centers
    gaussian_centres = []
    for x in range(0, grid_size, step_size):
        for y in range(0, grid_size, step_size):
            template = np.zeros((image_height, image_width))
            template[y, x+x_offset] = 1
            template = np.ndarray.reshape(template, (image_height*image_width))
            current_index = np.nonzero(template)
            if current_index in indicies:
                gaussian_centres.append([y, x + x_offset])

    gaussian_array = []
    factors = []
    for centre in gaussian_centres:
        square_factor = makeGaussian(grid_size, fwhm=gaussian_size, center=centre)
        square_factor = np.where(square_factor > 0.01, square_factor, 0)
        template = np.zeros((image_height, image_width))
        template[:, 4:-4] = square_factor
        template = np.ndarray.reshape(template, (image_height * image_width))
        factor = template[indicies]
        factors.append(factor)
        gaussian_array.append(square_factor)

    gaussian_array = np.array(gaussian_array)
    gaussian_array = np.mean(gaussian_array, axis=0)

    initial_factors = np.array(factors)
    initial_factors = np.transpose(initial_factors)

    initial_factors = tf.convert_to_tensor(initial_factors, dtype=tf.float32)
    return initial_factors


def view_factors(base_directory, model, save_directory, iteration):

    # Load Factors
    factors = model.neuron_weights
    number_of_factors = np.shape(factors)[1]

    # Load Mask Details
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    # Create Figure
    figure_1 = plt.figure(figsize=(80, 60))

    columns = 10
    rows = 10
    grid_spec_1 = gridspec.GridSpec(ncols=columns, nrows=rows, figure=figure_1)
    #grid_spec_1.tight_layout(figure_1)

    factor_weights = np.sum(factors, axis=0)

    factor_weights_list_unsorted = np.copy(factor_weights)
    factor_weights_list_sorted = np.copy(factor_weights)

    factor_weights_list_unsorted = list(factor_weights_list_unsorted)
    factor_weights_list_sorted = list(factor_weights_list_sorted)

    factor_weights_list_sorted.sort(reverse=True)

    indicies_sorted_by_weight = []
    for weight in factor_weights_list_sorted:
        indicies_sorted_by_weight.append(factor_weights_list_unsorted.index(weight))

    factor_index = 0
    for column_index in range(columns):
        for row_index in range(rows):
            selected_factor = indicies_sorted_by_weight[factor_index]
            #print("Selected Factor", selected_factor)
            #selected_factor = factor_index

            factor = factors[:, selected_factor]

            factor_image = Widefield_General_Functions.create_image_from_data(factor, indicies, image_height, image_width)

            axis = figure_1.add_subplot(grid_spec_1[row_index, column_index])
            #vmax=np.percentile(factor_image, 99)
            axis.imshow(factor_image, cmap='plasma', vmin=0)
            axis.axis('off')
            factor_index += 1

    plt.savefig(os.path.join(save_directory, str(iteration).zfill(4) + ".png"))
    plt.close()


def compute_cosine_simmilarity(a, b):
    # x shape is n_a * dim
    # y shape is n_b * dim
    # results shape is n_a * n_b

    normalize_a = tf.nn.l2_normalize(a,1)
    normalize_b = tf.nn.l2_normalize(b,1)
    simmilarity = tf.matmul(normalize_a, normalize_b, transpose_b=True)
    mean_simmilarity = tf.reduce_mean(simmilarity)

    return mean_simmilarity




def get_image_center_of_mass(weight_vector, y_coords_vector, x_coords_vector):

    # Weight Coordinates
    y_coords_vector = tf.math.multiply(y_coords_vector, weight_vector)
    x_coords_vector = tf.math.multiply(x_coords_vector, weight_vector)

    # Get Mean Coords
    mean_y_pos = tf.math.reduce_sum(y_coords_vector)
    mean_x_pos = tf.math.reduce_sum(x_coords_vector)

    # Normalise Back To 1
    weight_sum = tf.math.reduce_sum(weight_vector)
    mean_y_pos = tf.math.divide_no_nan(mean_y_pos, weight_sum)
    mean_x_pos = tf.math.divide_no_nan(mean_x_pos, weight_sum)

    return mean_y_pos, mean_x_pos


def get_penalty_vector(y_center, x_center, radius, y_coordinate_vector, x_coordinate_vector):

    ones_matrix = tf.ones(y_coordinate_vector.shape)
    zeros_matrix = tf.zeros(y_coordinate_vector.shape)
    penality_grid = tf.where(tf.math.sqrt(tf.math.square(y_coordinate_vector - y_center) + tf.math.square(x_coordinate_vector - x_center)) < radius, zeros_matrix, ones_matrix)
    return penality_grid


def view_factor(factor, indicies):
    factor = np.array(factor)
    template = np.zeros((600 * 608))
    template[indicies] = factor
    template = np.ndarray.reshape(template, (600, 608))
    plt.imshow(template)
    plt.savefig("/media/matthew/29D46574463D2856/Orthogonal_NMF/loss_functions/001.png")
    plt.close()

    return factor



def get_tensor_sparsity(tensor):

    number_of_factors = tensor.shape[0]
    number_of_pixels = tensor.shape[1]

    mean_sparsity = 0
    for factor_index in range(number_of_factors):
        non_zero_count = tf.math.count_nonzero(tensor[factor_index])

        # Get 100 Free Pixwla
        #non_zero_count = non_zero_count - 500
        #non_zero_count = tf.clip_by_value(non_zero_count, clip_value_min=0, clip_value_max=number_of_pixels)

        sparsity = non_zero_count / number_of_pixels
        mean_sparsity += (sparsity / number_of_factors)

    mean_sparsity = tf.cast(mean_sparsity, tf.float32)

    return mean_sparsity


def get_locality_loss(tensor, x_coordinate_vector, y_coordinate_vector, radius, indicies):

    # Get Number of Factors
    number_of_factors = tensor.shape[0]

    locality_penality = 0
    for factor_index in range(number_of_factors):

        # Select Factor
        factor = tensor[factor_index]

        # Get Factor Centre of Mass
        y_center, x_center = get_image_center_of_mass(factor, y_coordinate_vector, x_coordinate_vector)

        # Get Penalty Vector
        penalty_vector = get_penalty_vector(y_center, x_center, radius, y_coordinate_vector, x_coordinate_vector)

        # Get Penalty
        factor_penalty = tf.math.multiply(factor, penalty_vector)

        #cpf = tf.numpy_function(view_factor, [factor_penalty, indicies], tf.float32)

        factor_penalty = tf.math.reduce_mean(factor_penalty)
        locality_penality += factor_penalty / number_of_factors

    return locality_penality



class tensor_decomposition_model(keras.Model):


    def __init__(self,  number_of_timepoints, number_of_factors, number_of_neurons, initial_weights, image_height, image_width, indicies, y_coordinate_vector, x_coordinate_vector, **kwargs):
        super(tensor_decomposition_model, self).__init__(**kwargs)

        # Setup Variables
        self.number_of_timepoints   = number_of_timepoints
        self.number_of_factors      = number_of_factors
        self.number_of_neurons      = number_of_neurons
        self.image_height = image_height
        self.image_width = image_width
        self.indicies = indicies
        self.x_coordinate_vector = x_coordinate_vector
        self.y_coordinate_vector = y_coordinate_vector




        # Create Weights
        initial_time_weights = tf.zeros([self.number_of_factors, self.number_of_timepoints])
        self.neuron_weights = tf.Variable(shape=(self.number_of_neurons, self.number_of_factors),   initial_value=initial_weights, trainable=True, constraint=tf.keras.constraints.NonNeg())
        self.time_weights = tf.Variable(shape=(self.number_of_factors, self.number_of_timepoints),  initial_value=initial_time_weights, trainable=True,  constraint=tf.keras.constraints.NonNeg())

        # Setup Loss Tracking
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")


    def reconstruct_matrix(self):

        # Create Empty Matrix To Hold Output
        reconstructed_matrix = tf.matmul(self.neuron_weights, self.time_weights)

        # Return Reconstruced Matrix
        return reconstructed_matrix

    @property
    def metrics(self):
        return [self.reconstruction_loss_tracker]


    def call(self, data):
        reconstructed_matrix = self.reconstruct_matrix()
        return reconstructed_matrix

    def train_step(self, data):

        data = data[0]

        with tf.GradientTape() as tape:

            # Clip Weights To Be Between 0 and 1
            self.neuron_weights.assign(tf.clip_by_value(self.neuron_weights, clip_value_min=0, clip_value_max=1))

            # Reconstruct Matrix
            reconstruction = self.reconstruct_matrix()

            # Get Reconstruction Error
            reconstruction_error = tf.subtract(data, reconstruction)
            reconstruction_error = tf.abs(reconstruction_error)
            reconstruction_error = tf.reduce_mean(reconstruction_error)

            # Add Overlap Loss
            factors = tf.transpose(self.neuron_weights)
            overlap_error = compute_cosine_simmilarity(factors, factors)

            # Add Sparisty Loss
            sparsity_loss = get_tensor_sparsity(factors)

            # Add Locality Loss
            locality_loss = get_locality_loss(factors, self.x_coordinate_vector, self.y_coordinate_vector, 100, self.indicies)

            # Scale Losses
            total_loss = 0
            total_loss += 1 * reconstruction_error
            total_loss += 1 * overlap_error
            total_loss += 1 * sparsity_loss
            total_loss += 1000000 * locality_loss


        # Get Gradients
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update Loss
        self.reconstruction_loss_tracker.update_state(total_loss)

        return {"reconstruction_loss":  self.reconstruction_loss_tracker.result() }




def create_model(base_directory, number_of_factors, number_of_neurons, sample_size, initial_factors):

    # Load Mask Details
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    # Create Coordinate Vectors For Spatial Loss Functions
    y_coordinate_vector, x_coordinate_vector = create_coordinate_vectors(indicies, image_height, image_width)

    """
    for x in range(100):
        tempalte = np.zeros((image_height*image_width))
        tempalte[indicies] = initial_factors[:, x]
        tempalte = np.ndarray.reshape(tempalte, (image_height, image_width))
        plt.imshow(tempalte)
        plt.show()
    """

    # Create Model
    model = tensor_decomposition_model(sample_size, number_of_factors, number_of_neurons, initial_factors, image_height, image_width, indicies, y_coordinate_vector, x_coordinate_vector)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.1))

    """
    # Load Delta F Sample
    number_of_timepoints = np.shape(delta_f_matrix)[0]
    indicies_list = list(range(0, number_of_timepoints))
    sample_indicies = np.random.choice(indicies_list, size=sample_size, replace=False)
    sample_indicies = list(sample_indicies)
    sample_indicies.sort()
    delta_f_sample = delta_f_matrix[sample_indicies, :]
    delta_f_sample = np.transpose(delta_f_sample)
    delta_f_array = [delta_f_sample]
    delta_f_array = np.array(delta_f_array).astype('float32')
    delta_f_array = tf.convert_to_tensor(delta_f_array)
    print("Delta F Sample", delta_f_array.shape)

    model(delta_f_array)
    """
    return model







def create_dataset(session_list):

    onset_file_meta_list = [
        ["visual_context_stable_vis_1_onsets.npy"],
        ["visual_context_stable_vis_2_onsets.npy"],
        ["odour_context_stable_vis_1_onsets.npy"],
        ["odour_context_stable_vis_2_onsets.npy"]
    ]
    start_window = -10
    stop_window = 70
    tensor_name = "NMF_Segmentation"

    for base_directory in session_list:

        # Get Average Response For Each Condition
        mean_activity_tensor_list = []
        for onset_file_list in onset_file_meta_list:
            print("Creating Average Tensor For Session: ", base_directory)
            activity_tensor = Create_Activity_Tensor.create_activity_tensor(base_directory, onset_file_list, start_window, stop_window, tensor_name, spatial_smmoothing=True, save_tensor=False)
            activity_tensor = np.nan_to_num(activity_tensor)
            mean_tensor = np.mean(activity_tensor, axis=0)
            mean_activity_tensor_list.append(mean_tensor)

        # Concatenate Mean Responses
        session_concatenated_tensor = np.concatenate(mean_activity_tensor_list)

        # Save Concatenated Tensor
        save_directory = os.path.join(base_directory, "Activity_Tensors", tensor_name + "_Activity_Tensor.npy")
        np.save(save_directory, session_concatenated_tensor)



def train_model(session_list, model_save_directory, plot_save_directory):

    # Seed Initial Factors
    initial_factors = seed_initial_regions_grid(session_list[0])
    print("Initial Factors Shape", np.shape(initial_factors))

    # Load Data
    tensor_list = []
    for base_directory in session_list:
        nmf_tensor = np.load( os.path.join(base_directory, "Activity_Tensors", "NMF_Segmentation_Activity_Tensor.npy"))
        tensor_list.append(nmf_tensor)

    delta_f_array = np.concatenate(tensor_list)
    delta_f_sample = np.transpose(delta_f_array)
    print("Delta F Array Shape", np.shape(delta_f_array))
    delta_f_array = [delta_f_sample]
    delta_f_array = np.array(delta_f_array).astype('float32')
    delta_f_array = tf.convert_to_tensor(delta_f_array)


    # Create Model
    sample_size = np.shape(delta_f_array)[0]
    number_of_pixels = np.shape(delta_f_array)[1]
    number_of_factors = np.shape(initial_factors)[1]

    print("Sample Size", sample_size)
    print("Number of pixels", number_of_pixels)
    print("Number of factors", number_of_factors)
    model = create_model(session_list[0], number_of_factors, number_of_pixels, sample_size, initial_factors)
    model(delta_f_array)

    #model.load_weights(model_save_directory)

    # Train Model
    iteration = 0

    # View Initial Factors Factors
    view_factors(session_list[0], model, plot_save_directory, iteration)

    for x in range(100):
        print("Iteration: ", iteration)

        # Increment Iteration
        iteration += 1

        # Fit Model
        model.fit([delta_f_array], epochs=200, batch_size=1)

        # View Factors
        view_factors(session_list[0], model, plot_save_directory, iteration)

        # Save Model
        model.save(model_save_directory)



os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

controls = [
            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_02_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_08_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_23_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_31_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_15_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_10_29_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_11_05_Transition_Imaging"
            ]



# Get Directories
model_base_directory = "/media/matthew/29D46574463D2856/Orthogonal_NMF_Trial_Average"
model_save_directory = os.path.join(model_base_directory, "Model")
plot_save_directory  = os.path.join(model_base_directory, "Plots")

Widefield_General_Functions.check_directory(model_base_directory)
Widefield_General_Functions.check_directory(model_save_directory)
Widefield_General_Functions.check_directory(plot_save_directory)


# Create Dataset
#create_dataset(controls)

train_model(controls, model_save_directory, plot_save_directory)