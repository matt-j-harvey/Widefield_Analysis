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

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions



def seed_initial_regions(base_directory):

    #pixel_assignments_image = np.load(os.path.join(base_directory, "Pixel_Assignmnets_Image.npy"))
    #plt.imshow(pixel_assignments_image)
    #plt.show()

    # Load Pixel Assignments
    pixel_assignments = np.load(os.path.join(base_directory, "Pixel_Assignmnets.npy"))
    number_of_regions = np.max(pixel_assignments)

    # Load Mask Details
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    initial_factors = []

    # Factors Per Region
    factors_per_region = 3
    for region in range(2, number_of_regions):
        factor_vector = np.where(pixel_assignments == region, 1, 0)
        for x in range(factors_per_region):
            initial_factors.append(factor_vector)

    initial_factors = np.array(initial_factors)
    initial_factors = np.transpose(initial_factors)
    print("Initial Factors Shape", np.shape(initial_factors))
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

    columns = 15
    rows = 10
    grid_spec_1 = gridspec.GridSpec(ncols=columns, nrows=rows, figure=figure_1)
    grid_spec_1.tight_layout(figure_1)

    factor_index = 0
    for column_index in range(columns):
        for row_index in range(rows):
            factor = factors[:, factor_index]
            factor_image = Widefield_General_Functions.create_image_from_data(factor, indicies, image_height, image_width)

            axis = figure_1.add_subplot(grid_spec_1[row_index, column_index])
            axis.imshow(factor_image, cmap='jet', vmin=0, max=1)
            axis.axis('off')
            factor_index += 1

    plt.show()
    #plt.savefig(os.path.join(save_directory, str(iteration).zfill(4) + ".png"))
    #plt.close()






    print("Factors", np.shape(factors))





def get_tensor_overlap(tensor):

    number_of_factors = tensor.shape[0]
    #overlap_matrix = np.zeros([number_of_factors, number_of_factors])

    overlap_matrix = []
    matrix_size = number_of_factors * number_of_factors

    total_overlap = 0
    for factor_1_index in range(number_of_factors):
        factor_1_loadings = tensor[factor_1_index]

        for factor_2_index in range(number_of_factors):
            factor_2_loadings = tensor[factor_2_index]

            if factor_1_index != factor_2_index:
                factor_product = tf.math.multiply(factor_1_loadings, factor_2_loadings)
                product_mean = tf.math.reduce_mean(factor_product)
                total_overlap += product_mean


    mean_overlap = total_overlap / matrix_size

    return mean_overlap



class tensor_decomposition_model(keras.Model):


    def __init__(self,  number_of_timepoints, number_of_factors, number_of_neurons, **kwargs):
        super(tensor_decomposition_model, self).__init__(**kwargs)

        # Setup Variables
        self.number_of_timepoints   = number_of_timepoints
        self.number_of_factors      = number_of_factors
        self.number_of_neurons      = number_of_neurons

        # Create Weights
        self.neuron_weights = self.add_weight(shape=(self.number_of_neurons, self.number_of_factors),   initializer='normal', trainable=True, name='neuron_weights', constraint=tf.keras.constraints.NonNeg())
        self.time_weights = self.add_weight(shape=(self.number_of_factors, self.number_of_timepoints),  initializer='normal', trainable=True, name='time_weights', constraint=tf.keras.constraints.NonNeg())

        # Setup Loss Tracking
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")




    def reconstruct_matrix(self):

        # Create Empty Matrix To Hold Output
        reconstructed_matrix = tf.matmul(self.neuron_weights, self.time_weights)
        print("reconstructed matrix", reconstructed_matrix.shape)

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
        print("Data shape", np.shape(data))

        with tf.GradientTape() as tape:

            total_loss = 0

            # Reconstruct Matrix
            reconstruction = self.reconstruct_matrix()

            # Create loss Functions
            reconstruction_error = tf.subtract(data, reconstruction)
            reconstruction_error = tf.abs(reconstruction_error)
            reconstruction_error = tf.reduce_mean(reconstruction_error)

            factors = tf.transpose(self.neuron_weights)

            #overlap_error = get_tensor_overlap(factors)
            #overlap_error = overlap_error * 10
            #total_loss += overlap_error

            total_loss += reconstruction_error




        # Get Gradients
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update Loss
        self.reconstruction_loss_tracker.update_state(total_loss)

        return {"reconstruction_loss":  self.reconstruction_loss_tracker.result() }


def smooth_data(base_directory, data_sample):

    # Load Mask Details
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    smoothed_sample = []
    for frame in data_sample:
        template = np.zeros((image_height * image_width))
        template[indicies] = frame
        template = np.ndarray.reshape(template, (image_height, image_width))
        template = scipy.ndimage.gaussian_filter(template, sigma=1)
        template = np.ndarray.reshape(template, image_height * image_width)
        template_data = template[indicies]
        smoothed_sample.append(template_data)

    smoothed_sample = np.array(smoothed_sample)
    return smoothed_sample



def create_model(base_directory, number_of_factors, sample_size):

    # Load Delta F Matrix
    delta_f_matrix_filepath = os.path.join(base_directory, "Delta_F.h5")
    delta_f_matrix_container = tables.open_file(delta_f_matrix_filepath, mode='r')
    delta_f_matrix = delta_f_matrix_container.root['Data']
    number_of_neurons = np.shape(delta_f_matrix)[1]

    # Create Model
    model = tensor_decomposition_model(sample_size, number_of_factors, number_of_neurons)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1))

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

    return model

def train_model(base_directory, sample_size):

    # Load Delta F Matrix
    delta_f_matrix_filepath = os.path.join(base_directory, "Delta_F.h5")
    delta_f_matrix_container = tables.open_file(delta_f_matrix_filepath, mode='r')
    delta_f_matrix = delta_f_matrix_container.root['Data']
    print("Delta F Matrix", np.shape(delta_f_matrix))

    # Load Delta F Sample
    number_of_timepoints = np.shape(delta_f_matrix)[0]
    indicies_list = list(range(0, number_of_timepoints))
    sample_indicies = np.random.choice(indicies_list, size=sample_size, replace=False)
    sample_indicies = list(sample_indicies)
    sample_indicies.sort()
    delta_f_sample = delta_f_matrix[sample_indicies, :]

    delta_f_sample = np.nan_to_num(delta_f_sample)
    delta_f_sample = smooth_data(base_directory, delta_f_sample)

    delta_f_sample = np.transpose(delta_f_sample)
    delta_f_array = [delta_f_sample]
    delta_f_array = np.array(delta_f_array).astype('float32')
    delta_f_array = tf.convert_to_tensor(delta_f_array)
    print("Delta F Sample", delta_f_array.shape)

    # Fit Model
    model.fit([delta_f_array], epochs=100, batch_size=1)

    # Return Model
    return model

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



session_list = ["/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",
                "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging",
                "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NRXN78.1D/2020_11_29_Switching_Imaging",
                "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NRXN78.1A/2020_12_09_Switching_Imaging",
                "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging",
                "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging",
                "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging",
                "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging"]


model_save_directory = "/media/matthew/29D46574463D2856/Orthogonal_NMF/Model"
plot_save_directory = "/media/matthew/29D46574463D2856/Orthogonal_NMF/Plots"


initial_factors = seed_initial_regions(session_list[0])
number_of_factors = np.shape(initial_factors)[1]

# Create Model
#number_of_factors = 36
sample_size = 1000
model = create_model(session_list[0], number_of_factors, sample_size)
print(model.layers[0].get_weights())

#model.layersp0neuron_weights.set_weights(initial_factors)

iteration = 0
for x in range(100):
    for base_directory in session_list:
        print("Iteration: ", iteration)

        # Train Model
        model = train_model(base_directory, sample_size)

        # View Factors
        view_factors(base_directory, model, plot_save_directory, iteration)

        # Save Model
        model.save(model_save_directory)

        # Increment Iteration
        iteration += 1

