import tensorflow as tf
import numpy as np



def get_tensor_overlap(tensor):

    number_of_factors = tensor.shape[0]
    overlap_matrix = np.zeros([number_of_factors, number_of_factors])

    for factor_1_index in range(number_of_factors):
        factor_1_loadings = tensor[factor_1_index]

        for factor_2_index in range(number_of_factors):
            factor_2_loadings = tensor[factor_2_index]

            if factor_1_index != factor_2_index:
                factor_product = tf.math.multiply(factor_1_loadings, factor_2_loadings)
                factor_sum = tf.math.reduce_mean(factor_product)
                overlap_matrix[factor_1_index][factor_2_index] = factor_sum

    overlap_matrix = tf.convert_to_tensor(overlap_matrix)
    overlap_matrix = tf.abs(overlap_matrix)
    mean_overlap = tf.reduce_mean(overlap_matrix)

    return mean_overlap







x = tf.random.uniform([10, 100])


mean_overlap = get_tensor_overlap(x)
print(mean_overlap)