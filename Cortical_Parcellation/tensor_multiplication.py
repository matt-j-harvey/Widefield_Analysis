import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


def correlation(x, y):
    mx = tf.math.reduce_mean(x)
    my = tf.math.reduce_mean(y)
    xm, ym = x - mx, y - my
    r_num = tf.math.reduce_mean(tf.multiply(xm, ym))
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    correlation_matrix = r_num / r_den
    print(correlation_matrix)
    return correlation_matrix


def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = tf.reduce_mean(x, axis=1, keepdims=True)
    my = tf.reduce_mean(y, axis=1, keepdims=True)
    xm, ym = x - mx, y - my
    t1_norm = tf.nn.l2_normalize(xm, axis=1)
    t2_norm = tf.nn.l2_normalize(ym, axis=1)
    cosine = tf.losses.categorical_crossentropy(t1_norm, t2_norm, axis=1, from_logits=True)
    return cosine




tf.py_function(myfunc,[X],[tf.float32],name='myfunc')



matrix = tf.random.uniform((10, 100))
#matrix = tf.transpose(matrix)
factor_correlation = pearson_r(matrix, matrix)
print(factor_correlation.shape)

"""
numberof_neurons = 10
number_of_factors = 3
number_of_timepoints = 20


raw_matrix = tf.random.uniform([numberof_neurons, number_of_timepoints])
factor_matrix = tf.random.uniform([numberof_neurons, number_of_factors])
time_matrix = tf.random.uniform([number_of_factors, number_of_timepoints])

reconstructued_matrix = tf.matmul(factor_matrix, time_matrix)

print(reconstructued_matrix.shape)

"""
