import tensorflow as tf
import numpy as np


def myfunc(matrix):
    matrix = np.array(matrix)
    correlation_matrix = np.corrcoef(matrix)
    return correlation_matrix

matrix = tf.random.uniform((10, 100))


# create graph
Y = tf.py_function(func=myfunc,inp=[matrix],Tout=[tf.float32], name='myfunc')
print(Y)
