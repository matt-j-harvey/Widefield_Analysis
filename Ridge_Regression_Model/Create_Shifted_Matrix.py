import numpy as np
import matplotlib.pyplot as plt

def create_lagged_matrix(matrix, n_lags=3):

    """
    :param matrix: Matrix of shape (n_dimensionns, n_samples)
    :param n_lags: Number Of steps to include lagged versions of the matrix
    :return: Matrix with duplicated shifted version of origional matrix with shape (n_dimensions * n_lages, n_samples)
    """
    print("Getting Lagged Matrix Shape", np.shape(matrix))

    lagged_combined_matrix = []
    for lag_index in range(n_lags):
        lagged_matrix = np.copy(matrix)
        lagged_matrix = np.roll(a=lagged_matrix, axis=1, shift=lag_index)
        lagged_matrix[:, 0:lag_index] = 0
        lagged_combined_matrix.append(lagged_matrix)

    lagged_combined_matrix = np.hstack(lagged_combined_matrix)

    return lagged_combined_matrix


matrix = [
    [0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10],
    [10, 11, 12, 13, 14, 15, 16, 17, 18, 18, 20],
    [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
]

matrix = np.array(matrix)
matrix = np.transpose(matrix)
print("Matrix Shape", np.shape(matrix))

lagged_matrix = create_lagged_matrix(matrix)
print("lagged Matrix shape", np.shape(lagged_matrix))

plt.imshow(lagged_matrix)
plt.show()