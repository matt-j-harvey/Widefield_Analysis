import numpy as np
import matplotlib.pyplot as plt


def get_lagged_matrix(matrix, n_lags=3):

    """
    :param matrix: Matrix of shape (n_dimensionns, n_samples)
    :param n_lags: Number Of steps to include lagged versions of the matrix
    :return: Matrix with duplicated shifted version of origional matrix with shape (n_dimensions * n_lages, n_samples)
    """

    lagged_combined_matrix = []
    for lag_index in range(n_lags):
        lagged_matrix = np.copy(matrix)
        lagged_matrix = np.roll(a=lagged_matrix, axis=1, shift=lag_index)
        lagged_matrix[:, 0:lag_index] = 0
        lagged_combined_matrix.append(lagged_matrix)
        plt.title("Lag" + str(lag_index))
        plt.imshow(lagged_matrix)
        plt.show()

    lagged_combined_matrix = np.vstack(lagged_combined_matrix)
    return lagged_combined_matrix



matrix = np.array([
    list(range(0, 10)),
    list(range(10, 20)),
    list(range(20,30)),
])

plt.imshow(matrix)
plt.show()

get_lagged_matrix(matrix)