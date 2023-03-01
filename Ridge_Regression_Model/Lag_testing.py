import numpy as np
import matplotlib.pyplot as plt

def create_lagged_matrix(matrix, n_lags=14):
    """
    :param matrix: Matrix of shape (n_samples, n_dimensionns)
    :param n_lags: Number Of steps to include lagged versions of the matrix
    :return: Matrix with duplicated shifted version of origional matrix with shape (n_samples, n_dimensions * n_lags)
    """

    lagged_combined_matrix = []
    lagged_combined_matrix.append(matrix)

    for lag_index in range(1, n_lags):

        original_matrix = np.copy(matrix)
        shifted_matrix = np.roll(a=original_matrix, axis=0, shift=lag_index)
        shifted_matrix[0:lag_index] = 0

        figure_1 = plt.figure()
        original_axis = figure_1.add_subplot(2,1,1)
        lagged_axis = figure_1.add_subplot(2,1,2)

        plt.title("shifted matrix")
        original_axis.imshow(np.transpose(original_matrix))
        lagged_axis.imshow(np.transpose(shifted_matrix))
        plt.title(str(lag_index))
        plt.show()


        lagged_combined_matrix.append(shifted_matrix)

    lagged_combined_matrix = np.hstack(lagged_combined_matrix)

    return lagged_combined_matrix



values = np.sin(np.linspace(start=0, stop=2*np.pi, num=20))
matrix = np.ones((5, 20))
matrix = np.multiply(matrix, values)
matrix = np.transpose(matrix)

print("Matrix shape", np.shape(matrix))

plt.imshow(np.transpose(matrix))
plt.show()


lagged_matrix = create_lagged_matrix(matrix)
plt.imshow(np.transpose(lagged_matrix))
plt.show()