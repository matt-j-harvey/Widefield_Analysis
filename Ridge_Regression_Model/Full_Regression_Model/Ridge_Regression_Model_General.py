import os
number_of_threads = 1
os.environ["OMP_NUM_THREADS"] = str(number_of_threads) # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = str(number_of_threads) # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = str(number_of_threads) # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(number_of_threads) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(number_of_threads) # export NUMEXPR_NUM_THREADS=1

#from sklearnex import patch_sklearn
#patch_sklearn()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import tables
from tqdm import tqdm
import joblib
from datetime import datetime
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold

from Widefield_Utils import widefield_utils
from Ridge_Regression_Model import Get_Cross_Validated_Ridge_Penalties
from Files import Session_List


def fit_ridge_model(delta_f_matrix, design_matrix, save_directory, chunk_size=1000):

    # Get Chunk Structure
    number_of_pixels = np.shape(delta_f_matrix)[1]
    number_of_chunks, chunk_sizes, chunk_start_list, chunk_stop_list = widefield_utils.get_chunk_structure(chunk_size, number_of_pixels)

    # Fit Model For Each Chunk
    regression_intercepts_list = np.zeros(number_of_pixels)
    number_of_regressors = np.shape(design_matrix)[1]
    regression_coefs_list = np.zeros((number_of_pixels, number_of_regressors))
    ridge_penalty_list = np.zeros(number_of_pixels)

    # Fit Each Chunk
    for chunk_index in tqdm(range(number_of_chunks), position=1, desc="Chunk: ", leave=False):

        # Get Chunk Data
        chunk_start = chunk_start_list[chunk_index]
        chunk_stop = chunk_stop_list[chunk_index]
        chunk_data = delta_f_matrix[:, chunk_start:chunk_stop]
        chunk_data = np.nan_to_num(chunk_data)

        # Get Cross Validated Ridge Penalties
        ridge_penalties = Get_Cross_Validated_Ridge_Penalties.get_cross_validated_ridge_penalties(design_matrix, chunk_data)

        # Create Model
        model = Ridge(solver='auto', alpha=ridge_penalties)

        # Fit Model
        model.fit(y=chunk_data, X=design_matrix)

        # Get Coefs
        model_coefs = model.coef_
        model_intercepts = model.intercept_

        # Save These
        regression_coefs_list[chunk_start:chunk_stop] = model_coefs
        regression_intercepts_list[chunk_start:chunk_stop] = model_intercepts
        ridge_penalty_list[chunk_start:chunk_stop] = ridge_penalties

    # Save These

    # Create Regression Dictionary
    regression_dict = {
        "Coefs": regression_coefs_list,
        "Intercepts": regression_intercepts_list,
        "Ridge_Penalties": ridge_penalty_list,
    }

    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    np.save(os.path.join(save_directory, "Regression_Dictionary_Simple.npy"), regression_dict)



