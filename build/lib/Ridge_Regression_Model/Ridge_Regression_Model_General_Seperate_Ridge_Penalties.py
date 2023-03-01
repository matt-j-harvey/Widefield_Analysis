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
from Ridge_Regression_Model import Get_Cross_Validated_Ridge_Penalties_Seperate, Ridge_Model_Seperate_Penalties_Class
from Files import Session_List


def fit_ridge_model(delta_f_matrix, design_matrix, onset_file_list, start_window, stop_window, save_directory, chunk_size=1000):

    # Get Stim Details
    n_stim = len(onset_file_list)
    n_timepoints = stop_window - start_window
    print("N Stim", n_stim)
    print("n timepoints", n_timepoints)

    number_of_regressors = np.shape(design_matrix)[1]
    print("")
    Nbehav = number_of_regressors - (n_stim * n_timepoints)
    print("N Behav", Nbehav)

    # Zero Delta F Matrix
    print("Delta F Matrixx Shape", np.shape(delta_f_matrix))
    delta_f_mean = np.mean(delta_f_matrix, axis=0)
    delta_f_matrix = np.subtract(delta_f_matrix, delta_f_mean)
    print("Delta F Mean", np.shape(delta_f_mean))

    # Get Cross Validated Ridge Penalties
    ridge_penalties = Get_Cross_Validated_Ridge_Penalties_Seperate.get_cross_validated_ridge_penalties(design_matrix, delta_f_matrix, n_stim, n_timepoints, Nbehav)

    # Create Model
    model = Ridge_Model_Seperate_Penalties_Class.ridge_model(n_stim, n_timepoints, Nbehav, stimulus_ridge_penalty, behaviour_ridge_penalty)

    """
    # Fit Model
    model.fit(y=chunk_data, X=design_matrix)

    # Get Coefs
    model_coefs = model.parameters

    # Save These
    regression_coefs_list[chunk_start:chunk_stop] = model_coefs
    ridge_penalty_list[chunk_start:chunk_stop] = ridge_penalties

    # Save These

    # Create Regression Dictionary
    regression_dict = {
        "Coefs": regression_coefs_list,
        "Ridge_Penalties": ridge_penalty_list,
    }

    np.save(os.path.join(save_directory, "Regression_Dictionary_Seperate_Penalties.npy"), regression_dict)
    """


