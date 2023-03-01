import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
import mat73
from scipy.io import loadmat
import matplotlib.ticker as mtick


def get_sessions(data_directory):

    session_name_list = []

    file_list = os.listdir(data_directory)
    for file_name in file_list:
        split_file = file_name.split("_")
        session_name = split_file[0:6]
        session_name = '_'.join(session_name)
        if session_name not in session_name_list:
            session_name_list.append(session_name)

    return session_name_list



def load_session_data(session_name, data_directory):

    session_data_file = session_name + "_data"
    session_timings_file = session_name + "_timings.mat"

    # Load Session Timings
    timings_file = loadmat(os.path.join(data_directory, session_timings_file))
    data_arrays = timings_file['timings'][0][0]

    # Load Session Delta F Matrix
    activity_matrix = np.load(os.path.join(data_directory, session_data_file), allow_pickle=True)

    # Package into Dictionary
    session_data = {
        'session_name':session_name,
        'delta_f':activity_matrix,
        'relVis1Onsets':    list(data_arrays[0][0]),
        'relVis1Offsets':   list(data_arrays[1][0]),
        'relVis2Onsets':    list(data_arrays[2][0]),
        'relVis2Offsets':   list(data_arrays[3][0]),
        'irrelVis1Onsets':  list(data_arrays[4][0]),
        'irrelVis1Offsets': list(data_arrays[5][0]),
        'irrelVis2Onsets':  list(data_arrays[6][0]),
        'irrelVis2Offsets': list(data_arrays[7][0]),
        'odr1Onsets':       list(data_arrays[8][0]),
        'odr1Offsets':      list(data_arrays[9][0]),
        'odr2Onsets':       list(data_arrays[10][0]),
        'odr2Offsets':      list(data_arrays[11][0]),
    }

    return session_data


def load_decoding_data(data_directory):

    # Get Session Names
    session_list = get_sessions(data_directory)

    # Load Session Data
    session_data_list = []
    for session in session_list:
        session_data = load_session_data(session, data_directory)
        session_data_list.append(session_data)

    return session_data_list

