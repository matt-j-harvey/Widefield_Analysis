import numpy as np

class ridge_model():

    def __init__(self, Nstim, Nt, Nbehav, stimulus_ridge_penalty, behaviour_ridge_penalty):

        # inputs:
        # Nstim - Number of stimuli
        # Nt - Number of timepoints in each Trial
        # stimulus_ridge_penalty - regularisation coefficient for stimulus inputs
        # behaviour_ridge_penalty - regularisation coefficient for behavioural weights

        #  Save Initialisation Variables
        self.Nstim = Nstim
        self.Nt = Nt
        self.stimulus_ridge_penalty = stimulus_ridge_penalty
        self.behaviour_ridge_penalty = behaviour_ridge_penalty
        self.parameters = None

        # Create Regularisation Matrix (Tikhonov matrix)
        self.Tikhonov = np.zeros([Nstim * Nt + Nbehav, Nstim * Nt + Nbehav])
        self.Tikhonov[0:(Nstim * Nt), 0:(Nstim * Nt)] = np.sqrt(stimulus_ridge_penalty) * np.eye(Nstim * Nt)
        self.Tikhonov[(Nstim * Nt):, (Nstim * Nt):] = np.sqrt(behaviour_ridge_penalty) * np.eye(Nbehav)


    def fit(self, design_matrix, delta_f_matrix):

        ## Perform least squares fit with L2 penalty
        self.MVAR_parameters = np.linalg.solve(design_matrix.T @ design_matrix + self.Tikhonov.T @ self.Tikhonov, design_matrix.T @ delta_f_matrix.T)  # Tikhonov regularisation
        self.MVAR_parameters = self.MVAR_parameters.T


    def predict(self, design_matrix):
        self.prediction = self.MVAR_parameters @ design_matrix.T
        return self.prediction

