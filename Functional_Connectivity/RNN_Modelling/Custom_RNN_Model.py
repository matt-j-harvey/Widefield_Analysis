import torch


class custom_rnn(torch.nn.Module):

    def __init__(self, n_inputs, n_neurons, device):
        super(custom_rnn, self).__init__()

        # Initialise Weights
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.input_weights = torch.nn.Linear(n_inputs, n_neurons, bias=True).float()
        self.recurrent_weights = torch.nn.Linear(n_neurons, n_neurons, bias=True).float()

        # Initialise Hidden State
        self.hidden_state = torch.zeros(1, n_neurons, dtype=torch.float, device=device)


    def forward(self, external_input):

        # Get External Input
        input_contribution = self.input_weights(external_input)

        # Get Recurrent Input
        recurrent_contribution = self.recurrent_weights(self.hidden_state)

        # Sum These and Biases
        new_activity = input_contribution + recurrent_contribution

        # Put Through Activation Function
        new_activity = torch.sigmoid(new_activity)

        self.hidden_state = new_activity

        return new_activity