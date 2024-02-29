import torch
from torch import nn

class CriticNetL2(nn.Module):
    """A critic network that estimates a dual Q-function."""
    
    def __init__(self, state_size, action_size, hidden_sizes, output_size=1, average_reward_offset=0.3):
        """Creates networks.
        
        Args:
          state_dim: State size.
          action_dim: Action size.
        """
        super(CriticNetL2, self).__init__()
        # Define the layers of the network
        layers = []
        input_size = state_size + action_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size # Output the next state

        # Final layer to output the next state
        self.final_layer = nn.Linear(input_size, output_size)

        # Register the layers as a ModuleList
        self.layers = nn.ModuleList(layers)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
        nn.init.orthogonal_(self.final_layer.weight)
        self.average_reward_offset = average_reward_offset
        # Optimizer
        # self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, x, u):
        """
        Predict next state given current state and action
        :param x: Current state
        :param u: Current action
        :return: Predicted next reward
        """
        xu = torch.cat((x, u), dim=-1)
        for layer in self.layers:
            xu = layer(xu)
        result = self.final_layer(xu)
        # result[:, 0] += self.average_reward_offset
        return result.squeeze(-1)


class CriticNetDD(CriticNetL2):
    def __init__(self, state_size, action_size, hidden_sizes, 
                 num_atoms, Vmin, Vmax, lr=1e-3, weight_decay=1e-4):
        """Initializes the CriticNetDD which outputs a discrete distribution.
        
        Args:
          state_size: Size of the state space.
          action_size: Size of the action space.
          hidden_sizes: A list of integers defining the size of hidden layers.
          num_atoms: Number of discrete atoms in the output distribution.
          Vmin: Minimum value of the support of the distribution.
          Vmax: Maximum value of the support of the distribution.
          lr: Learning rate for the optimizer.
          weight_decay: Weight decay for the optimizer.
        """
        super(CriticNetDD, self).__init__(state_size, action_size, hidden_sizes, lr, weight_decay)
        
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.delta_z = (Vmax - Vmin) / (num_atoms - 1)
        
        # Override the final layer to output the logits for the discrete distribution
        self.final_layer = nn.Linear(hidden_sizes[-1], num_atoms)
        
        # Initialize weights of the final layer
        nn.init.orthogonal_(self.final_layer.weight)
    
    def forward(self, x, u):
        """
        Predicts the action-value distribution given the current state and action.
        
        Args:
          x: Current state.
          u: Current action.
          
        Returns:
          A tensor of shape [batch_size, num_atoms] representing the probabilities
          of each atom in the output distribution.
        """
        xu = torch.cat((x, u), dim=-1)
        for layer in self.layers:
            xu = layer(xu)
        
        logits = self.final_layer(xu)
        probabilities = F.softmax(logits, dim=-1)
        return probabilities
    
    def get_atoms(self):
        """
        Computes the atom values of the distribution.
        
        Returns:
          A tensor of shape [num_atoms] representing the values of each atom.
        """
        atoms = torch.linspace(self.Vmin, self.Vmax, steps=self.num_atoms)
        return atoms
    