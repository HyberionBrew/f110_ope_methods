import torch 
from torch import nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import yaml
import os


# min and max state are only needed here for ensebmle model, should use this for all models
def build_dynamics_model(dynamics_model, min_state=None, max_state=None):    
    if dynamics_model == "DeltaDynamicsModel":
        print("Using DeltaDynamicsModel")
        dynamics_model = DeltaDynamicsModel([256,256,256,256], 1/20, 
                                        lr=1e-3,
                                        weight_decay=1e-4)
    elif dynamics_model == "SimpleDynamicsModel":
        print("Using SimpleDynamicsModel")
        dynamics_model = SimpleDynamicsModel(7, 2, [256,256,256,256],
                                        lr=1e-3,
                                        weight_decay=1e-4)
    elif dynamics_model == "ProbDynamicsModel":
        dynamics_model = ProbDynamicsModel(7,2, [256,256,256,256],
                                        lr=1e-3,
                                        weight_decay=1e-4)
    elif dynamics_model == "ProbsDeltaDynamicsModel":
        dynamics_model = ProbsDeltaDynamicsModel([256,256,256,256], 1/20, 
                                        lr=1e-3,
                                        weight_decay=1e-4,
                                        min_state=min_state,
                                        max_state=max_state)
    elif dynamics_model == "AutoregressiveModel":
        dynamics_model = AutoregressiveModel(7,2, [256,256,256,256], 
                                             lr=1e-3,
                                        weight_decay=1e-4)
    elif dynamics_model == "EnsemblePDDModel":
        dynamics_model = ModelBasedEnsemble("ProbsDeltaDynamicsModel", 5, 
                                            min_state=min_state, 
                                            max_state=max_state)
    
    return dynamics_model


class RewardModel(nn.Module):
    """A class that implements a reward model in PyTorch."""

    def __init__(self, state_dim, hidden_size, min_reward, max_reward, lr=1e-4, weight_decay=1e-5):
        super(RewardModel, self).__init__()
        self.min_reward = min_reward
        self.max_reward = max_reward
        # Define the layers
        # print(state_dim)
        # print(hidden_size)
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 1)

        # Initialize weights using orthogonal initializer
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.fc3.weight)
        nn.init.orthogonal_(self.fc4.weight)
        nn.init.orthogonal_(self.fc5.weight)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, x):
        # Concatenate the state and action tensors along the last dimension
        x_a = x

        # Pass through the layers with ReLU activations
        x_a = F.relu(self.fc1(x_a))
        x_a = F.relu(self.fc2(x_a))
        x_a = F.relu(self.fc3(x_a))
        x_a = F.relu(self.fc4(x_a))

        # Output layer (squeeze the last dimension to match TensorFlow's behavior)
        x_a = self.fc5(x_a).squeeze(-1)
        # Clip the output to be within the min and max reward bounds
        x_a = torch.clamp(x_a, self.min_reward, self.max_reward)
        return x_a

    def update(self, original_states, rewards):
        # Compute the model's predictions
        predicted_rewards = self.forward(original_states)

        # Calculate loss (e.g., Mean Squared Error)
        loss = F.mse_loss(predicted_rewards, rewards)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
    def save(self, path, filename="model_based_torch_checkpoint.pth"):
        torch.save(self.state_dict(), os.path.join(path, filename))
    def load(self, path, filename="model_based_torch_checkpoint.pth"):
        self.load_state_dict(torch.load(os.path.join(path, filename)))

class DoneNetwork(RewardModel):
    def __init__(self, state_dim, hidden_size, lr=1e-4, weight_decay=1e-5):
        super(DoneNetwork, self).__init__(state_dim, hidden_size,0.0,1.0, lr, weight_decay)
        
        # Modify the output layer for binary output
        # Assuming the last layer of the reward model is named 'output_layer'
        self.output_layer = nn.Linear(hidden_size, 1)

        # Using Binary Cross Entropy Loss
        self.loss_function = nn.BCEWithLogitsLoss()

        # Optimizer can remain the same, adjust if necessary
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, x):
        # Assuming 'x' is the input to the network
        # Use the same forward pass as the reward model but adjust the last layer activation
        out = super(DoneNetwork, self).forward(x)
        return torch.sigmoid(out)  # Sigmoid activation for binary output

    def update(self, states, dones):
        # Assuming 'dones' is a tensor of 0s and 1s indicating terminal states

        # Forward pass
        predicted_dones = self(states)

        # Compute loss
        loss = self.loss_function(predicted_dones, dones)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    def save(self, path, filename="model_based_torch_checkpoint.pth"):
        torch.save(self.state_dict(), os.path.join(path, filename))
    def load(self, path, filename="model_based_torch_checkpoint.pth"):
        self.load_state_dict(torch.load(os.path.join(path, filename)))





# input_obs_keys = ['theta_sin', 'theta_cos', 'ang_vels_z', 'linear_vels_x', 'linear_vels_y']
class SimpleDynamicsModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes, lr=1e-4, weight_decay=1e-5):
        super(SimpleDynamicsModel, self).__init__()

        # Define the layers of the network
        layers = []
        input_size = state_size + action_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size # Output the next state

        # Final layer to output the next state
        layers.append(nn.Linear(input_size, state_size))

        # Register the layers as a ModuleList
        self.layers = nn.ModuleList(layers)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.min_state = None
        self.max_state = None
    def set_min_state(self, min_state):
        self.min_state = min_state
    def set_max_state(self, max_state):
        self.max_state = max_state

    def forward(self, x, u):
        """
        Predict next state given current state and action
        :param x: Current state
        :param u: Current action
        :return: Predicted next state
        """
        xu = torch.cat((x, u), dim=-1)
        for layer in self.layers:
            xu = layer(xu)
        next_state = torch.clamp(xu, self.min_state, self.max_state)
        return next_state

    def save(self, path, filename="flexible_dynamics_model_checkpoint.pth"):
        torch.save(self.state_dict(), os.path.join(path, filename))

    def load(self, path, filename="flexible_dynamics_model_checkpoint.pth"):
        self.load_state_dict(torch.load(os.path.join(path, filename)))

    def update(self, states, actions, next_states):
        
        self.optimizer.zero_grad()
        pred_states = self(states, actions)
        loss = F.mse_loss(pred_states, next_states)
        loss.backward()
        self.optimizer.step()
        return loss.item(), 0, 0, 0
    
    def set_device(self, device):
        self.device = device
        self.min_state = self.min_state.to(device)
        self.max_state = self.max_state.to(device)
        self.to(device)

class ProbDynamicsModel(SimpleDynamicsModel):
    def __init__(self, state_size, action_size, hidden_sizes, lr=1e-3, weight_decay=1e-5):
        #print(self.type)
        super(ProbDynamicsModel, self).__init__(state_size, action_size, hidden_sizes, lr, weight_decay)
        # replace the last layer with a layer that outputs the mean and variance
        self.layers[-1] = nn.Linear(hidden_sizes[-1], state_size*2)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.state_size = state_size
    def forward(self, x, u, train=False):
        """
        Predict next state given current state and action
        :param x: Current state
        :param u: Current action
        :return: Predicted next state
        """
        xu = torch.cat((x, u), dim=-1)
        for layer in self.layers:
            xu = layer(xu)
        
        mean = xu[:,0:self.state_size]
        logvar = xu[:,self.state_size:]
        # clamp the mean
        mean = torch.clamp(mean, self.min_state, self.max_state)
        if train:
            return mean, logvar
        else:
            return mean
        
    def update(self, states, action, next_states):
        self.optimizer.zero_grad()
        mean, logvar = self(states, action, train=True)
        cov_matrix = torch.diag_embed(torch.exp(logvar))
        normal_distribution = torch.distributions.MultivariateNormal(mean, cov_matrix)
        log_prob = normal_distribution.log_prob(next_states)
        #print(states.shape)
        #print(mean.shape)
        #print(logvar.shape)
        #print(log_prob.shape)
        #print("---")
        loss = -log_prob.mean() 
        loss.backward()
        self.optimizer.step()
        return loss.item(), 0, 0, 0


output_keys =  ['poses_x', 'poses_y', 'theta_sin', 'theta_cos', 'ang_vels_z', 'linear_vels_x', 'linear_vels_y']
mb_keys = output_keys + ['previous_action_steer', 'previous_action_speed']
class DeltaDynamicsModel(nn.Module):
    def __init__(self,hidden_size, dt, #min_state, max_state,
                 lr=1e-4, weight_decay=1e-5,
                 min_state=None,
                 max_state=None):
        
        super().__init__()
        state_size = len(output_keys)
        action_size = 2
        self.min_state = min_state #min_state
        self.max_state = max_state # max_state
        self.state_size, self.action_size, self.dt = state_size, action_size, dt
        A_size, B_size = state_size * state_size, state_size * action_size
        # hidden layers for A
        self.A_layers = nn.ModuleList()
        self.A_layers.append(self._make_layer(state_size + action_size, hidden_size[0]))
        for i in range(1, len(hidden_size)):
            self.A_layers.append(self._make_layer(hidden_size[i-1], hidden_size[i]))
        self.A_layers.append(self._make_layer(hidden_size[-1], A_size))

        # hidden layers for B
        self.B_layers = nn.ModuleList()
        self.B_layers.append(self._make_layer(state_size + action_size, hidden_size[0]))
        for i in range(1, len(hidden_size)):
            self.B_layers.append(self._make_layer(hidden_size[i-1], hidden_size[i]))
        self.B_layers.append(self._make_layer(hidden_size[-1], B_size))

        self.optimizer_dynamics = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.column_names = output_keys
    def set_min_state(self, min_state):
        self.min_state = min_state
    def set_max_state(self, max_state):
        self.max_state = max_state

    def _make_layer(self, in_dim, out_dim):
        layer = nn.Linear(in_dim, out_dim)
        init.orthogonal_(layer.weight)
        return layer
    
    def set_device(self, device):
        self.device = device
        self.min_state = self.min_state.to(device)
        self.max_state = self.max_state.to(device)
        self.to(device)

    def forward(self, x, u):
        """
        Predict x_{t+1} = f(x_t, u_t)
        :param x: a batch of states
        :param u: a batch of actions
        """
        assert self.column_names is not None
        #in order to make learning easier apply to the u the clipping and scaling
        # u = torch.clip(u, -1, 1) * 0.05
        xu = torch.cat((x, u), -1)
        # set the x and y states to 0 
        x_column = self.column_names.index('poses_x')
        y_column = self.column_names.index('poses_y')
        theta_sin_column = self.column_names.index('theta_sin')
        theta_cos_column = self.column_names.index('theta_cos')
        xu[:,x_column] = 0.0  # Remove dependency in (x,y)
        xu[:,y_column] = 0.0  # Remove dependency in (x,y)
        xu[:,theta_sin_column] = 0.0  # Remove dependency in theta
        xu[:,theta_cos_column] = 0.0  # Remove dependency in theta  # Remove dependency in (x,y)
        for layer in self.A_layers[:-1]:
            xu = F.relu(layer(xu))

        A = self.A_layers[-1](xu)  # Last layer
        A = torch.reshape(A, (x.shape[0], self.state_size, self.state_size))
        
        # Reset and pass through B hidden layers
        xu = torch.cat((x, u), -1)
        xu[:, x_column] = 0.0
        xu[:, y_column] = 0.0
        xu[:,theta_sin_column] = 0.0  
        xu[:,theta_cos_column] = 0.0 
        for layer in self.B_layers[:-1]:  # All but the last layer
            xu = F.relu(layer(xu))
        
        B = self.B_layers[-1](xu)  # Last layer
        B = torch.reshape(B, (x.shape[0], self.state_size, self.action_size))
        
        dx = A @ x.unsqueeze(-1) + B @ u.unsqueeze(-1)
        x = x + dx.squeeze()*self.dt
        # normalize the theta, it should sum to 1
        theta_sin_cos_norm = torch.sqrt(x[:, theta_sin_column]**2 + x[:, theta_cos_column]**2)
        new_mean = x.clone()
        new_mean[:, theta_sin_column] = x[:, theta_sin_column] / theta_sin_cos_norm
        new_mean[:, theta_cos_column] = x[:, theta_cos_column] / theta_sin_cos_norm
        x_new = torch.clamp(new_mean, self.min_state, self.max_state)
        return x_new
    
    def save(self, path, filename="model_based_torch_checkpoint.pth"):
        torch.save(self.state_dict(), os.path.join(path, filename))

        
    def load(self, path, filename="model_based_torch_checkpoint.pth"):
        self.load_state_dict(torch.load(os.path.join(path, filename)))

    def update(self, states, actions, next_states):
        self.optimizer_dynamics.zero_grad()
        pred_states = self(states, actions)
        dyn_loss = F.mse_loss(pred_states, next_states, reduction='none')
        dyn_loss = (dyn_loss).mean()
        dyn_loss.backward()
        self.optimizer_dynamics.step()
        return dyn_loss.item(), 0, 0 , 0
    
class AutoregressiveModel(SimpleDynamicsModel):
    def __init__(self, state_size, action_size, hidden_sizes, lr=1e-4, weight_decay=1e-5):
        super(AutoregressiveModel, self).__init__(state_size, action_size, hidden_sizes, lr, weight_decay)
        layers = []
        input_size = state_size * 3 + action_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size # Output the next state
        # Final layer to output the next state
        layers.append(nn.Linear(input_size,2)) # output has only size one

        # Register the layers as a ModuleList
        self.layers = nn.ModuleList(layers)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.min_state = None
        self.max_state = None
        self.state_size = state_size
    
    def forward(self,states, actions):
        s_next_states = torch.zeros_like(states)
        for i in range(self.state_size):
            # build the input
            # one hot encode i 
            s_one_hot = torch.zeros(states.shape[0], self.state_size)
            # Set the ith position to 1
            s_one_hot[:, i] = 1
            input_states = torch.cat((states, s_next_states, s_one_hot), dim=1)
            s_next_states[:,i] = self.single_pass(input_states, actions, i)

        return s_next_states

    def single_pass(self, augmented_states, actions,i, train = False):
        xu = torch.cat((augmented_states, actions), dim=-1)
        for layer in self.layers:
            xu = layer(xu)
        
        mean = xu[:,0]
        logvar = xu[:,1]
        # clamp the mean
        mean = torch.clamp(mean, self.min_state[i], self.max_state[i])
        if train:
            return mean, logvar
        else:
            return mean 

    def update(self, states, actions, next_states):
        self.optimizer.zero_grad()
        total_loss = 0
        for i in range(self.state_size):
            # build the input
            # one hot encode i 
            s_one_hot = torch.zeros(states.shape[0], self.state_size, device=self.device)
            # Set the ith position to 1
            s_one_hot[:, i] = 1
            s_next_state = next_states.clone()
            s_next_state[:,i:] = 0.0
            target = next_states[:,i]
            #print(s_one_hot)
            #print(s_next_state)
            input_states = torch.cat((states, s_next_state, s_one_hot), dim=1)
            mean, logvar = self.single_pass(input_states, actions,i, train=True)

            cov_matrix = torch.diag_embed(torch.exp(logvar))
            normal_distribution = torch.distributions.MultivariateNormal(mean, cov_matrix)
            log_prob = normal_distribution.log_prob(target)
            loss = -log_prob.mean()
            total_loss += loss
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item(), 0, 0, 0


class AutoregressiveDeltaModel(AutoregressiveModel):
    def __init__(self, hidden_size, dt, lr=1e-3, weight_decay=1e-4):
        super().__init__(hidden_size, dt, lr, weight_decay)
        state_size = len(output_keys)
        action_size = 2
        self.min_state = None #min_state
        self.max_state = None # max_state
        self.state_size, self.action_size, self.dt = state_size, action_size, dt
        A_size, B_size = state_size * state_size, state_size * action_size
        # hidden layers for A
        self.A_layers = nn.ModuleList()
        self.A_layers.append(self._make_layer(state_size * 3 + action_size, hidden_size[0]))
        for i in range(1, len(hidden_size)):
            self.A_layers.append(self._make_layer(hidden_size[i-1], hidden_size[i]))
        self.A_layers.append(self._make_layer(hidden_size[-1], 1))

        # hidden layers for B
        self.B_layers = nn.ModuleList()
        self.B_layers.append(self._make_layer(state_size * 3 + action_size, hidden_size[0]))
        for i in range(1, len(hidden_size)):
            self.B_layers.append(self._make_layer(hidden_size[i-1], hidden_size[i]))
        self.B_layers.append(self._make_layer(hidden_size[-1], 1))

        self.optimizer_dynamics = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.coloumn_names = output_keys

    def forward(self, x, u):
        """
        Predict x_{t+1} = f(x_t, u_t)
        :param x: a batch of states
        :param u: a batch of actions
        """
        assert self.column_names is not None
        #in order to make learning easier apply to the u the clipping and scaling
        # u = torch.clip(u, -1, 1) * 0.05
        xu = torch.cat((x, u), -1)
        # set the x and y states to 0 
        x_column = self.column_names.index('poses_x')
        y_column = self.column_names.index('poses_y')

        xu[:,x_column] = 0.0  # Remove dependency in (x,y)
        xu[:,y_column] = 0.0  # Remove dependency in (x,y)
        xu[:,x_column + self.state_size] = 0.0  # Remove dependency in (x,y) (next_state)
        xu[:,y_column + self.state_size] = 0.0  # Remove dependency in (x,y) (next_state)

        for layer in self.A_layers[:-1]:
            xu = F.relu(layer(xu))

        A = self.A_layers[-1](xu)  # Last layer
        A = torch.reshape(A, (x.shape[0], 1))
        
        # Reset and pass through B hidden layers
        xu = torch.cat((x, u), -1)
        xu[:, x_column] = 0.0
        xu[:, y_column] = 0.0
        for layer in self.B_layers[:-1]:  # All but the last layer
            xu = F.relu(layer(xu))
        
        B = self.B_layers[-1](xu)  # Last layer
        B = torch.reshape(B, (x.shape[0], 1))
        
        dx = A @ x.unsqueeze(-1) + B @ u.unsqueeze(-1)
        x = x + dx.squeeze()*self.dt
        x_new = torch.clamp(x, self.min_state, self.max_state)
        return x_new


class ModelBasedEnsemble(object):
    def __init__(self, dynamics_network, N, min_state, max_state):
        # clone the dynamics network
        self.N = N
        self.models = [build_dynamics_model(dynamics_network, 
                                                       min_state=min_state, 
                                                       max_state=max_state) for _ in range(N)]
        self.min_state = None # this is a dummy value, since assigning badly the other networks
        self.max_state = None # this is a dummy value, since assigning badly the other networks
        # print(self.dynamics_networks.min_state)
    def __call__(self, x, u):
        predictions = []
        for model in self.models:
            prediction = model(x, u)
            predictions.append(prediction)
        # average the predictions
        predictions = torch.stack(predictions, dim=0).mean(dim=0)
        return predictions
    def set_min_state(self, min_state):
        self.min_state = min_state
        for model in self.models:
            model.min_state = min_state

    def set_max_state(self, max_state):
        self.max_state = max_state
        for model in self.models:
            model.max_state = max_state

    def update(self, states, action, next_states):
        all_loss = 0
        for i, model in enumerate(self.models):
            # Sample with replacement to create a bootstrapped batch
            batch_size = states.size(0)
            indices = torch.randint(0, batch_size, (batch_size,)).to(self.device)
            bootstrapped_states = states[indices]
            bootstrapped_actions = action[indices]
            bootstrapped_next_states = next_states[indices]
            #print(bootstrapped_actions.device)

            # Reset gradients for the current model
            loss, _, _ , _ = model.update(bootstrapped_states, bootstrapped_actions, bootstrapped_next_states)
            all_loss += loss
        """
        if self.dynamics_optimizer_iterations % 10 == 0:
            self.writer.add_scalar(f"train/loss{i}/xy", loss[0], global_step=self.dynamics_optimizer_iterations)
            self.writer.add_scalar(f"train/loss{i}/theta", loss[1], global_step=self.dynamics_optimizer_iterations)
            self.writer.add_scalar(f"train/loss{i}/vel", loss[2], global_step=self.dynamics_optimizer_iterations)
            self.writer.add_scalar(f"train/loss{i}/progr", loss[3], global_step=self.dynamics_optimizer_iterations)
            # add mean of all loses
            #self.writer.add_scaler(f"train/loss{i}/mean", loss[0], global_step=self.dynamics_optimizer_iterations)
        self.dynamics_optimizer_iterations += 1
        """
        return loss, 0, 0, 0

    def set_device(self, device):
        self.device = device
        for model in self.models:
            model.set_device(device)
    def save(self, path, filename="model_based_torch_checkpoint.pth"):
        for i, model in enumerate(self.models):
            model.save(path, filename + f"_{i}")
    def load(self, path, filename="model_based_torch_checkpoint.pth"):
        for i, model in enumerate(self.models):
            model.load(path, filename + f"_{i}")
        

class ProbsDeltaDynamicsModel(DeltaDynamicsModel):
    def __init__(self, hidden_size, dt, lr=1e-3, weight_decay=1e-4, min_state=None, max_state=None):
        super().__init__(hidden_size, dt, lr, weight_decay, min_state= min_state, max_state=max_state)
        # replace the last layers with layers that help output the mean and variance
        # hidden layers for A
        self.A_layers[-1] = self._make_layer(hidden_size[-1], self.state_size*self.state_size*2)
        self.B_layers[-1] = self._make_layer(hidden_size[-1], self.state_size*self.action_size*2)
        self.optimizer_dynamics = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


    def forward(self, x, u , train=False):
        assert self.column_names is not None
        #in order to make learning easier apply to the u the clipping and scaling
        # u = torch.clip(u, -1, 1) * 0.05
        xu = torch.cat((x, u), -1)
        # set the x and y states to 0 
        x_column = self.column_names.index('poses_x')
        y_column = self.column_names.index('poses_y')
        theta_sin_column = self.column_names.index('theta_sin')
        theta_cos_column = self.column_names.index('theta_cos')
        xu[:,x_column] = 0.0  # Remove dependency in (x,y)
        xu[:,y_column] = 0.0  # Remove dependency in (x,y)
        xu[:,theta_sin_column] = 0.0  # Remove dependency in theta
        xu[:,theta_cos_column] = 0.0  # Remove dependency in theta


        for layer in self.A_layers[:-1]:
            xu = F.relu(layer(xu))

        A = self.A_layers[-1](xu)  # Last layer
        A = torch.reshape(A, (x.shape[0], self.state_size*2, self.state_size))
        
        # Reset and pass through B hidden layers
        xu = torch.cat((x, u), -1)
        xu[:, x_column] = 0.0
        xu[:, y_column] = 0.0
        xu[:,theta_sin_column] = 0.0  # Remove dependency in theta
        xu[:,theta_cos_column] = 0.0  # Remove dependency in theta

        for layer in self.B_layers[:-1]:  # All but the last layer
            xu = F.relu(layer(xu))
        
        B = self.B_layers[-1](xu)  # Last layer
        B = torch.reshape(B, (x.shape[0], self.state_size*2, self.action_size))
        #x_a = torch.zeros_like(x)
        #x_a[:,4:] = x[:,4:]
        # print(x_a[0])
        dx = A @ x.unsqueeze(-1) + B @ u.unsqueeze(-1)
        #print(dx.shape)
        mean = dx[:,0:self.state_size].squeeze()
        #print(mean.shape)
        logvar = dx[:,self.state_size:].squeeze()
        #print(logvar.shape)
        
        mean = x + dx[:,0:self.state_size].squeeze()*self.dt

        # normalize the theta, it should sum to 1
        theta_sin_cos_norm = torch.sqrt(mean[:, theta_sin_column]**2 + mean[:, theta_cos_column]**2)
        new_mean = mean.clone()
        new_mean[:, theta_sin_column] = mean[:, theta_sin_column] / theta_sin_cos_norm
        new_mean[:, theta_cos_column] = mean[:, theta_cos_column] / theta_sin_cos_norm
 #print(mean.shape)
        x_new = torch.clamp(new_mean, self.min_state, self.max_state)
        #print("---")
        if train:
            return x_new, logvar
        else:
            return x_new
            
    def update(self, states, action, next_states):
        self.optimizer_dynamics.zero_grad()
        mean, logvar = self(states, action, train=True)
        # clip the logvar to be between -10 and 10, 
        # otherwise early training might crash with nans
        logvar = torch.clamp(logvar, -10, 10)
        cov_matrix = torch.diag_embed(torch.exp(logvar))
        normal_distribution = torch.distributions.MultivariateNormal(mean, cov_matrix)
        log_prob = normal_distribution.log_prob(next_states)
        loss = -log_prob.mean() 
        loss.backward()
        self.optimizer_dynamics.step()
        return loss.item(), 0, 0, 0

class F110ModelBased(object):
    def __init__(self,env, state_dim, action_dim, dynamics_model, hidden_size, dt, min_state,max_state,
                 logger, fn_normalize, fn_unnormalize,obs_keys, use_reward_model=False, use_done_model=False,
                 learning_rate=1e-3, weight_decay=1e-4,target_reward=None):
        self.use_reward_model = use_reward_model
        self.use_done_model = use_done_model
        self.env = env
        self.device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.relevant_keys = output_keys
        self.state_dim, self.action_dim, self.hidden_size, self.dt = state_dim, action_dim, hidden_size, dt
        self.fn_normalize = fn_normalize
        self.fn_unnormalize = fn_unnormalize

        self.obs_keys = obs_keys
        # gather the relevant keys
        # add a 0 dim to min_states

        min_state, column_names = self.gather_relevant_keys(torch.unsqueeze(min_state, 0))
        max_state, _ = self.gather_relevant_keys(torch.unsqueeze(max_state, 0))

        
        # print(self.min_state.shape)
        self.dynamics_model = dynamics_model
        self.min_state = min_state[0]#.to(self.device)
        self.max_state = max_state[0]#.to(self.device)
        # print(self.min_state)
        self.min_state.to(self.device)
        self.max_state.to(self.device)
        self.dynamics_model.min_state = self.min_state
        self.dynamics_model.max_state = self.max_state
        self.dynamics_model.set_min_state(self.min_state)
        self.dynamics_model.set_max_state(self.max_state)
                            #DynamicsModel(hidden_size, dt, 
                              #              self.min_state, self.max_state, 
                              #              column_names=output_keys,
                              #              lr=learning_rate,
                              #              weight_decay=weight_decay)
        if self.device == "cuda":
            self.dynamics_model.to(self.device)
        """
        self.dynamics_model = ModelBasedEnsemble(state_dim, 
                                                   action_dim,
                                                    hidden_size,
                                                    dt,
                                                    min_state,
                                                    max_state,
                                                    logger, fn_normalize, fn_unnormalize,
                                                    obs_keys,
                                                    learning_rate=learning_rate,
                                                    weight_decay=weight_decay,
                                                    N=3)
        """
        #config = Config(target_reward)
        #self.reward_model = GroundTruthRewardFast(dataset, 20,config)
        self.reward_config = target_reward
        self.writer=logger
        
        self.dynamics_optimizer_iterations = 0
        if self.use_reward_model:
            self.reward_model = RewardModel(state_dim, 
                                            hidden_size[0],
                                            0.0,
                                            1.0)
        if self.use_done_model:
            self.done_model = DoneNetwork(state_dim, 
                                        hidden_size[0])
    def set_device(self, device):
        self.device = device
        self.dynamics_model.set_device(self.device)
        if self.use_reward_model:
            self.reward_model.to(self.device)
        if self.use_done_model:
            self.done_model.to(self.device)
    """
    @brief actions are not requiered since they are recovered from the next states!
    """
    def update(self, original_states, actions, original_next_states, rewards, masks):
        states_in = self.env.get_specific_obs(original_states, output_keys)
        next_states_in = self.env.get_specific_obs(original_next_states, output_keys)
        curr_actions = self.env.get_specific_obs(original_next_states, ['previous_action_steer',
                                                                         'previous_action_speed'])
        # send to cuda if self device is cuda
        states_in = states_in.to(self.device)
        next_states_in = next_states_in.to(self.device)
        curr_actions = curr_actions.to(self.device)
        masks = masks.to(self.device)
        rewards = rewards.to(self.device)
        original_states = original_states.to(self.device)
        #print(curr_actions.shape)
        #print(curr_actions.max())
        #print(curr_actions.min())
        #print("...")
        loss, _ , _ , _ = self.dynamics_model.update(states_in, curr_actions, next_states_in) #, masks, self.dynamics_optimizer_iterations)
        loss_reward = 0
        loss_done = 0
        if self.use_reward_model:
            loss_reward = self.reward_model.update(original_states, rewards)

        if self.use_done_model:
            loss_done = self.done_model.update(original_states, masks)
        
        # print(loss)
        self.dynamics_optimizer_iterations += 1

        return loss, loss_reward , loss_done
    
    def evaluate_ss(self, states, actions, next_states):
        pred_states = self(states, actions)
        mse = F.mse_loss(pred_states, next_states)
        return mse


    def __call__(self, states, actions, logvar=False):
        states_in = self.env.get_specific_obs(states, output_keys)
        action_states = self.env.get_specific_obs(states, ['previous_action_steer',
                                                      'previous_action_speed'])
        action_states_unnorm = self.fn_unnormalize(action_states, ['previous_action_steer',
                                                        'previous_action_speed'])
        actions = action_states_unnorm + actions
        #print(states_in)
        #print(self.fn_unnormalize(states_in,["poses_x", "poses_y"]))
        #print(actions)
        # renormalize the actions
        actions = self.fn_normalize(actions, ['previous_action_steer',
                                                        'previous_action_speed'])
        
        #print(actions)
        #print(states_in)
        if logvar:
            # TODO! check if model supports it
            next_states, logvar = self.dynamics_model(states_in, actions, train=logvar)
            #next_states = torch.cat((next_states, actions), dim=1)
            return next_states, logvar
        else:
            next_states = self.dynamics_model(states_in, actions)
        #print(next_states)
        # now we need to add the rest of the observations, these are the previous actions
        # lets append the filtered out observations
        # 1. the new previous actions
        next_states = torch.cat((next_states, actions), dim=1)
        #print(next_states)
        assert states.shape == next_states.shape , "states and next states should have the same shape, but have {} and {}".format(states.shape, next_states.shape)
        return next_states
    
    def gather_relevant_keys(self, states ,keys = None):
        if keys is None:
            keys = self.relevant_keys
        # Find the indices of the relevant keys in obs_keys
        indices = [self.obs_keys.index(key) for key in keys if key in self.obs_keys]
        # Use these indices to extract the relevant columns from states
        relevant_states = states[:, indices]
        relevant_column_names = [self.obs_keys[i] for i in indices]
        return relevant_states, relevant_column_names



    def termination_model(self,trajectories, map_path):
        """
        @brief takes in normalized states and applies the termination model to each state
        @trajectories (trajectory, timesteps, state_dim)
        """
        # trajectories = self.fn_unnormalize(trajectories, self.relevant_keys)
        #print(trajectories)
        #with open(map_path, 'r') as file:
        #    map_metadata = yaml.safe_load(file)
        #    map_image_path = os.path.join(os.path.dirname(map_path), map_metadata['image'])
        #    map_image = Image.open(map_image_path).convert('L')  # Convert to grayscale
        #map_array = np.array(map_image)
        terminations = []
        for trajectory in trajectories:
            # for each state get the scan
            laser_scans = self.env.get_laser_scan(trajectory, 20)
            laser_scans = self.env.normalize_laser_scan(laser_scans)
            terminations_trajectory ,_ = self.env.termination_scan(trajectory, laser_scans, np.zeros((trajectory.shape[0])))
            # add the check if we are in/outside the map, propably not needed
            if np.where(terminations_trajectory==1)[0].size == 0:
                terminations.append(trajectory.shape[0] + 1)
            else:
                terminations.append(np.where(terminations_trajectory==1)[0][0])
        return np.array(terminations)
    
        # for each trajectory


    def rollout(self, states, actions=None, get_target_action=None, horizon=10, batch_size=256, use_dynamics=True):
        #print("HI")
        with torch.no_grad():
            states_initial = states[:,0,:]
            state_batches = torch.split(states_initial, batch_size) # only do rollouts from timestep 0
            #print(states.shape)
            #print((0, horizon, states.shape[-1]))
            all_states = torch.zeros((0, horizon, states.shape[-1]))
            all_actions = torch.zeros((0, horizon, 2))
            
            for num_batch, state_batch in enumerate(state_batches):
                assert len(state_batch.shape) == 2
                assert state_batch.shape[0] <= batch_size
                all_state_batches = torch.zeros((state_batch.shape[0], 0, state_batch.shape[-1]))
                all_actions_batches = torch.zeros((state_batch.shape[0], 0, 2))
                for i in range(horizon):

                    if get_target_action is None:
                        action = actions[num_batch*batch_size:batch_size*(num_batch+1),i,:] # (batch,2)
                        action = action.float()
                        assert(action.shape[0] == state_batch.shape[0])
                        assert(action.shape[1]==2)
                    else:
                        action = get_target_action(state_batch, keys=mb_keys) #.to('cpu').numpy())
                        assert(action.shape[0] == state_batch.shape[0])
                        assert(action.shape[1]==2)
                        action = action
                        #make dtype float32
                        action = action.float()
                    # add the action to all_batch_actions along dim=1
                    #print("--")
                    #print(state_batch.unsqueeze(1).shape)
                    #print(action.shape)
                    #print(all_state_batches.shape)
                
                    all_actions_batches = torch.cat([all_actions_batches, action.unsqueeze(1)], dim=1)
                    all_state_batches = torch.cat([all_state_batches, state_batch.unsqueeze(1)], dim=1)
                    
                    if use_dynamics:
                        #print(state_batch.shape)
                        #print("????")
                        state_batch = self(state_batch, action)
                    elif horizon-1 != i:
                        state_batch = states[:,i+1,:]

                all_states = torch.cat([all_states, all_state_batches], dim=0)
                all_actions = torch.cat([all_actions, all_actions_batches], dim=0)
            return all_states, all_actions
    
    """
    @brief requires unnormalized states
    """
    def calculate_progress(self, states):
        from f110_orl_dataset.compute_progress import Progress, Track
        progress_obs_np = np.zeros((states.shape[0],states.shape[1],1))
        track_path = "/home/fabian/msc/f110_dope/ws_release/f1tenth_gym/gym/f110_gym/maps/Infsaal3/Infsaal3_centerline.csv"
        track = Track(track_path)
        progress = Progress(track, lookahead=200)
        pose = lambda traj_num, timestep: np.array([(states[traj_num,timestep,0],states[traj_num,timestep,1])])
        for i in range(0,states.shape[0]):
            # progress = Progress(states_inf[i,0,:])
            progress.reset(pose(i,0))
            for j in range(0,states.shape[1]):
                progress_obs_np[i,j,0] = progress.get_progress(pose(i,j))
        return progress_obs_np

    def estimate_mse_pose(self, states, get_target_action, horizon=250, batch_size=256):
        with torch.no_grad():
            inital_states = states[:,0,:]
            actions = torch.zeros((states.shape[0], 1, 2))
            all_states, all_actions = self.rollout(states, actions,get_target_action, horizon, batch_size)
            #print(all_states.shape)
            #print(states.shape)
            mse = torch.mean((all_states[:,:,:2] - states[:,0:horizon+1,:2])**2)
            
            return mse


    def estimate_returns(self, inital_states, get_target_action, map, horizon=250, discount=0.99, batch_size=256, plot=False, done_function=None):
        from f110_orl_dataset.fast_reward import MixedReward
        from f110_orl_dataset.config_new import Config

        with torch.no_grad():
            states = inital_states.unsqueeze(1)
            actions = torch.zeros((states.shape[0], 1, 2))
            all_states, all_actions = self.rollout(states, actions,get_target_action, horizon, batch_size)
            if not self.use_reward_model:
                config = Config(self.reward_config)
                mixedReward = MixedReward(self.env, config)
                # we need to massage all_states and all_actions into the right format
                unnormalized_states = self.fn_unnormalize(all_states, self.relevant_keys)
                progress_obs = self.calculate_progress(unnormalized_states)

                unnormalized_states = np.concatenate((unnormalized_states, progress_obs), axis=2)
                # add a zero columen, thats the format fast reward expects
                unnormalized_states = np.concatenate((unnormalized_states, np.zeros((unnormalized_states.shape[0],unnormalized_states.shape[1],1))), axis=2)
                
                num_trajectories = len(all_states)


                all_rewards = np.zeros((num_trajectories, horizon))
                for i in range(num_trajectories):
                    # print("!")
                    obs = unnormalized_states[i]
                    action = self.env.get_specific_obs(obs, ["previous_action_steer","previous_action_speed"])
                    col = ~self.is_drivable(map, obs[:,0:2])
                    ter = col
                    
                    laser_scan = self.env.get_laser_scan(obs, 20)
                    # add a dimension to all the inputs
                    laser_scan =  np.expand_dims(laser_scan, axis=0)
                    obs = np.expand_dims(obs, axis=0)
                    action = np.expand_dims(action, axis=0)
                    col = np.expand_dims(col, axis=0)
                    ter = np.expand_dims(ter, axis=0)
                    rewards, _ = mixedReward(obs, action,col, ter,laser_scan=laser_scan)
                    all_rewards[i] = rewards
                    # set rewards to zero where is not drivable / terminal
                    #print(col)
                    #print(col.shape)
                    #print(rewards.shape)
                    first_crash = np.where(col == True)[0]
                    if first_crash.shape[0] > 0:
                        all_rewards[i, first_crash[0]:] = 0.0
                # for each trajectory plot the reward
                # plot all trajectories
            else: # i.e. we use the reward model
                #print(all_states.shape)
                #print(all_actions.shape)
                num_trajectories = len(all_states)
                all_rewards = np.zeros((num_trajectories, horizon))
                for i in range(num_trajectories):
                    with torch.no_grad():
                        #print(all_states[i].shape)
                        rewards = self.reward_model(all_states[i])
                        all_rewards[i] = rewards.cpu().numpy()

            # now lets apply the done
            all_dones = np.ones((num_trajectories, horizon))
            if self.use_done_model:
                pass 
                # this does not really learn anything, just skip for now
                """
                num_trajectories = len(all_states)
                for i in range(num_trajectories):
                    with torch.no_grad():
                        dones = self.done_model(all_states[i])
                        dones = dones.cpu().numpy()
                        first_done = np.where(dones == False)[0]
                        print(dones)
                        print(first_done)
                        if first_done.shape[0] > 0:
                            all_rewards[i, first_done[0]:] = 0.0
                            all_dones[i, first_done[0]:] = 1.0
                """
                #raise NotImplementedError
            else:
                unnormalized_states = self.fn_unnormalize(all_states, self.relevant_keys)
                for i in range(num_trajectories):
                    obs = unnormalized_states[i]
                    col = ~self.is_drivable(map, obs[:,0:2].numpy())
                    first_crash = np.where(col == True)[0]
                    if first_crash.shape[0] > 0:
                        all_rewards[i, first_crash[0]:] = 0.0
                        all_dones[i, first_crash[0]:] = 1.0

            if plot:
                self.plot_trajectories_on_map(map, unnormalized_states)

                for i in range(num_trajectories):
                    plt.plot(all_rewards[i])
                    plt.show()
                    # also plot each trajectory on map
                    self.plot_poses_on_map(map, unnormalized_states[i], all_dones[i])
                # apply discount to the rewards
            discounted_rewards = np.zeros((num_trajectories,))
            for i in range(num_trajectories):
                discounted_rewards[i] = np.sum(all_rewards[i] * np.power(discount, np.arange(len(all_rewards[i]))))

            return np.mean(discounted_rewards), np.std(discounted_rewards)
            

    def plot_trajectories_on_map(self, yaml_path, poses):
        # Load map metadata
        with open(yaml_path, 'r') as file:
            map_metadata = yaml.safe_load(file)

        # Construct the path for the map image
        map_image_path = os.path.join(os.path.dirname(yaml_path), map_metadata['image'])
        map_image = Image.open(map_image_path)
        map_array = np.array(map_image)

        # Display the map
        plt.imshow(map_image, cmap='gray')

        # Map parameters
        resolution = map_metadata['resolution']
        origin = map_metadata['origin']

        # Number of trajectories
        num_trajectories = poses.shape[0]

        # Generate color map
        colors = cm.rainbow(np.linspace(0, 1, num_trajectories))

        # Plot each trajectory
        for i in range(num_trajectories):
            # Convert poses to pixel coordinates, invert y-axis
            pixel_poses = poses[i].copy()
            pixel_poses[:, 0] = (pixel_poses[:, 0] - origin[0]) / resolution
            pixel_poses[:, 1] = map_array.shape[0] - ((pixel_poses[:, 1] - origin[1]) / resolution)

            # Plot trajectory
            plt.plot(pixel_poses[:, 0], pixel_poses[:, 1], color=colors[i], label=f'Trajectory {i+1}')

        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.title('Trajectories on Map')
        plt.legend()
        plt.show()

    def is_drivable(self,yaml_path, poses):
        # Load map metadata
        with open(yaml_path, 'r') as file:
            map_metadata = yaml.safe_load(file)

        # Load the map image
        map_image_path = os.path.join(os.path.dirname(yaml_path), map_metadata['image'])
        map_image = Image.open(map_image_path)
        map_array = np.array(map_image)

        # Map parameters
        resolution = map_metadata['resolution']  # meters per pixel
        origin = map_metadata['origin']  # [x, y, theta]
        occupied_thresh = map_metadata['occupied_thresh']

        # Convert poses to pixel coordinates, invert y-axis
        pixel_poses = poses.copy()
        pixel_poses[:, 0] = (pixel_poses[:, 0] - origin[0]) / resolution
        pixel_poses[:, 1] = map_array.shape[0] - 1 - ((pixel_poses[:, 1] - origin[1]) / resolution)
        pixel_poses = pixel_poses.astype(int)

        # Check bounds
        in_bounds = (pixel_poses[:, 0] >= 0) & (pixel_poses[:, 0] < map_array.shape[1]) & \
                    (pixel_poses[:, 1] >= 0) & (pixel_poses[:, 1] < map_array.shape[0])

        # Check if the area is drivable (not occupied)
        # Assuming drivable area is white (high pixel value)
        drivable_threshold = int(255 * (1 - occupied_thresh))
        is_drivable = np.array([map_array[pixel_y, pixel_x] > drivable_threshold if in_bounds[i] else False 
                                for i, (pixel_x, pixel_y) in enumerate(pixel_poses)])
        first_false = np.where(is_drivable == False)[0]
        #print(first_false)
        if len(first_false) > 0:
            first_false = first_false[0]
            is_drivable[first_false:] = False
        return is_drivable
    
    def plot_poses_on_map(self, yaml_path, poses, is_drivable):
        import matplotlib.patches as mpatches
        # Load map metadata
        with open(yaml_path, 'r') as file:
            map_metadata = yaml.safe_load(file)

        # Construct the path for the map image
        map_image_path = os.path.join(os.path.dirname(yaml_path), map_metadata['image'])
        map_image = Image.open(map_image_path)
        map_array = np.array(map_image)

        # Display the map
        plt.imshow(map_image, cmap='gray')

        # Map parameters
        resolution = map_metadata['resolution']
        origin = map_metadata['origin']

        # Convert poses to pixel coordinates, invert y-axis
        pixel_poses = poses.copy()
        pixel_poses[:, 0] = (pixel_poses[:, 0] - origin[0]) / resolution
        pixel_poses[:, 1] = map_array.shape[0] - ((pixel_poses[:, 1] - origin[1]) / resolution)

        # Plot each pose
        for pose, drivable in zip(pixel_poses, is_drivable):
            # print(is_drivable)
            if drivable:
                plt.plot(pose[0], pose[1], 'o', color='green')  # Drivable: green circle
            else:
                plt.plot(pose[0], pose[1], 'x', color='red')  # Not drivable: red x

        # Add legend
        drivable_patch = mpatches.Patch(color='green', label='Drivable')
        not_drivable_patch = mpatches.Patch(color='red', label='Not Drivable')
        plt.legend(handles=[drivable_patch, not_drivable_patch])

        plt.gca().invert_yaxis()  # Invert y-axis to match ROS map orientation
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.title('Poses on Map')
        plt.show()
    
    def save(self, save_path, filename="model_based2_torch_checkpoint.pth"):
        """
        Save the model's state dictionaries.
        
        Args:
        - save_path (str): The directory path where the model should be saved.
        - filename (str, optional): The name of the checkpoint file. Defaults to "model_based2_checkpoint.pth".
        """
        
        # Ensure the directory exists
        os.makedirs(save_path, exist_ok=True)

        # Define the checkpoint

        self.dynamics_model.save(save_path, filename)
        if self.use_reward_model:
            self.reward_model.save(save_path, "reward_model_"+filename)
        if self.use_done_model:
            self.done_model.save(save_path, "done_model_"+filename)
    
    def load(self, save_path, filename="model_based2_torch_checkpoint.pth"):
        """
        Load the model's state dictionaries.
        
        Args:
        - save_path (str): The directory path where the model should be loaded from.
        - filename (str, optional): The name of the checkpoint file. Defaults to "model_based2_checkpoint.pth".
        """
        
        self.dynamics_model.load(save_path, filename)
        if self.use_reward_model:
            self.reward_model.load(save_path, "reward_model_"+filename)
        if self.use_done_model:
            self.done_model.load(save_path, "done_model_"+filename)

    