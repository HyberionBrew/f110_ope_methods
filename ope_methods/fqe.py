import torch 
from torch import nn
# import optim
import torch.optim as optim
import torch.nn.functional as F
from ope_methods.fqe_critics import CriticNetL2, CriticNetDD

class QFitterBase(nn.Module):
    """Base class for fitting Q-functions, providing common utilities."""

    def __init__(self, critic_net, critic_target, critic_lr, weight_decay, min_reward, max_reward, tau=0.005):
        """Initializes the QFitterBase with a specific critic network.

        Args:
            critic_net: An instance of a critic network.
            critic_lr: Learning rate for the critic optimizer.
            weight_decay: Weight decay for the optimizer.
        """
        super(QFitterBase, self).__init__()
        self.critic = critic_net
        self.critic_target = critic_target # Assuming args and kwargs are stored in critic_net
        self.tau = tau
        self.soft_update(self.critic, self.critic_target, tau=1.0)
        self.optimizer = optim.AdamW(self.critic.parameters(), lr=critic_lr, weight_decay=weight_decay)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_reward = torch.tensor(min_reward)
        self.max_reward = torch.tensor(max_reward)

    def set_device(self, device):
        """Moves the module's parameters and buffers to the specified device."""
        self.critic.to(device)
        self.critic_target.to(device)
        self.device = device
        self.min_reward = self.min_reward.to(device)
        self.max_reward = self.max_reward.to(device)
        return self

    def save(self, path, i=0):
        """Saves the critic and critic target state dictionaries."""
        torch.save(self.critic.state_dict(), f"{path}/critic{i}.pth")
        torch.save(self.critic_target.state_dict(), f"{path}/critic_target{i}.pth")

    def load(self, path, i=0):
        """Loads the critic and critic target state dictionaries."""
        self.critic.load_state_dict(torch.load(f"{path}/critic{i}.pth"))
        self.critic_target.load_state_dict(torch.load(f"{path}/critic_target{i}.pth"))

    def forward(self, *args, **kwargs):
        """Placeholder for forward pass to be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement this!")

    def update(self, *args, **kwargs):
        """Placeholder for update logic to be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement this!")
    
    def soft_update(self, local_model, target_model, tau=0.005):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class QFitterL2(QFitterBase):
    """A critic network that estimates a dual Q-function."""

    def __init__(self, state_dim, action_dim, hidden_sizes, min_reward, max_reward, average_reward, critic_lr=3e-5, 
                 weight_decay=1e-5, tau=0.005, discount=0.99, 
                 log_frequency=500, logger=None):
        """Creates networks.
        
        Args:
          state_dim: State size.
          action_dim: Action size.
          critic_lr: Critic learning rate.
          weight_decay: Weight decay.
          tau: Soft update discount.
        """
        super(QFitterL2, self).__init__(CriticNetL2(state_dim, action_dim, hidden_sizes, average_reward_offset=average_reward*(1-discount)), 
                                        CriticNetL2(state_dim, action_dim, hidden_sizes, average_reward_offset=average_reward*(1-discount)), 
                                        critic_lr, 
                                        weight_decay, 
                                        min_reward=min_reward,
                                        max_reward=max_reward,
                                        tau=tau)
       
        self.log_frequency = log_frequency
        self.optimizer_iterations = 0
        self.writer = logger
        self.discount = discount

    def forward(self, states, actions, timesteps=None, batch_size=1000):
        # batch this call to avoid OOM
        n_data = states.shape[0]
        results = []

        for i in range(0, n_data, batch_size):
            batch_states = states[i: min(i + batch_size, n_data)]
            batch_actions = actions[i: min(i + batch_size, n_data)]
            batch_result = self.critic_target(batch_states, batch_actions)
            results.append(batch_result)

        final_result = torch.cat(results, dim=0) 
        #print("inside results", final_result)
        return final_result / (1 - self.discount)  # adding the average reward, so less updates requiered
    
    def update(self, states, actions, next_states, next_actions, rewards, masks):
        """Updates critic parameters."""
        # if cuda is available send all there and if model is there

        if self.device =="cuda":
            states = states.cuda()
            actions = actions.cuda()
            next_states = next_states.cuda()
            next_actions = next_actions.cuda()
            rewards = rewards.cuda()
            masks = masks.cuda()

        self.optimizer.zero_grad()
        
        with torch.no_grad():
            next_q = self.critic_target(next_states, next_actions) / (1 - self.discount)
            target_q = (rewards + self.discount * masks * next_q)
            #print(rewards.shape)
            #print(masks.shape)
            #print(next_q.shape)
            #print(target_q.shape)
            target_q = torch.clamp(target_q, self.min_reward, self.max_reward)
            #print(target_q.shape)
            #print("..")
        
        q = self.critic(states, actions) / (1 - self.discount)
        #print("Q", q.shape)
        #print("Target Q", target_q.shape)
        #print(self.max_reward)
        critic_loss = torch.sum((target_q - q) ** 2) # take the mean? maybe
        #critic_loss = torch.mean((target_q - q) ** 2)
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0) 
        self.optimizer.step()

        self.soft_update(self.critic, self.critic_target, self.tau)

        if self.optimizer_iterations % self.log_frequency == 0:
            self.writer.add_scalar('train/fqe/loss', critic_loss.item(), global_step=self.optimizer_iterations)
        self.optimizer_iterations += 1
        return critic_loss.item()
    
    def estimate_returns(self, initial_states, initial_actions,scans=None):
        """Estimate returns with fitted q learning."""
        # to cuda
        initial_states = initial_states.cuda()
        initial_actions = initial_actions.cuda()
        with torch.no_grad():
            preds = self(initial_states, initial_actions)
            # back to cpu
            # clamp predictions to min and max
            preds = torch.clamp(preds, self.min_reward, self.max_reward)

            preds = preds.cpu()
            # print("Singular", preds)
            # print(preds)
            mean = torch.mean(preds)
            std = torch.std(preds)
        return mean, std

    def estimate_returns_unweighted(self, initial_states, get_action):
        """Estimate returns unweighted."""
        with torch.no_grad():
            initial_actions = get_action(initial_states)
            preds = self(initial_states, initial_actions)
        return preds 
    

class QFitterLME(QFitterBase):
    def __init__(self, state_dim, action_dim, hidden_sizes, min_reward, max_reward, average_reward, critic_lr=3e-5, 
                 weight_decay=1e-5, tau=0.005, discount=0.99, 
                 log_frequency=500, logger=None):
        super(QFitterLME, self).__init__(CriticNetL2(state_dim, action_dim, hidden_sizes, 
                                                     average_reward_offset=average_reward*(1-discount),
                                                     output_size=2), 
                                        CriticNetL2(state_dim, action_dim, hidden_sizes, 
                                                    average_reward_offset=average_reward*(1-discount),
                                                    output_size=2), 
                                        critic_lr, 
                                        weight_decay, 
                                        min_reward=min_reward,
                                        max_reward=max_reward,
                                        tau=tau)
       
        self.log_frequency = log_frequency
        self.optimizer_iterations = 0
        self.writer = logger
        self.discount = discount

    def forward(self, states, actions, timesteps=None, batch_size=1000):
        # batch this call to avoid OOM
        n_data = states.shape[0]
        results = []

        for i in range(0, n_data, batch_size):
            batch_states = states[i: min(i + batch_size, n_data)]
            batch_actions = actions[i: min(i + batch_size, n_data)]
            batch_result = self.critic_target(batch_states, batch_actions)
            results.append(batch_result)

        final_result = torch.cat(results, dim=0) 
        #print("inside results", final_result)
        return final_result / (1 - self.discount)  # adding the average reward, so less updates requiered
    
    def update(self, states, actions, next_states, next_actions, rewards, masks):
        """Updates critic parameters."""
        # if cuda is available send all there and if model is there

        if self.device =="cuda":
            states = states.cuda()
            actions = actions.cuda()
            next_states = next_states.cuda()
            next_actions = next_actions.cuda()
            rewards = rewards.cuda()
            masks = masks.cuda()

        self.optimizer.zero_grad()
        
        with torch.no_grad():
            next_q_logvar = self.critic_target(next_states, next_actions) 
            next_q = next_q_logvar[:, 0] / (1 - self.discount)
        
            #cov_matrix = torch.diag_embed(torch.exp(next_q_logvar))
            #normal_dist = torch.distributions.MultivariateNormal(next_q, cov_matrix)

            target_q = (rewards + self.discount * masks * next_q)
            #print(rewards.shape)
            #print(masks.shape)
            #print(next_q.shape)
            #print(target_q.shape)
            target_q = torch.clamp(target_q, self.min_reward, self.max_reward)
            #print(target_q.shape)
            #print("..")
        #print("inpout")
        #print(states)
        #print(actions)
        #print("..")
        q_logvar = self.critic(states, actions) 
        q = q_logvar[:, 0] / (1 - self.discount)
        logvar = q_logvar[:, 1]
        # print(q_logvar)
        cov_matrix = torch.diag_embed(torch.exp(logvar))

        normal_dist = torch.distributions.MultivariateNormal(q, cov_matrix)
        log_prob = normal_dist.log_prob(target_q)
        # critic_loss = torch.sum((target_q - q) ** 2)
        loss = -log_prob.mean()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0) 
        self.optimizer.step()

        self.soft_update(self.critic, self.critic_target, self.tau)

        if self.optimizer_iterations % self.log_frequency == 0:
            self.writer.add_scalar('train/fqe/loss', loss.item(), global_step=self.optimizer_iterations)
        self.optimizer_iterations += 1
        return loss.item()
    
    def estimate_returns(self, initial_states, initial_actions,scans=None):
        """Estimate returns with fitted q learning."""
        # to cuda
        initial_states = initial_states.cuda()
        initial_actions = initial_actions.cuda()
        with torch.no_grad():
            preds = self(initial_states, initial_actions)
            preds = preds[:, 0]
            # back to cpu
            # clamp predictions to min and max
            preds = torch.clamp(preds, self.min_reward, self.max_reward)

            preds = preds.cpu()
            # print("Singular", preds)
            print(preds)
            mean = torch.mean(preds)
            std = torch.std(preds)
        return mean, std
import matplotlib.pyplot as plt
class QFitterDD(QFitterBase):
    def __init__(self, state_dim, action_dim, hidden_sizes, 
                 min_reward, max_reward, critic_lr=3e-5, 
                 weight_decay=1e-5, tau=0.005, discount=0.99, 
                 log_frequency=500, logger=None, num_atoms=51):
        """Initializes the QFitterDD with CriticNetDD as the critic network.

        Args:
          state_dim: Size of the state space.
          action_dim: Size of the action space.
          num_atoms: Number of discrete atoms in the output distribution.
          Vmin: Minimum value of the support of the distribution.
          Vmax: Maximum value of the support of the distribution.
          critic_lr: Learning rate for the critic optimizer.
          weight_decay: Weight decay for the optimizer.
          tau: Soft update discount.
          discount: Discount factor for future rewards.
          log_frequency: Frequency of logging.
          writer: Logger for tensorboard.
        """
        super(QFitterDD, self).__init__(CriticNetL2(state_dim, action_dim, hidden_sizes, 
                                                    # average_reward_offset=average_reward*(1-discount),
                                                     output_size=num_atoms), 
                                        CriticNetL2(state_dim, action_dim, hidden_sizes, 
                                                    #average_reward_offset=average_reward*(1-discount),
                                                    output_size=num_atoms), 
                                        critic_lr = critic_lr, 
                                        weight_decay = weight_decay, 
                                        min_reward=min_reward,
                                        max_reward=max_reward,
                                        tau=tau)
       
        self.log_frequency = log_frequency
        self.optimizer_iterations = 0
        self.writer = logger
        self.discount = torch.tensor(discount)
        self.support = torch.linspace(min_reward, max_reward, num_atoms)
        self.delta_z = torch.tensor((max_reward - min_reward) / (num_atoms - 1))
        self.num_atoms = num_atoms
        print(self.support)
        print(self.delta_z)
    def set_device(self, device):
        # call parent set device
        super().set_device(device)
        self.support = self.support.to(device)
        self.delta_z = self.delta_z.to(device)
        self.discount = self.discount.to(device)
        return self
    
    def forward(self, states, actions, batch_size=1000):
        n_data = states.shape[0]
        results = []

        for i in range(0, n_data, batch_size):
            batch_states = states[i: min(i + batch_size, n_data)]
            batch_actions = actions[i: min(i + batch_size, n_data)]
            batch_result = self.critic_target(batch_states, batch_actions)
            results.append(batch_result)

        logits = torch.cat(results, dim=0) 
        #print("inside results", final_result)
        return logits
    
    def estimate_returns(self, initial_states, initial_actions,scans=None, plot=False):

        with torch.no_grad():
            initial_actions = initial_actions.cuda()
            initial_states = initial_states.cuda()
            preds = self(initial_states, initial_actions)
            # get index of the maximum value of the distribution
            # convert preds to softmax
            likelihood = torch.softmax(preds, dim=-1)

            preds = likelihood * self.support
        
            preds = preds.sum(dim=-1)
            # print(preds)
            #largest_atoms = preds.argmax(dim=-1)
            # transform to the actual value
            #preds = self.support[largest_atoms]

            preds = preds.cpu()
            # save the predictions as matplotlib
            if plot:
                # add figure with histogram of the prediction likelihood 
                # add figure with 2 subplots
                # 1. the actual value of the prediction
                # 2. the likelihood of the prediction
                # matplotlib figure
                fig, ax = plt.subplots(2)
                ax[0].hist(preds, bins=50)
                #print(likelihood.shape)
                #print(likelihood.sum(0).shape)
                x = torch.arange(101)

                ax[1].bar(x, likelihood.sum(0).cpu()/likelihood.shape[0])
                # save the fig

                plt.savefig(f"plots/histogram_{self.optimizer_iterations}.png")
                plt.close()
                

            # print("Singular", preds)
            mean = torch.mean(preds)
            std = torch.std(preds)
        return mean, std
    

    def update(self, states, actions, next_states, next_actions, rewards, masks):
        """Updates critic parameters."""
        # if cuda is available send all there and if model is there

        if self.device =="cuda":
            states = states.cuda()
            actions = actions.cuda()
            next_states = next_states.cuda()
            next_actions = next_actions.cuda()
            rewards = rewards.cuda()
            masks = masks.cuda()

        self.optimizer.zero_grad()
        # implementation similar to
        # https://github.com/facebookresearch/ReAgent/blob/main/reagent/training/c51_trainer.py#L120
        with torch.no_grad():
            logit_probs = self.critic_target(next_states, next_actions) 
            pmfs = torch.softmax(logit_probs, dim=-1)
            # self.support
            """
            print(rewards.unsqueeze(-1).shape)
            print(masks.unsqueeze(-1).shape)
            print(self.discount)
            print(self.support.view(1, -1).shape)
            # print the device of the tensors below
            print(rewards.unsqueeze(-1).device)
            print(masks.device)
            print(self.discount.device)
            print(self.support.device)
            """
            next_atoms = rewards.unsqueeze(-1) + self.discount * masks.unsqueeze(-1) * self.support.view(1, -1)
            next_atoms = torch.clamp(next_atoms, self.min_reward, self.max_reward)
            # projection
            b = (next_atoms - self.min_reward) / self.delta_z
            l = b.floor().clamp(0, self.num_atoms - 1)
            u = b.ceil().clamp(0, self.num_atoms - 1)
            d_m_l = (u + (l == u).float() - b) * pmfs
            d_m_u = (b - l) * pmfs
            target_pmfs = torch.zeros_like(pmfs)
            # this is some form of scatter add TODO! vis
            # plot

            
            for i in range(target_pmfs.size(0)):
                target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

        curr_atoms = self.critic(states, actions) 
        curr_pmfs = torch.softmax(curr_atoms, dim=-1)
        # critic_loss = torch.sum((target_q - q) ** 2)
        #print(target_pmfs[0])
        #print(curr_pmfs[0])
        loss =  (-(target_pmfs * curr_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0) 
        self.optimizer.step()
        self.soft_update(self.critic, self.critic_target, self.tau)
        #if self.optimizer_iterations % 100 == 0:
        #    self.soft_update(self.critic, self.critic_target, 1)

        if self.optimizer_iterations % self.log_frequency == 0:
            self.writer.add_scalar('train/fqe/loss', loss.item(), global_step=self.optimizer_iterations)
        self.optimizer_iterations += 1
        return loss.item()
