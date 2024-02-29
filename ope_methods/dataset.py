from torch.utils.data import Dataset
import numpy as np
import torch
import os

def create_save_dir(experiment_directory="test",algo="mb", reward_name="reward", dataset="f110-test", trajectory_length="250", target_policy="FTG", seed="1"):
    save_directory = os.path.join(experiment_directory)
    if not os.path.exists(save_directory):
      os.makedirs(save_directory)
    # now the algo directory
    save_directory = os.path.join(save_directory, f"{algo}")
    if not os.path.exists(save_directory):
      os.makedirs(save_directory)
    # path
    save_directory = os.path.join(save_directory, f"{dataset}")
    if not os.path.exists(save_directory):
      os.makedirs(save_directory)

    #now the max_timesteps directory
    save_directory = os.path.join(save_directory, f"{trajectory_length}")
    if not os.path.exists(save_directory):
      os.makedirs(save_directory)
    # now the target policy directory
    save_directory = os.path.join(save_directory, f"{target_policy}")
    if not os.path.exists(save_directory):
      os.makedirs(save_directory)
      # add seed
    save_directory = os.path.join(save_directory, f"{seed}")
    if not os.path.exists(save_directory):
      os.makedirs(save_directory)
    return save_directory


class F110Dataset(Dataset):
    def __init__(self, 
               d4rl_env,
               normalize_states = False,
               normalize_rewards = False,
               remove_agents = [],
               only_agents = [],
               state_mean=None,
               state_std = None,
               reward_mean = None,
               reward_std = None,
               train_only=False,
               eval_only=False):
        # Load the dataset

        d4rl_dataset = d4rl_env.get_dataset(
            remove_agents=remove_agents, 
            only_agents=only_agents,
            train_only=train_only,
            eval_only=eval_only,
        )
        # Assuming 'observations' and 'next_observations' are keys in the dataset
        #self.observations = self.data['observations']
        self.timestep_constant = None
        if 'timesteps_constant' in d4rl_dataset.keys():
            self.timestep_constant = d4rl_dataset['timesteps_constant']
        
        self.states = torch.from_numpy(d4rl_dataset['observations'].astype(np.float32))
        self.model_names = np.array(d4rl_dataset['model_name'])
        self.scans = torch.from_numpy(d4rl_dataset['scans'].astype(np.float32))
        self.actions = torch.from_numpy(d4rl_dataset['actions'].astype(np.float32))
        self.next_actions = None # assign this if needed
                # self.raw_actions = torch.from_numpy(d4rl_dataset['raw_actions'].astype(np.float32))
        self.rewards = torch.from_numpy(d4rl_dataset['rewards'].astype(np.float32))
        self.masks = torch.from_numpy(1.0 - d4rl_dataset['terminals'].astype(np.float32))
        self.log_probs = torch.from_numpy(d4rl_dataset['log_probs'].astype(np.float32))
        # self.timesteps = torch.from_numpy(d4rl_dataset['timesteps'].astype(np.float32))
        self.obs_keys = d4rl_dataset["infos"]["obs_keys"]
        # now we need to do next states and next scans
        # first check where timeout and where terminal
        finished = np.logical_or(d4rl_dataset['terminals'], d4rl_dataset['timeouts'])
        print(np.sum(d4rl_dataset['terminals']))
        print(np.sum(d4rl_dataset['timeouts']))
        print("..")
        self.finished = finished
        # rolled finished to the right by 1
        finished = torch.from_numpy(finished)
        self.mask_inital = torch.roll(finished, 1)
        assert(self.mask_inital[0] == True)
        # now lets loop over the [finished[i-1], finished[i]] and set the next states
        finished_indices = torch.where(finished)[0]
        start = 0
        self.states_next = torch.zeros_like(self.states)
        self.scans_next = torch.zeros_like(self.scans)
        self.rewards_next = torch.zeros_like(self.rewards)
        # zeros like (len(finished_indices), obs_shape)
        self.initial_states = torch.zeros((len(finished_indices), self.states.shape[-1]))
        self.initial_scans = torch.zeros((len(finished_indices), self.scans.shape[-1]))
        # unused inital_weights
        self.initial_weights = torch.ones(len(self.initial_states))
        for i, stop in enumerate(finished_indices):
            # append to dim 0
            next_states = torch.cat((self.states[start+1:stop+1], self.states[stop].unsqueeze(0)), dim=0)
            next_scans = torch.cat((self.scans[start+1:stop+1], self.scans[stop].unsqueeze(0)), dim=0)
            next_rewards = torch.cat((self.rewards[start+1:stop+1], torch.zeros_like(self.rewards[stop].unsqueeze(0))), dim=0)
            self.states_next[start:stop+1] = next_states
            self.scans_next[start:stop+1] = next_scans
            self.rewards_next[start:stop+1] = next_rewards
            self.initial_states[i] = self.states[start]
            self.initial_scans[i] = self.scans[start]
            start = stop + 1
        print("initial states", self.initial_states.shape)
        # now perform intelligent normalization from the dataset
        if normalize_states == True:
            if state_mean is None:
                self.state_mean = torch.mean(self.states, axis=0)
                #set the state mean of the cos and sin to zero, manual hack
                self.state_mean[2] = 0.0
                self.state_mean[3] = 0.0
                self.state_std = torch.std(self.states, axis=0)
                self.state_std[2] = 1.0
                self.state_std[3] = 1.0
            else:
                self.state_mean = state_mean
                self.state_std = state_std
            self.states = self.normalize_states(self.states)
            self.initial_states = self.normalize_states(self.initial_states)
            self.states_next = self.normalize_states(self.states_next)
        else:
            self.state_mean = torch.zeros_like(self.states[0])
            self.state_std = torch.ones_like(self.states[0])

        if normalize_rewards == True:
            # do this here
            if reward_mean is None:
                self.reward_mean = torch.mean(self.rewards)
                self.reward_std = torch.std(self.rewards)
            else:  
                self.reward_mean = reward_mean
                self.reward_std = reward_std

            self.reward_mean = torch.mean(self.rewards)
            self.reward_std = torch.std(self.rewards)
            self.rewards = self.normalize_rewards(self.rewards)
            self.rewards_next = self.normalize_rewards(self.rewards_next)
        else:
            self.reward_mean = 0.0
            self.reward_std = 1.0

    def normalize_rewards(self, rewards):
        return (rewards - self.reward_mean) / max(self.reward_std, 1e-8)
    
    def normalize_states(self, states, keys=None):
        if keys is None:
            keys = self.obs_keys
        indices = [self.obs_keys.index(key) for key in keys]

        states_return = states.clone()
        if len(states.shape) == 1:
            states_return = states_return.unsqueeze(0)

        for idx, key_idx in enumerate(indices):
            states_return[..., idx] = (states_return[..., idx] - self.state_mean[key_idx]) / max(self.state_std[key_idx], 1e-8)

        if len(states.shape) == 1:
            states_return = states_return.squeeze(0)
        return states_return

    def unnormalize_states(self, states, keys=None, eps=1e-8):
        if keys is None:
            keys = self.obs_keys
        indices = [self.obs_keys.index(key) for key in keys]

        states_return = states.clone()
        if len(states.shape) == 1:
            states_return = states_return.unsqueeze(0)

        for idx, key_idx in enumerate(indices):
            states_return[..., idx] = states_return[..., idx] * max(self.state_std[key_idx], eps) + self.state_mean[key_idx]

        if len(states.shape) == 1:
            states_return = states_return.squeeze(0)
        return states_return
    
    def unnormalize_rewards(self, rewards):
        return rewards * self.reward_std + self.reward_mean

    def __len__(self):
        return len(self.states)

    # returns (states, scans, actions, next_states, next_scans, rewards, masks, weights,
    # log_prob, timesteps)
    def __getitem__(self, idx):
        current_state = self.states[idx]
        next_state = self.states_next[idx]
        action = self.actions[idx]
        scan = self.scans[idx]
        next_scan = self.scans_next[idx]
        reward = self.rewards[idx]
        mask = self.masks[idx]
        log_prob = self.log_probs[idx]
        if self.next_actions is not None:
            next_action = self.next_actions[idx]
            return current_state, scan, action, next_state, next_scan, reward, mask, 1.0, log_prob, next_action
        #timestep = self.timesteps[idx]
        # Include other components like actions, rewards, etc. if needed
        return current_state, scan, action, next_state, next_scan, reward, mask, 1.0, log_prob 
    
    def get_only_indices(self, indices):
        self.states = self.states[indices]
        self.model_names = self.model_names[indices]
        self.scans = self.scans[indices]
        self.actions = self.actions[indices]
        self.rewards = self.rewards[indices]
        self.masks = self.masks[indices]
        self.finished = self.finished[indices]
        self.mask_inital = self.mask_inital[indices]
        self.log_probs = self.log_probs[indices]
        self.states_next = self.states_next[indices]
        self.scans_next = self.scans_next[indices]
        self.rewards_next = self.rewards_next[indices]
        # recompute the initial states, based on the new finished indices
        print(self.finished.shape)
        finished_indices = np.where(self.finished)[0]
        self.inital_states = torch.zeros((len(finished_indices), self.states.shape[-1]))
        self.initial_scans = torch.zeros((len(finished_indices), self.scans.shape[-1]))
        for i, stop in enumerate(finished_indices):
            self.initial_states[i] = self.states[stop]
            self.initial_scans[i] = self.scans[stop]
        return self
    

        
        
def random_split_trajectories(dataset, train_size=0.7, test_size=0.2, val_size=0.1):
    assert abs(train_size + test_size + val_size - 1) < 1e-6, "The sum of the sizes must equal 1"

    # Identify the start and end indices of each trajectory

    finished_indices = np.where(dataset.finished)[0]
    start_indices = np.where(np.roll(dataset.finished, 1))[0]

    # Pair up start and end indices of trajectories
    trajectories = list(zip(start_indices, finished_indices))

    # Shuffle the trajectories
    np.random.shuffle(trajectories)

    # Allocate trajectories to each set
    total_trajectories = len(trajectories)
    train_end = int(train_size * total_trajectories)
    test_end = train_end + int(test_size * total_trajectories)

    train_trajectories = trajectories[:train_end]
    test_trajectories = trajectories[train_end:test_end]
    val_trajectories = trajectories[test_end:]

    # Flatten the list of indices for each set
    train_indices = [idx for start, end in train_trajectories for idx in range(start, end + 1)]
    test_indices = [idx for start, end in test_trajectories for idx in range(start, end + 1)]
    val_indices = [idx for start, end in val_trajectories for idx in range(start, end + 1)]

    return np.array(train_indices), np.array(test_indices), np.array(val_indices)

def random_split_indices(dataset_length, train_size=0.7, test_size=0.2, val_size=0.1):
    # assert train_size + test_size + val_size == 1, "The sum of the sizes must equal 1"
    
    indices = np.arange(dataset_length)
    np.random.shuffle(indices)

    train_end = int(train_size * dataset_length)
    test_end = train_end + int(test_size * dataset_length)

    train_indices = indices[:train_end]
    test_indices = indices[train_end:test_end]
    val_indices = indices[test_end:]

    return train_indices, test_indices, val_indices

def model_split_indices(dataset, train_size=0.6, test_size=0.3, val_size = 0.1,  train_model_names=None, test_model_names=None, val_model_names=None):
    """
    Splits the dataset indices based on specified model names for test, validation, and training sets.

    :param dataset: The F110Dataset object.
    :param test_model_names: Array of model names to include in the test set.
    :param val_model_names: Array of model names to include in the validation set.
    :param train_model_names: Optional array of model names to include in the training set. If None, use the rest.
    :return: Indices for train_set, test_set, val_set.
    """
    if train_model_names is not None:
        assert test_model_names is not None, "If train_model_names is specified, test_model_names must also be specified."
        assert val_model_names is not None, "If train_model_names is specified, val_model_names must also be specified."
    if train_model_names is None:
        unique_model_names = np.unique(dataset.model_names)
        print("Available models:", unique_model_names)
        # select 30% models for training set
        train_model_names = np.random.choice(unique_model_names, int(train_size * len(unique_model_names)), replace=False)
        remaining_model_names = np.setdiff1d(unique_model_names, train_model_names)
        test_size_of_rem = test_size / (1 - train_size)
        test_model_names = np.random.choice(remaining_model_names, int(test_size_of_rem * len(remaining_model_names)), replace=False)
        val_model_names = np.setdiff1d(remaining_model_names, test_model_names)
        
    train_indices = []
    test_indices = []
    val_indices = []
    # 
    train_indices = np.where(np.isin(dataset.model_names,train_model_names))[0]
    test_indices = np.where(np.isin(dataset.model_names,test_model_names))[0]
    val_indices = np.where(np.isin(dataset.model_names,val_model_names))[0]

    assert len(train_indices) + len(test_indices) + len(val_indices) == len(dataset), "The sum of the sizes must equal the dataset length"

    return np.array(train_indices), np.array(test_indices), np.array(val_indices)

def family_split_indices(dataset, validation_family_name=None):
    unique_model_names = np.unique(dataset.model_names)
    validation_names = [ name for name in unique_model_names if validation_family_name in name]
    #print(validation_names)
    train_names = np.setdiff1d(unique_model_names, validation_names)
    # from the test_names sample 20 percent for validation
    val_size = 0.2
    test_names = np.random.choice(train_names, int(val_size * len(train_names)), replace=False)
    train_names = np.setdiff1d(train_names, test_names)

    
    test_indices = np.where(np.isin(dataset.model_names,test_names))[0]
    val_indices = np.where(np.isin(dataset.model_names,validation_names))[0]
    train_indices = np.where(np.isin(dataset.model_names,train_names))[0]
    assert len(train_indices) + len(test_indices) + len(val_indices) == len(dataset), "The sum of the sizes must equal the dataset length"
    return np.array(train_indices), np.array(test_indices), np.array(val_indices)

    

class F110DatasetSequence(F110Dataset):
    def __init__(self, *args, sequence_length=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequence_length = sequence_length

        self.valid_indices = self._precompute_valid_indices()
        self.model_names = np.array([self.model_names[idx] for idx in self.valid_indices])
        self._update_finished_array()


    def _precompute_valid_indices(self):
        """
        Precomputes valid indices for sequences that do not cross episode boundaries.
        """
        valid_indices = []
        for idx in range(len(self.states) - self.sequence_length + 1):
            # -1 because the last state is allowed to be terminal
            if not any(self.finished[idx:idx + self.sequence_length]): 
                valid_indices.append(idx)
        return valid_indices
    
    def _update_finished_array(self):
        """
        Updates the 'finished' array to reflect the end of valid sequences.
        """
        # Initialize a new 'finished' array with all False
        new_finished = np.zeros(len(self.model_names), dtype=bool)
        #plt.plot(self.finished)
        # Iterate through the valid indices
        for new_idx, idx in enumerate(self.valid_indices):
            # Mark the sequence as finished if its last state is finished
            assert (self.finished[idx : idx+ self.sequence_length] == False).all()
            if self.finished[idx + self.sequence_length]:
                new_finished[new_idx] = True
        #plt.plot(new_finished)
        #plt.show()
        self.finished = new_finished

    def __getitem__(self, idx):
        # Get the actual index from the precomputed valid indices
        actual_idx = self.valid_indices[idx]

        # Gather sequences for states, actions, and other data
        sequence_states = self.states[actual_idx:actual_idx + self.sequence_length]
        sequence_next_states = self.states_next[actual_idx:actual_idx + self.sequence_length]

        sequence_actions = self.actions[actual_idx:actual_idx + self.sequence_length]
        sequence_scans = self.scans[actual_idx:actual_idx + self.sequence_length]
        sequence_next_scans = self.scans_next[actual_idx:actual_idx + self.sequence_length]
        sequence_rewards = self.rewards[actual_idx:actual_idx + self.sequence_length]
        sequence_masks = self.masks[actual_idx:actual_idx + self.sequence_length]
        sequence_log_probs = self.log_probs[actual_idx:actual_idx + self.sequence_length]
        sequence_model_names = self.model_names[idx]
        return sequence_states,sequence_scans,sequence_actions, sequence_next_states,sequence_next_scans, sequence_rewards, sequence_masks,sequence_model_names, sequence_log_probs
        #return current_state, scan, action, next_state, next_scan, reward, mask, 1.0, log_prob 
    def __len__(self):
        return len(self.valid_indices)

if __name__ == "__main__":
    import f110_gym
    import f110_orl_dataset
    import gymnasium as gym
    F110Env = gym.make('f110-real-stoch-v1',
    # only terminals are available as of right now 
      **dict(name='f110-real-stoch-v1',
          config = dict(map="Infsaal2", num_agents=1,
          params=dict(vmin=0.0, vmax=5.0)),
            render_mode="human")
        )
    env = F110Env
    dataset = F110DatasetAutoregressive(
            env,
            normalize_rewards=True,
            normalize_states=True,
            sequence_length=5,
            only_agents = [], #['det'], #+ [FLAGS.target_policy] , #+ ["min_lida", "raceline"],
    )
    print(len(dataset.model_names))
    print(len(dataset))
    print(dataset.finished)
    print(np.sum(dataset.finished))
    # do some testing here