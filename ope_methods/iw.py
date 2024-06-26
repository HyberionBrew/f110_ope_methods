import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
})
sns.set_theme(style="whitegrid")
# fill starting with termination with the average of the previous values
def to_equal_length_mean(log_diff, terminations, inital_skips=15, fill_value=-np.inf):
    result_log_diff = log_diff.copy()
    for i in range(log_diff.shape[0]):
        # fill with the previous average value
        if len(log_diff[i, inital_skips :int(terminations[i])+1]) <= 0:
            result_log_diff[i, int(terminations[i]) + 1:] = fill_value
        else:
            result_log_diff[i, int(terminations[i]) + 1:] = np.mean(log_diff[i, inital_skips :int(terminations[i])+1])
        assert (np.isnan(result_log_diff[i]) == False).all()
    return result_log_diff

def to_equal_length_value(log_diff, terminations, value):
    result_log_diff = log_diff.copy()
    for i in range(log_diff.shape[0]):
        # fill with the previous average value

        result_log_diff[i, int(terminations[i]) + 1:] = value #np.mean(log_diff[i, :int(terminations[i])])
        #plt.plot(log_diff[i])
        #plt.plot(result_log_diff[i])
        #print(terminations[i])
        
        #plt.show()

    return result_log_diff

def ImportanceSamplingContinousStart(behavior_log, target_log, behavior_agent_names,
                                     terminations, rewards,
                                     starts_,starts_eval_, start_distance,
                                     start_prob_method = "l2",
                                     discount=0.99,
                                     plot=False,
                                     agent_name="",
                                     iw_type="wis",
                                     fill_type = "mean",
                                     get_actions=None,
                                     model=None,
                                     start_scans=None,
                                     normalize_states=None,
                                     ):
    """
    Terminations are inclusive
    """

    rewards = to_equal_length_value(rewards, terminations, 0.0)

    log_diff = target_log - behavior_log
    # assert no nans in logdiff
    assert (np.isnan(log_diff) == False).all()


    if fill_type=="mean":
        log_diff = to_equal_length_mean(log_diff, terminations, 15)
    if fill_type=="zero":
        log_diff = to_equal_length_value(log_diff, terminations, 0.0)
    if fill_type =="minus_inf":
        log_diff = to_equal_length_value(log_diff, terminations, -np.inf)

    assert (np.isnan(log_diff) == False).all()

    log_cumsum = np.cumsum(log_diff, axis=1)
   
    colors =  plt.cm.tab20(np.linspace(0, 1, len(np.unique(behavior_agent_names)))) #plt.cm.tab20(np.linspace(0, 1, len(np.unique(behavior_agent_names))))
    unique_agents = np.unique(behavior_agent_names)
    color_map = {agent: colors[i] for i, agent in enumerate(unique_agents)}
    if False: # used for creating plots in the thesis
        plt.figure() #figsize=(15, 10))
        break_at = 10
       
        agents_plot = np.unique(behavior_agent_names)
        agents_plot = ["pure_pursuit2_0.8_0.85_raceline3_0.3_0.5",
                        "pure_pursuit2_0.8_0.8_raceline2_0.3_0.5",
                        "pure_pursuit2_0.7_1.1_raceline3_0.3_0.5",
                        "StochasticContinousFTGAgent_0.45_3_0.5_0.03_0.1_5.0_0.3_0.5",
                        #"StochasticContinousFTGAgent_0.48_7_0.5_0.03_0.1_5.0_0.3_0.5",
                        #"StochasticContinousFTGAgent_0.7_5_0.5_0.02_0.1_5.0_0.3_0.5",
                        # "pure_pursuit2_1.2_1.1_raceline3_0.3_0.5",
                        #"pure_pursuit2_0.5_0.5_raceline5_0.3_0.5",
                        #"pure_pursuit2_0.5_0.6_raceline3_0.3_0.5",
                        "pure_pursuit2_0.7_1.0_raceline2_0.3_0.5"
                        ]
        colors =  plt.cm.tab20(np.linspace(0, 1, len(np.unique(agents_plot))))
        color_map = {agent: colors[i] for i, agent in enumerate(np.unique(agents_plot))}
        for ag_idx, agent in enumerate(reversed(agents_plot)):
            print(agent)
            agents_idxs = np.where(behavior_agent_names == str(agent))[0]
            for agent_idx in agents_idxs:
                #if terminations[agent_idx] < 25:
                #    continue
                # Exponentiate the log cumulative sum values
                #exp_cumsum = np.exp(log_cumsum[agent_idx, :])
                #plt.plot(log_diff[:int(terminations[agent_idx]+1)], color=color_map[agent], alpha=0.4) #
                # plot a histogram of the log_diff
                plt.hist(log_diff[agent_idx, :int(terminations[agent_idx]+1)], bins=100, color=color_map[agent], alpha=0.4)
                # plot a line for the mean and std deviation across x axis 
                plt.axvline(np.mean(log_diff[agent_idx, :int(terminations[agent_idx]+1)]), color=color_map[agent], linestyle="--")
                #plt.axvline(np.mean(log_diff[agent_idx, :int(terminations[agent_idx]+1)]) + np.std(log_diff[agent_idx, :int(terminations[agent_idx]+1)]), color=color_map[agent], linestyle="--", alpha=0.6)
                #plt.axvline(np.mean(log_diff[agent_idx, :int(terminations[agent_idx]+1)]) - np.std(log_diff[agent_idx, :int(terminations[agent_idx]+1)]), color=color_map[agent], linestyle="--", alpha=0.6)
        # x-axis is the log-prob
        plt.xlabel("Log Probability")
        plt.ylabel("Frequency")
        plt.title(f"Log Probability Distribution for target agent\n {agent_name}")
        plt.show()
        for ag_idx, agent in enumerate(agents_plot):
            print(agent)
            agents_idxs = np.where(behavior_agent_names == str(agent))[0]
            for agent_idx in agents_idxs:
                #if terminations[agent_idx] < 25:
                #    continue
                # Exponentiate the log cumulative sum values
                exp_cumsum = np.exp(log_cumsum[agent_idx, :])
                plt.plot(exp_cumsum[:int(terminations[agent_idx]+1)], color=color_map[agent], alpha=0.4) # Use exp_cumsum for plotting
            #if ag_idx == break_at:
            #    break
        for ag_idx, agent in enumerate(agents_plot):
            agents_idxs = np.where(behavior_agent_names == str(agent))[0]
            for agent_idx in agents_idxs:
                if terminations[agent_idx] - 1 < 250:
                    #print("==")
                    exp_cumsum = np.exp(log_cumsum[agent_idx, :])
                    # make the marker in the color of the colormap

                    plt.plot(terminations[agent_idx], exp_cumsum[int(terminations[agent_idx])], color=color_map[agent], marker="x")
                             #"rx")
                    if "StochasticContinousFTGAgent" in agent:
                        # plot it larger and in black
                        if terminations[agent_idx] - 1 < 15:
                            # make the marker thiker
                            plt.plot(terminations[agent_idx], exp_cumsum[int(terminations[agent_idx])], color=color_map[agent], marker="x", markersize=20, markeredgewidth=5)
            #if ag_idx == break_at:
            #    break
        # Set y-axis to logarithmic scale
        plt.yscale('log')

        # Handling the legend in a more seaborn-integrated manner
        # Extracting the labels and corresponding colors in the correct order
        labels = list(color_map.keys())
        handles = [plt.Rectangle((0,0),1,1, color=color_map[label]) for label in labels]
        #plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), title="Agents")

        # plt.ylim(-1200, 100)  # You may need to adjust this based on the exponentiated values

        plt.xlabel("Episode Timestep t")
        plt.ylabel("$p_{1:t}$")
        #plt.subplots_adjust(right=0.7)  # Adjust as necessary to fit the legend outside the plot
        plt.title(f"Cumulative importance-sampling ratio vs. timesteps for \n {agent_name}")
        # save as pdf
        plt.savefig(f"iw_prob_plots/{agent_name}_cumulative_importance_sampling_ratio.pdf")
        plt.show()
        colors =  plt.cm.tab20(np.linspace(0, 1, len(np.unique(behavior_agent_names))))
        color_map = {agent: colors[i] for i, agent in enumerate(np.unique(behavior_agent_names))}
        plt.figure(figsize=(15, 12), dpi=100)

        for ag_idx, agent in enumerate(unique_agents):
            agents_idxs = np.where(behavior_agent_names == str(agent))[0]
            for agent_idx in agents_idxs:
                #if terminations[agent_idx] < 25:
                #    continue
                plt.plot(log_cumsum[agent_idx, :], color=color_map[agent]) # :int(terminations[agent_idx])
        for ag_idx, agent in enumerate(unique_agents):
            agents_idxs = np.where(behavior_agent_names == str(agent))[0]
            for agent_idx in agents_idxs:
                plt.plot(terminations[agent_idx], log_cumsum[agent_idx, int(terminations[agent_idx])], "rx")
        # Handling the legend
        # Extracting the labels and corresponding colors in the correct order
        labels = list(color_map.keys())
        handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[label]) for label in labels]
        plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), title="Agents")
        # y (0, -1200)
        plt.ylim(-1200, 100)
        # add x-axis label
        plt.xlabel("Episode Timesteps")
        plt.ylabel("Cumulative Log Importance Weight")
        plt.subplots_adjust(right=0.7)  # Adjust as necessary to fit the legend outside the plot
        plt.title(f"Log Importance Weight for {agent_name}")
        plt.show()
        #plt.savefig(f"iw_prob_plots/{agent_name}_log_importance_weight.png")
        
        plt.figure(figsize=(15, 12), dpi=100)
        for ag_idx, agent in enumerate(unique_agents):
            agents_idxs = np.where(behavior_agent_names == str(agent))[0]
            for agent_idx in agents_idxs:
                if terminations[agent_idx] < 25:
                    continue
                # clip logdiff to be < 1
                
                logdiff_clip = log_diff.copy()
                logdiff_clip = np.exp(logdiff_clip) 
                #logdiff_clip[logdiff_clip > 1] = 1
                #logdiff_clip[logdiff_clip > 0] = 0
                plt.plot(np.cumprod(logdiff_clip[agent_idx, :]), color=color_map[agent])
        labels = list(color_map.keys())
        handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[label]) for label in labels]
        plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), title="Agents")
        # add x-axis label
        plt.xlabel("Episode Timesteps")
        plt.ylabel("Cumulative Log Importance Weight (EXP)")
        plt.subplots_adjust(right=0.7)  # Adjust as necessary to fit the legend outside the plot
        plt.title(f"EXP Log Importance Weight for {agent_name}")
        #plt.savefig(f"iw_prob_plots/{agent_name}_exp_log_importance_weight.png")
        plt.show()
        
        
    # since we have very large sums of logprobs, the weighted importance sampling 
    # simply degenerates into being ~1 at the closest trajectory and zero elsewhere.
    starts_eval = starts_eval_[:, :2] # only need x and y
    starts = starts_[:, :2]
    # plot all starts 
    #plt.scatter(starts[:, 0], starts[:, 1])
    #plt.scatter(starts_eval[:, 0], starts_eval[:, 1])
    #plt.legend(["Training Starts", "Evaluation Starts"])
    #plt.show()
    reward = 0
    num_rewards = 0
    picked_agents = []

    for starting_idx, start in enumerate(starts_eval):
        # compute the distance to all starting points
        distances = np.linalg.norm(starts - start, axis=1)
        # print the closest 10 starts 
        # print(np.argsort(distances))
        # if there are less than 5 close starting points, we drop it
        if len(np.where(distances < start_distance)[0])<5:
            # this never happens in our dataset
            #plt.scatter(starts[:, 0], starts[:, 1])
            #plt.scatter(starts_eval[:, 0], starts_eval[:, 1])
            #plt.scatter(start[0], start[1], color="red")
            #plt.legend(["Training Starts", "Evaluation Starts", "in question"])
            #plt.show()
            raise ValueError("Not enough close starting points") 
        else:
            if len(np.where(distances < start_distance)[0]) < 10:
                print(f"Found {len(np.where(distances < start_distance)[0])} starting points")
        # all starting points that are closer than start distance are considered
        close_points_idx = np.where(distances < start_distance)[0]
        
        if iw_type == "step_wis":
            is_fn = step_wis
        elif iw_type == "step_wis_termination": # this is TPDWIS
            is_fn = step_wis_termination
        elif iw_type == "wis_termination":
            pass
        elif iw_type == "simple_is":
            is_fn = simple_is
        elif iw_type == "simple_step_is":
            is_fn = simple_step_is
        elif iw_type == "cobs_wis":
            is_fn = cobs_wis
        elif iw_type == "phwis":
            is_fn = phwis
        elif iw_type == "phwis_heuristic":
            is_fn = phwis_heuristic
        else:
            raise NotImplementedError("Unknown importance sampling type")
        

        discounted_reward = is_fn(log_diff[close_points_idx], terminations[close_points_idx], rewards[close_points_idx], discount)
        
        if model is not None and start_scans is not None and get_actions is not None:
            complete_offset = 0

            samples = 20

            for i in range(samples):
                normed_starts = normalize_states(torch.tensor(starts_[close_points_idx]))
                actions_eval = get_actions(normed_starts, 
                                            scans = torch.tensor(start_scans[close_points_idx]), 
                                            deterministic=False)
                eval_starts = normalize_states(torch.tensor(starts_eval_[starting_idx]).unsqueeze(0))

                offset, std_offset = model.estimate_returns(normed_starts.cuda(), torch.tensor(actions_eval).cuda())

                complete_offset += offset.cpu().detach().numpy()
            discounted_reward += complete_offset / samples
        reward += discounted_reward
        num_rewards += 1


    reward = reward / num_rewards
    return reward

"""
Start of the actual IS method implementations
Each method has the following parameters, with N being the number of trajectories and T the number of timesteps:
@param log_diff: the log difference between the target and behavior policy, shape (N, T)
@param terminations: the integer termination timesteps, shape (N,)
@param rewards: the rewards, shape (N,T)
@param discount: the discount factor, float
"""

def phwis_heuristic(log_diff, terminations, rewards, discount):
    return phwis(log_diff, terminations, rewards, discount, heuristic=True)

def phwis(log_diff, terminations, rewards, discount, heuristic=False):
    # is simply one extended
    log_diff = log_diff.astype(np.longfloat)
    # set logdiff to zero for all timesteps after termination (corresponds to one extended)
    log_diff = to_equal_length_value(log_diff, terminations, 0.0)
    # loop over timesteps
    prob_cumprod = np.exp(np.cumsum(log_diff, axis=1))
    reward = 0.0
    """
    for r in range(rewards.shape[0]):
        # skip if no termination
        if terminations[r] > 100:
            continue
        plt.plot(rewards[r])
        # plot a 10 at the termination
        plt.plot(terminations[r], rewards[r, int(terminations[r])], "ro")
    plt.show()
    """
    #raise ValueError("Not implemented, properly, check the t t+1")
    all_reward = np.zeros((log_diff.shape[1]))
    #print(max(terminations))
    all_bitmask = 0
    for t in range(log_diff.shape[1]):
        # select all that terminate at this timestep
        # compute Wl
        bit_maks_term = terminations == t
        all_bitmask += sum(bit_maks_term)
        Wl_nominator = np.sum(prob_cumprod[bit_maks_term, t])
        if heuristic and t > 0:
            Wl_nominator = np.sum(prob_cumprod[bit_maks_term, t])**(1/(t))
        Wl_denominator = np.sum(np.sort(prob_cumprod[:, t]))
        Wl = Wl_nominator / Wl_denominator        
        # inner sum
        w_nominator_weights = prob_cumprod[bit_maks_term, t]
        w_nominator_rewards = np.sum(rewards[bit_maks_term, :t] * discount ** np.arange(t), axis=1)
        w_nominator = np.sum(w_nominator_weights * w_nominator_rewards)
        if prob_cumprod[bit_maks_term, t].shape[0] == 0:
            continue
        w_denominator = np.sum(prob_cumprod[bit_maks_term, t])
        w = w_nominator / w_denominator
        reward += Wl * w
        all_reward[t] = Wl * w

    assert all_bitmask == len(terminations)
    #plt.plot(all_reward)
    
    #plt.show()
    
    return reward.astype(np.float32)

def step_wis_termination(log_diff, terminations, rewards, discount):
    # cast to float 64
    #for i in range(log_diff.shape[0]):
    #    plt.plot(log_diff[i])
    
    #print(terminations)
    #plt.show()

    log_diff = log_diff.astype(np.longfloat)
    # print(log_diff.shape)
    log_cumsum = np.cumsum(log_diff, axis=1)
    #print(log_cumsum.dtype)
    prob_cumprod = np.exp(log_cumsum)

    
    trajectory_probs = np.zeros((prob_cumprod.shape[0],prob_cumprod.shape[1]))
    acc_rewards = np.zeros(prob_cumprod.shape[1]) # like timesteps
    # print(acc_rewards.shape)
    remaining_prob = 1.0
    remaining_trajectory_idxs = np.arange(prob_cumprod.shape[0])

    for timestep in range(prob_cumprod.shape[1]):
        weight = np.sum(np.sort(prob_cumprod[remaining_trajectory_idxs,timestep]))

        for trajectory_idx in remaining_trajectory_idxs:
            # for each timestep, calculate the probability            
            log_prob_traj = np.log(prob_cumprod[trajectory_idx, timestep]) - np.log(weight)

            assert (np.isnan(log_prob_traj) == False).all()
            trajectory_probs[trajectory_idx, timestep] = np.exp(log_prob_traj) * remaining_prob
            acc_rewards[timestep] += trajectory_probs[trajectory_idx, timestep] * rewards[trajectory_idx, timestep] * discount**timestep
        term_prob = 0.0
        for trajectory_idx in remaining_trajectory_idxs:
            if timestep == terminations[trajectory_idx]:
                #print("timestep", timestep)
                term_prob += trajectory_probs[trajectory_idx, timestep]
                remaining_trajectory_idxs = remaining_trajectory_idxs[remaining_trajectory_idxs != trajectory_idx]
        remaining_prob = remaining_prob * (1 - term_prob)
        if remaining_prob < 0.0 or np.isclose(remaining_prob, 0.0):
            break

    #plt.plot(acc_rewards)
    #plt.title("Accumulated Rewards")
    #plt.show()
    return np.sum(acc_rewards).astype(np.float32)

def simple_is(log_diff, terminations, rewards, discount):
    log_cumsum = np.cumsum(log_diff, axis=1)
    prob = np.zeros(log_diff.shape[0])
    for i in range(log_diff.shape[0]):
        prob[i] = np.exp(log_cumsum[i, terminations[i].astype(int)])
    acc_rewards = rewards * discount ** np.arange(log_diff.shape[1])

    terms = (terminations+1).astype(int)
    acc_reward = np.zeros((log_diff.shape[0]))
    for i, term in enumerate(terms):
        acc_reward[i] = np.sum(acc_rewards[i, :term])
    return np.nanmean(acc_reward * prob).astype(np.float32)

def simple_step_is(log_diff, terminations, rewards, discount):
    log_cumsum = np.cumsum(log_diff, axis=1)
    acc_rewards = rewards * discount ** np.arange(log_diff.shape[1])
    acc_rewards = acc_rewards * np.exp(log_cumsum)
    terms = (terminations+1).astype(int)
    acc_reward_term = np.zeros((log_diff.shape[0]))
    for i, term in enumerate(terms):
        acc_reward_term[i] = np.sum(acc_rewards[i, :term])
    # acc_rewards = np.sum(acc_rewards, axis=1)
    return np.nanmean(acc_reward_term).astype(np.float32)

# compare https://github.com/clvoloshin/COBS/blob/master/ope/algos/traditional_is.py
# ignore already dones
def cobs_wis(log_diff, terminations, rewards, discount):

    log_diff = log_diff.astype(np.longfloat)
    log_cumsum = np.cumsum(log_diff, axis=1)
    #print(log_cumsum[np.where(terminations==4)[0]])
    #for i in range(log_diff.shape[0]):
    #    plt.plot(log_cumsum[i])
    #    plt.plot(terminations[i], log_cumsum[i, int(terminations[i])], "ro")
        #print(log_cumsum[i, terminations[i]])
    
    
    prob_cumprod = np.exp(log_cumsum)
    probs = np.zeros(log_diff.shape[0])
    for i in range(log_diff.shape[0]):
        probs[i] = prob_cumprod[i, terminations[i].astype(int)]
    #print(probs[np.where(terminations==4)[0]])
    #print(probs.shape)
    #plt.show()
    acc_rewards = rewards * discount ** np.arange(log_diff.shape[1])
    rewards_sum = np.zeros(log_diff.shape[0])
    for i in range(log_diff.shape[0]):
        rewards_sum[i] = np.sum(acc_rewards[i, :terminations[i].astype(int)+1])
    probs = probs / np.sum(probs)
    #print(probs)
    #print(np.sum(probs))
    #print("--------")
    return np.sum(probs * rewards_sum).astype(np.float32)

def wis_extended(log_diff, terminations, rewards, discount):
    log_diff = log_diff.astype(np.longfloat)
    log_cumsum = np.cumsum(log_diff, axis=1)
    prob_cumprod = np.exp(log_cumsum)
    probs = prob_cumprod[:, -1]
    acc_rewards = rewards * discount ** np.arange(log_diff.shape[1])
    rewards_sum = np.zeros(log_diff.shape[0])
    for i in range(log_diff.shape[0]):
        rewards_sum[i] = np.sum(acc_rewards[i, :terminations[i].astype(int)+1])
    probs = probs / np.sum(probs)
    return np.sum(probs * rewards_sum).astype(np.float32)

def step_wis(log_diff, terminations, rewards, discount):
    # cast to float 64
    log_diff = log_diff.astype(np.longfloat)
    log_cumsum = np.cumsum(log_diff, axis=1)
    prob_cumprod = np.exp(log_cumsum)

    
    trajectory_probs = np.zeros((prob_cumprod.shape[0],prob_cumprod.shape[1]))
    acc_rewards = np.zeros(prob_cumprod.shape[1]) # like timesteps
    # print(acc_rewards.shape)
    for timestep in range(prob_cumprod.shape[1]):
        weight = np.sum(np.sort(prob_cumprod[:,timestep]))
        for trajectory_idx in range(prob_cumprod.shape[0]):
            # for each timestep, calculate the probability
            if timestep > terminations[trajectory_idx]:
                continue
            
            log_prob_traj = np.log(prob_cumprod[trajectory_idx, timestep]) - np.log(weight)
            assert (np.isnan(log_prob_traj) == False).all()
            trajectory_probs[trajectory_idx, timestep] = np.exp(log_prob_traj)
            acc_rewards[timestep] += trajectory_probs[trajectory_idx, timestep] * rewards[trajectory_idx, timestep] * discount**timestep
    # here we return the sum, normalization already happened with the weights
    return np.sum(acc_rewards).astype(np.float32)