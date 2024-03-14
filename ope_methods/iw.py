import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
        # no nan#
        #print(terminations[i])
        #print(log_diff[i])

        #print(result_log_diff[i])
        #print(inital_skips)
        #print(int(terminations[i]) + 1)
        #print(log_diff[i, inital_skips :int(terminations[i])+1])
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
                                     starts,starts_eval, start_distance,
                                     start_prob_method = "l2",
                                     discount=0.99,
                                     plot=False,
                                     agent_name="",
                                     iw_type="wis",
                                     fill_type = "mean"):
    """
    Terminations are inclusive
    """
    
    #behavior_log = np.clip(behavior_log, -7, 2)
    #target_log = np.clip(target_log, -7, 2)
    #print(behavior_log.shape)
    #print(target_log.shape)

    log_diff = target_log - behavior_log
    # assert no nans in logdiff
    assert (np.isnan(log_diff) == False).all()

    #log_diff[log_diff > 0] = 0
    # log_diff = to_equal_length_value(log_diff, terminations, 0.0)
    if fill_type=="mean":
        log_diff = to_equal_length_mean(log_diff, terminations, 15)
    if fill_type=="zero":
        log_diff = to_equal_length_value(log_diff, terminations, 0.0)
    if fill_type =="minus_inf":
        log_diff = to_equal_length_value(log_diff, terminations, -np.inf)
    # no nans
    assert (np.isnan(log_diff) == False).all()
    # print first 10 terminations
    # print(terminations[:10])
    log_cumsum = np.cumsum(log_diff, axis=1)
    """
    for i in range(10):
      
        plt.plot(log_cumsum[i])
        plt.plot(terminations[i], log_cumsum[i, int(terminations[i])], "ro")
        print(log_cumsum[i, terminations[i]])
    plt.show()
    """
    #exit()
    # add the discount sum in the log space
    # 0.99**i
    # discount_sum = np.log(discount ** np.arange(behavior_log.shape[1]))
    #print(np.min(terminations))
    #plt.plot(log_diff[10])
    #plt.plot(log_diff[0])
    #plt.show()
    # handle early terminations, by using the average
    

    # for each model plot the log_diff
    
    
    
    #for i in range(log_cumsum.shape[0]):
    #    plt.plot(log_cumsum[i])
    #plt.show()
    # colors like np.unique number
    #print("======")
    colors =  plt.cm.tab20(np.linspace(0, 1, len(np.unique(behavior_agent_names)))) #plt.cm.tab20(np.linspace(0, 1, len(np.unique(behavior_agent_names))))
    unique_agents = np.unique(behavior_agent_names)
    color_map = {agent: colors[i] for i, agent in enumerate(unique_agents)}
    if False:
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
    starts_eval = starts_eval[:, :2] # only need x and yq
    starts = starts[:, :2]
    # plot all starts 
    #plt.scatter(starts[:, 0], starts[:, 1])
    #plt.scatter(starts_eval[:, 0], starts_eval[:, 1])
    #plt.legend(["Training Starts", "Evaluation Starts"])
    #plt.show()
    # we have multiple start cluster, assert that in each start cluster there is the same number of starting points
    reward = 0
    num_rewards = 0
    picked_agents = []
    for start in starts_eval:
        # compute the distance to all starting points
        distances = np.linalg.norm(starts - start, axis=1)
        # print the closest 10 starts 
        #print(np.argsort(distances))
        # if there are less than 5 close starting points, we drop it
        if len(np.where(distances < start_distance)[0])<5:
            #print(np.sum(distances < start_distance))
            #print("Smallest distances", np.sort(distances)[:10])
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
        #print("Close points", len(close_points_idx))
        # for each of the close point find we dont need the check
        if iw_type == "step_wis":
            is_fn = step_wis
        elif iw_type == "step_wis_termination":
            is_fn = step_wis_termination
        elif iw_type == "wis_termination":
            pass
            # is_fn = wis_termination
        elif iw_type == "simple_is":
            is_fn = simple_is
        elif iw_type == "simple_step_is":
            is_fn = simple_step_is
        elif iw_type == "cobs_wis":
            is_fn = cobs_wis
        else:
            raise NotImplementedError("Unknown importance sampling type")
        

        discounted_reward = is_fn(log_diff[close_points_idx], terminations[close_points_idx], rewards[close_points_idx], discount)
        

        #print(max_log_sum)
        #print(max_log_sum_point)
        #plt.plot(rewards[max_log_sum_point])
        #plt.show()
        # discounted_reward = 1#np.sum(rewards[max_log_sum_point] * discount ** np.arange(rewards[max_log_sum_point].shape[0]))
        #print(discounted_reward)
        
        reward += discounted_reward
        num_rewards += 1
        #picked_agents.append(behavior_agent_names[max_log_sum_point])
        # print("closest agent is:", behavior_agent_names[max_log_sum_point])
    # print(reward)

    reward = reward / num_rewards
    #print(f"Reward for agent: {reward}")
    print(f"Agents: {np.unique(np.array(picked_agents), return_counts=True)}")
    return reward

        # plot the closest 10 starts
        # plot all within the start distance
        #plt.scatter(starts[distances < start_distance, 0], starts[distances < start_distance, 1])
        # plt.scatter(starts[np.argsort(distances)[:10], 0], starts[np.argsort(distances)[:10], 1])
        # plot the current start
        #plt.scatter(start[0], start[1], color="red")
        #plt.show()
"""
# this implementation does n
def step_wis(log_diff, terminations, rewards, discount, fill_value=0.0):
    # cast to float 64
    log_diff = log_diff.astype(np.longfloat)
    prob_cumprod = np.exp(np.cumsum(to_equal_length_value(log_diff, terminations, fill_value), axis=1))
    ws = np.sum(prob_cumprod, axis=0)
    #print(log_cumsum.dtype)
    assert np.isclose(np.sum(prob_cumprod/ws, axis=0).all(),1.0)
    acc_rewards = np.zeros(prob_cumprod.shape[1]) # like timesteps
    for trajectory_idx in range(prob_cumprod.shape[0]):
            # for each timestep, calculate the probability
            print(prob_cumprod[trajectory_idx]/ws[trajectory_idx])
            print(prob_cumprod[trajectory_idx])
            print(ws[trajectory_idx])
            acc_rewards[:int(terminations[trajectory_idx]+1)] += (prob_cumprod[trajectory_idx]/ws[trajectory_idx])[:int(terminations[trajectory_idx] + 1)] * rewards[trajectory_idx, :int(terminations[trajectory_idx] + 1)] * discount ** np.arange(int(terminations[trajectory_idx] + 1))
    plt.plot(acc_rewards)
    plt.show()
    return np.sum(acc_rewards)
"""
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



"""
def wis(log_diff, terminations, rewards, discount):
    log_diff = log_diff.astype(np.longfloat)
    # print(log_diff.shape)
    log_cumsum = np.cumsum(log_diff, axis=1)
    #print(log_cumsum.dtype)
    prob_cumprod = np.exp(log_cumsum)
    prob = np.zeros(log_diff.shape[0])
    for trajectory_idx in range(prob_cumprod.shape[0]):
        # for each timestep, calculate the probability
        weight = np.sum(np.sort(prob_cumprod[:, int(terminations[trajectory_idx])]))
        prob[trajectory_idx] = prob_cumprod[trajectory_idx, int(terminations[trajectory_idx])] / weight
"""
"""
def wis(log_diff, terminations, rewards, discount):
    # we have the log_cumsum, the terminations and the rewards
    
    rewards_sum = np.zeros(log_diff.shape[0])
    # Create 2 subplots: one for log_cumsum and another for probabilities
    fig, axs = plt.subplots(4, 1, figsize=(10, 8)) # 2 rows, 1 column. Adjust figsize as needed
    log_cumsum = np.cumsum(log_diff, axis=1)
    print(log_cumsum)
    # Plot log_cumsum up to termination points in the first subplot
    for i in range(log_diff.shape[0]):
        axs[0].plot(log_cumsum[i,:int(terminations[i])], color="blue")
        rewards_sum[i] = np.sum(rewards[i, :int(terminations[i] + 1)] * discount ** np.arange(int(terminations[i] + 1)))

    # Calculate probabilities
    
    normalization_weight = np.zeros(log_cumsum.shape[0])
    probs = np.zeros(log_diff.shape[0])
    trajectory_prob = np.zeros(log_diff.shape[0])
    for i in range(log_diff.shape[0]):

        probs[i] = np.exp(log_cumsum[i, int(terminations[i])])
        # there is at least one trajectory which has a non zero probability, before floating point errors
        assert(log_cumsum[:, int(terminations[i])].any() != - np.inf)
        normalization_weight[i] = np.sum(np.sort(np.exp(log_cumsum[:, int(terminations[i])]))) / log_diff.shape[0]
        assert(normalization_weight[i] != 0)
        trajectory_prob[i] = probs[i]/normalization_weight[i]
        print(trajectory_prob[i])
    # axs[3].plot(prob/normalization_weight[:, int(terminations[i])-1])
    # filter all zeros from prob and the same indicies from normalization_weight

    axs[2].plot(normalization_weight) 
    print(normalization_weight.shape)
    #norm_prob = np.array([ prob / normalization_weight[int(terminations[i])-1] for i, prob in enumerate(probs)])
    norm_prob = probs/normalization_weight 
    largest_prob = np.max(norm_prob)
    # print(f"Largest probability: {largest_prob}")
    print(f"Largest probabilities: {np.sort(norm_prob)[::-1][:4]}")
    print("Indices of the 5 largest probabilities: ", np.argsort(norm_prob)[::-1][:5])
    print(np.where(norm_prob > largest_prob * 0.1)[0])
    
    for i in np.where(norm_prob > largest_prob * 0.1)[0]:
        axs[0].plot(log_cumsum[i, :int(terminations[i])], color="orange")

    #for i in np.argsort(norm_prob)[::-1][:5]:
    #    axs[0].plot(log_cumsum[i, :int(terminations[i])], color="red")
    #    print(f"Probability: {norm_prob[i]}, prob: {prob[i]}, norm: {normalization_weight[int(terminations[i])-1]}")
    
    # Plot the probabilities in the second subplot
    axs[1].bar(np.arange(len(probs)), probs, color='green')
    axs[1].set_title("Probabilities at Termination Points")
    axs[1].set_xlabel("Sequence Index")
    axs[1].set_ylabel("Probability")

    # Setting titles for clarity
    axs[0].set_title("Log Cumulative Sum up to Termination")
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Log Cumsum")

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show plot
    plt.show()
"""