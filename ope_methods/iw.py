import numpy as np
import matplotlib.pyplot as plt

# fill starting with termination with the average of the previous values
def to_equal_length(log_diff, terminations):
    result_log_diff = log_diff.copy()
    for i in range(log_diff.shape[0]):
        # fill with the previous average value

        result_log_diff[i, int(terminations[i]):] = np.mean(log_diff[i, :int(terminations[i])])
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
                                     agent_name=""):
    
    #behavior_log = np.clip(behavior_log, -7, 2)
    #target_log = np.clip(target_log, -7, 2)
    #print(behavior_log.shape)
    #print(target_log.shape)
    log_diff = target_log - behavior_log
    #log_diff[log_diff > 0] = 0
    log_diff = to_equal_length(log_diff, terminations)
    # add the discount sum in the log space
    # 0.99**i
    # discount_sum = np.log(discount ** np.arange(behavior_log.shape[1]))
    #print(np.min(terminations))
    #plt.plot(log_diff[10])
    #plt.plot(log_diff[0])
    #plt.show()
    # handle early terminations, by using the average
    

    # for each model plot the log_diff
    
    
    log_cumsum = np.cumsum(log_diff, axis=1)
    #for i in range(log_cumsum.shape[0]):
    #    plt.plot(log_cumsum[i])
    #plt.show()
    # colors like np.unique number
    #print("======")
    colors = plt.cm.tab20(np.linspace(0, 1, len(np.unique(behavior_agent_names))))
    unique_agents = np.unique(behavior_agent_names)
    color_map = {agent: colors[i] for i, agent in enumerate(unique_agents)}
    if plot:
        plt.figure(figsize=(15, 12), dpi=100)

        for ag_idx, agent in enumerate(unique_agents):
            agents_idxs = np.where(behavior_agent_names == str(agent))[0]
            for agent_idx in agents_idxs:
                if terminations[agent_idx] < 25:
                    continue
                plt.plot(log_cumsum[agent_idx, :], color=color_map[agent]) # :int(terminations[agent_idx])

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

            continue
        else:
            if len(np.where(distances < start_distance)[0]) < 10:
                print(f"Found {len(np.where(distances < start_distance)[0])} starting points")
        # all starting points that are closer than start distance are considered
        close_points_idx = np.where(distances < start_distance)[0]
        #print("Close points", len(close_points_idx))
        # for each of the close point find we dont need the check
        max_log_sum = -np.inf
        for point in close_points_idx:
            #print(point)
            #plt.plot(log_cumsum[point], label=behavior_agent_names[point])
            if log_cumsum[point,-1] > max_log_sum and terminations[point] > 25:
                max_log_sum = log_cumsum[point,-1]
                max_log_sum_point = point
        #plt.legend()
        #plt.show()

        #print(max_log_sum)
        #print(max_log_sum_point)
        #plt.plot(rewards[max_log_sum_point])
        #plt.show()
        discounted_reward = np.sum(rewards[max_log_sum_point] * discount ** np.arange(rewards[max_log_sum_point].shape[0]))
        #print(discounted_reward)
        num_rewards += 1
        reward += discounted_reward
        picked_agents.append(behavior_agent_names[max_log_sum_point])
        # print("closest agent is:", behavior_agent_names[max_log_sum_point])
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

        
        


    