from c_cma_es import ContextualCmaEs
import numpy as np
import matplotlib.pyplot as plt
import time

plt.rcParams.update({'font.size': 20})

# Hyper parameters
MAX_EPS = 500000000      # Max episodes

STATE_DIM = 5      # Dimension of context s

# Average reward to stop the training
STOP_THRESHOLD = -1e-10

ACTION_DIM = 3      # Dimension of parameters of action a

# G matrix from normal distribution
G = np.random.normal(0.0, 1.0, (ACTION_DIM, STATE_DIM))

# initial action
INITIAL_ACTION = np.random.normal(0.0, 1.0, (ACTION_DIM))

# create policy
ccmaes = ContextualCmaEs(state_dim=STATE_DIM, action_dim=ACTION_DIM, initial_action=INITIAL_ACTION,
                         context_feature_type='linear', baseline_feature_type='linear')

# initial counters
episode_counter = 0
update_counter = 1

# reward_list
rw_list = list()
en_list = list()
# get recommanded sample number
N = ccmaes.get_recommand_sample_number()
plt.figure(figsize=(18, 8), dpi=80)


while episode_counter < MAX_EPS:
    ccmaes.print_policy()
    temp_reward = 0.0
    for counter in range(N):
        print("\nEpisode: ", episode_counter+1)
        # get state
        state = np.random.uniform(1.0, 2.0, STATE_DIM)
        print("\nState: ", state)

        # get action
        action = ccmaes.generate_action(state)
        print("\n\nAction:\n====>    delta_t0: ", action[0],
              "\n====>           T: ", action[1], "\n====>           w: ", action[2])

        # get reward
        reward = 0
        x = action + np.matmul(G, state)

        # Sphere case:
        #reward += -np.matmul(x, x)

        # Rosenbrock case:
        for i in range(ACTION_DIM-1):
           reward += -(100 * pow(x[i+1]-pow(x[i],2),2)+pow(1-x[i],2))
        print("\n====>      reward: ", reward, "\n")
        temp_reward += reward

        # store
        ccmaes.store_episode(state, action, reward)
        episode_counter += 1

    rw_list.append(temp_reward / N)

    if temp_reward / N > STOP_THRESHOLD:
        print("\nTarget achieved! in episode ", episode_counter)
        """       
        rw_list = rw_list[-50:]
        x = range(len(rw_list))
        plt.plot(x, rw_list)
        plt.show()
        """
        break

    plt.ion()
    plt.clf()
    ax1 = plt.subplot(1, 2, 1)
    x = np.arange(1, len(rw_list)+1) * N
    ax1.plot(x, rw_list)    
    ax1.set_xlabel("Episode number")
    ax1.set_ylabel("Average reward")
    ax1.set_xlim([0, len(rw_list)*N])
    ax2 = ax1.twiny()
    ax2.set_xlabel("Gereration number")
    ax2.set_xlim([0, len(rw_list)])


    ccmaes.learn()
    en_list.append(ccmaes.get_entropy())
    
    ax3 = plt.subplot(1, 2, 2)    
    ax3.plot(x, en_list)
    ax3.set_xlabel("Episode number")
    ax3.set_ylabel("Entropy of distribution")
    ax3.set_xlim([0, len(rw_list)*N])
    ax4 = ax3.twiny()
    ax4.set_xlabel("Gereration number")
    ax4.set_xlim([0, len(rw_list)])    


    plt.pause(0.1)
    update_counter += 1
plt.ioff()
plt.show()
