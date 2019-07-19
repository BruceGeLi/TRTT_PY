from c_cma_es import ContextualCmaEs
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
plt.rcParams.update({'font.size': 30})
plt.rcParams["font.family"] = "Times New Roman"
# Hyper parameters
MAX_EPS = 30000      # Max episodes

STATE_DIM = 6     # Dimension of context s
REPEAT = 5
# Average reward to stop the training
STOP_THRESHOLD = -1e-2

ACTION_DIM = 9      # Dimension of parameters of action a
GENERATION = 100
total_rw_array = np.zeros((REPEAT, GENERATION))


def rel_path(fname):
    return os.path.join(os.path.dirname(__file__), fname)
t1 = datetime.datetime.now()
for repeat in range(REPEAT):
    np.random.seed((int)(time.time()))

    # G matrix from normal distribution
    G = np.random.normal(0.0, 1.0, (ACTION_DIM, STATE_DIM))

    # initial action
    INITIAL_ACTION = np.random.normal(0.0, 1.0, (ACTION_DIM))
    
    # create policy
    ccmaes = ContextualCmaEs(state_dim=STATE_DIM, action_dim=ACTION_DIM, initial_action=INITIAL_ACTION,
                         context_feature_type='linear', baseline_feature_type='linear', save_file=None, load_file=None)                         
    """
    ccmaes = ContextualCmaEs(state_dim=STATE_DIM, action_dim=ACTION_DIM, initial_action=INITIAL_ACTION,
                         context_feature_type='linear', baseline_feature_type='linear', save_file='/tmp/temp_save', load_file=None)
    """

    # initial counters
    external_episode_counter = 0

    # reward_list
    rw_list = list()
    en_list = list()
    overall_en_list = list()
    # get recommanded sample number
    N = ccmaes.get_recommand_sample_number()
    
    while external_episode_counter < N * GENERATION:
        ccmaes.print_policy()
        temp_reward = 0.0
        for counter in range(N):
            internal_episode_counter, internal_update_counter = ccmaes.get_counters()
            print("\nEpisode: ", internal_episode_counter+1)
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
            external_episode_counter += 1
        """
        if temp_reward / N > STOP_THRESHOLD:
            internal_update_counter+1
            break
        """
        ccmaes.learn()
    rw_list = ccmaes.get_average_reward_list()
    total_rw_array[repeat] = rw_list
    


    
    """
    rw_list = ccmaes.get_average_reward_list()
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


    en_list, overall_en_list = ccmaes.get_entropy_list()
    
    ax3 = plt.subplot(1, 2, 2)    
    ax3.plot(x, en_list, x, overall_en_list)
    ax3.legend(["Entropy", "Overall Entropy"])
    ax3.set_xlabel("Episode number")
    ax3.set_ylabel("Entropy of distribution")
    ax3.set_xlim([0, len(rw_list)*N])
    ax4 = ax3.twiny()
    ax4.set_xlabel("Gereration number")    
    ax4.set_xlim([0, len(rw_list)])    

    plt.pause(0.1)
    #plt.savefig("/tmp/temp_fig"+str(internal_update_counter), dpi=100,format='png')
    """
t2 = datetime.datetime.now()
t3 = t2-t1
print(t3)


std_rw = np.std(total_rw_array, axis=0)
average_rw = np.average(total_rw_array, axis=0)

upper = average_rw + std_rw
lower = average_rw - std_rw

plt.figure(figsize=(12, 9), dpi=80)
plt.plot(np.arange(1, GENERATION+1), average_rw, color='b', linewidth=3)
plt.fill_between(np.arange(1, GENERATION+1), upper, lower, alpha=0.4)
plt.xlabel("Generation number")
plt.ylabel("Average return")
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#ax = plt.gca()
#ax.set_yscale('log')
plt.grid()
plt.legend(["Mean","Standard deviation"], loc='lower right')
plt.savefig("../figure/rosenbrock_baseline.png", bbox_inches='tight')
#plt.show()
