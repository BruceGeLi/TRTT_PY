import zmq
from c_cma_es import ContextualCmaEs
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import numpy as np
import scipy as sp
from scipy.spatial import distance
import os
import math
import time
import json

# Set font size of matplotlib figure
plt.rcParams.update({'font.size': 20})
plt.rcParams["font.family"] = "Times New Roman"
np.set_printoptions(precision=3)


# Max episodes
MAX_EPS = 100000      


# Valid boundary for w, delta_t0, and T
ACTION_BOUNDARY = np.array([[-2.5, 2.5], [0.70, 1.1], [0.45, 0.65]])

# If heuristic sample number is not used, then use the value below
SAMPLE_NUMBER = None


class TrttPortCCmaEs:
    def __init__(self, args):
        self.config = args.config
        self.STATE_DIM = 6       # Dimension of context s
        self.ACTION_DIM = 8      # Dimension of parameters of action a

        self.context_feature_type = ""
        self.baseline_feature_type = ""

        self.TARGET_COORDINATE = [0.35, -2.93, -0.99]
        self.COV_SCALE = 1
        self.STOP_THRESHOLD = 3.8
        self.INITIAL_TIME_VALUES = [0.90, 0.5]
        self.INITIAL_TIME_VARIANCE = [0.1, 0.1]
        self.OPPONENT_COURT_COORDINATE = None

        self.prior_promp_mean = None
        self.prior_promp_cov = None
        self.training_mode = 1

        self.save_file = None
        self.MAX_REWARD = None
        if None != args.save_file:
            self.save_file = rel_path(args.save_file)
        self.save_update_counter = args.save_update_counter

        self.load_file = None
        if None != args.load_file:
            self.load_file = rel_path(args.load_file)
        
        self.collect_file = None
        if None != args.collect_file:
            self.collect_file = rel_path(args.collect_file)            
            self.collected_reward_info = list()

        self.load_config()
        assert 0 <= self.training_mode <= 4, "Wrong training mode!"

        self.initial_mean, self.initial_cov = self.generate_initial_values()

        self.out_string = str("")
        self.openSocket()

        #test functions for debug usage
        #self.test_bound_function()
        #self.test_export_action()
        #self.test_reward_function()

        self.ccmaes = None
        if 1 != self.training_mode:
            self.ccmaes = ContextualCmaEs(state_dim=self.STATE_DIM, action_dim=self.ACTION_DIM, initial_action=self.initial_mean, initial_cov=self.initial_cov,
                                      sample_number=SAMPLE_NUMBER, cov_scale=self.COV_SCALE, context_feature_type='linear', baseline_feature_type='quadratic', save_file=self.save_file, save_update_counter=self.save_update_counter, load_file=self.load_file)
            self.mainLoop()
        else:
            # Dummy RL usage to only record the reward info 
            self.dummyMainLoop()





    def load_config(self):
        config = rel_path('../config/' + self.config)
        with open(config) as j_file:
            j_obj = json.load(j_file)
            self.training_mode = j_obj["training_mode"]
            self.context_feature_type = j_obj["context_feature_type"]
            self.baseline_feature_type = j_obj["baseline_feature_type"]
            self.TARGET_COORDINATE = j_obj["target_coordinate"]
            self.COV_SCALE = j_obj["cov_scale"]
            self.STOP_THRESHOLD = j_obj["stop_threshold"]
            self.INITIAL_TIME_VALUES = j_obj["initial_time_values"]
            self.INITIAL_TIME_VARIANCE = j_obj["initial_time_variance"]
            self.OPPONENT_COURT_COORDINATE = j_obj["opponent_court"]
            self.MAX_REWARD = j_obj["max_reward"]

        prior_promp_file = rel_path('../config/prior_promp.json')
        with open(prior_promp_file) as p_file:
            p_obj = json.load(p_file)
            self.prior_promp_mean = np.array(p_obj["model"]["mu_w"])
            self.prior_promp_cov = np.array(p_obj["model"]["Sigma_w"])

    def generate_initial_values(self):
        print("\n==============================================================")
        print("\nC-CMA-ES starts with Training mode", self.training_mode, ".")
        print(
            "\nPolicy intput parameters:\n Ball's 6 dimension state: [x, y, z, vx, vy, vz].")
        initial_mean = None
        initial_cov = None
        if 1 == self.training_mode:
            print("\nPolicy output parameters:\n None.")
            pass
        elif 2 == self.training_mode:
            print("\nPolicy output parameters:\n delta_t0 and T.")
            print("===> [delta_t0, T].")
            initial_mean = np.array(self.INITIAL_TIME_VALUES)
            initial_cov = np.eye(2) * 0.02
            self.ACTION_DIM = 2
            self.STATE_DIM = 6

        elif 3 == self.training_mode:
            print(
                "\nPolicy output parameters:\n Last DOF ProMP weight mean, delta_t0 and T.")
            print("===> [w7, delta_t0, T]")
            initial_mean = np.append(
            self.prior_promp_mean[33], self.INITIAL_TIME_VALUES)
            initial_cov = np.eye(3) * 0.02
            self.ACTION_DIM = 3
            self.STATE_DIM = 6

        elif 4 == self.training_mode:
            print(
                "\nPolicy output parameters:\n 7 DOF ProMP weights mean, delta_t0 and T.")
            print("===> [w1, w2, w3, w4, w5, w6, w7, delta_t0, T]")
            initial_mean = np.append(
            self.prior_promp_mean[3::5], self.INITIAL_TIME_VALUES)
            initial_cov = np.eye(9) * 0.02
            self.ACTION_DIM = 9
            self.STATE_DIM = 6

        else:
            assert False, "Wrong training mode!"
        print("\nInitial mean vector: \n", initial_mean)
        print("\nInitial covariance matrix: \n", initial_cov)
        print("\n==============================================================\n\n")

        return initial_mean, initial_cov

    def openSocket(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        try:
            self.socket.bind("tcp://*:8181")
            print("\n===> C-CMA-ES Connects to PONG!!!\n")
        except:
            self.socket.bind("tcp://*:8182")
            print("\n===> C-CMA-ES Connects to PING!!!\n")

    def get_ball_state(self):
        ball_obs_json = self.socket.recv_json()
        ball_state = ball_obs_json["ball_obs"]
        # print(ball_state)
        ball_state_array = np.array(ball_state)

        # y is constant and therefore be removed
        # ball_state_array = np.delete(ball_state_array, 1)

        return ball_state_array

    def bound_clip(self, action):
        if 1 == self.training_mode:
            self.out_string += "Training mode 1, no bound clip."
            pass
        elif 2 == self.training_mode:
            self.out_string += "\n\nAction before clip:\n====>    delta_t0: " + \
                str(action[0]) + "\n====>           T: " + str(action[1])
            action[0] = self.bound_detla_t0(action[0])
            action[1] = self.bound_T(action[1])
            self.out_string += "\n\nAction after clip:\n====>    delta_t0: " + \
                str(action[0]) + "\n====>           T: " + str(action[1])

        elif 3 == self.training_mode:
            self.out_string += "\n\nAction before clip:" + "\n====>           w: " +\
                                   str(action[0]) + \
                               "\n====>    delta_t0: " + \
                str(action[1]) + "\n====>           T: " + str(action[2])
            action[0] = self.bound_DOF(action[0])
            action[1] = self.bound_detla_t0(action[1])
            action[2] = self.bound_T(action[2])
            self.out_string += "\n\nAction after clip:" + "\n====>           w: " +\
                                   str(action[0]) + \
                               "\n====>    delta_t0: " + \
                str(action[1]) + "\n====>           T: " + str(action[2])

        elif 4 == self.training_mode:
            self.out_string += "\n\nAction before clip:" + "\n====>           w: " +\
                                   str(action[0:7]) + \
                               "\n====>    delta_t0: " + \
                str(action[7]) + "\n====>           T: " + str(action[8])
            for i in range(7):
                action[i] = self.bound_DOF(action[i])              
            action[7] = self.bound_detla_t0(action[7])
            action[8] = self.bound_T(action[8])
            self.out_string += "\n\nAction before clip:" + "\n====>           w: " +\
                                   str(action[0:7]) + \
                               "\n====>    delta_t0: " + \
                str(action[7]) + "\n====>           T: " + str(action[8])

        else:
            assert False, "Wrong training mode!"    

        return action

    def bound_DOF(self, w):
        if w<ACTION_BOUNDARY[0][0]:            
            w = ACTION_BOUNDARY[0][0]
        if w > ACTION_BOUNDARY[0][1]:            
            w = ACTION_BOUNDARY[0][1]
        return w

    def distance_DOF(self,w):
        distance = 0.0
        if w<ACTION_BOUNDARY[0][0]:
            distance = abs(w - ACTION_BOUNDARY[0][0])            
        if w > ACTION_BOUNDARY[0][1]:
            distance = abs(w - ACTION_BOUNDARY[0][1])
        return distance

    def bound_detla_t0(self,delta_t0):    
        if delta_t0 < ACTION_BOUNDARY[1][0]:
            delta_t0 = ACTION_BOUNDARY[1][0]
        if delta_t0 > ACTION_BOUNDARY[1][1]:    
            delta_t0 = ACTION_BOUNDARY[1][1]
        return delta_t0

    def distance_delta_t0(self, delta_t0):
        distance = 0.0
        if delta_t0 < ACTION_BOUNDARY[1][0]:
            distance = abs(ACTION_BOUNDARY[1][0] - delta_t0)        
        if delta_t0 > ACTION_BOUNDARY[1][1]:
            distance = abs(delta_t0 - ACTION_BOUNDARY[1][1])
        return distance
        
    def bound_T(self, T):        
        if T < ACTION_BOUNDARY[2][0]:    
            T = ACTION_BOUNDARY[2][0]
        if T > ACTION_BOUNDARY[2][1]:
            T = ACTION_BOUNDARY[2][1]        
        return T 

    def distance_T(self, T):
        distance = 0.0
        if T < ACTION_BOUNDARY[2][0]:
            distance = abs(ACTION_BOUNDARY[2][0] - T)            
        if T > ACTION_BOUNDARY[2][1]:
            distance = abs(T - ACTION_BOUNDARY[2][1])            
        return distance

    def test_bound_function(self):
        self.training_mode = 1
        self.bound_clip(np.array([1,2]))
        self.training_mode = 2
        self.bound_clip(np.array([1.0, 0.8]))
        self.bound_clip(np.array([-1.2, 0.3]))
        self.training_mode = 3
        self.bound_clip(np.array([2.6, 1.0, 0.8]))
        self.bound_clip(np.array([-2.6, -1.2, 0.3]))
        self.training_mode = 4
        self.bound_clip(np.array([2.6,2.6,2.6,2.6,2.6,2.6,2.6, 1.0, 0.8]))
        self.bound_clip(np.array([-2.6,2.6,2.6,2.6,2.6,2.6,-2.6, -1.2, 0.3]))
        print(self.out_string)
        assert False, "exit peacefully from test clip function!"

    def export_action(self, action):
        action_json = None
        if 1 == self.training_mode:
            action_json = {"dummy": 1.0}
        elif 2 == self.training_mode:
            action_json = {"delta_t0": action[0], "T": action[1]}
        elif 3 == self.training_mode:
            action_json = {"promp_mean": [action[0],], "delta_t0": action[1], "T": action[2]}
        elif 4 == self.training_mode:            
            action_json = {"promp_mean": action[0:7].tolist(), "delta_t0": action[7], "T": action[8]}
        else:
            assert False, "Wrong training mode!"
        
        #return action_json for         
        
        self.socket.send_json(action_json)    

    def test_export_action(self):
        self.training_mode = 1
        print(self.export_action(np.array([1,2])))
        self.training_mode = 2
        print(self.export_action(np.array([1.0, 0.8])))        
        self.training_mode = 3
        print(self.export_action(np.array([2.6, 1.0, 0.8])))
        self.training_mode = 4
        print(self.export_action(np.array([2.6,2.6,2.6,2.6,2.6,2.6,2.6, 1.0, 0.8])))            
        assert False, "exit peacefully from test export action function!"

    def get_reward_info(self):
        reward_info_json = self.socket.recv_json()
        landing_info = reward_info_json["landing_point"]
        ball_racket_dist_info = reward_info_json["min_distance"]
        hit_info = reward_info_json["hit"]
        return hit_info, landing_info, ball_racket_dist_info

    def collect_reward_info(self, hit_info, landing_info, reward):
        print("colect data!")
        info = dict()        
        if hit_info[0] < 0.0:
            info["hit"] = False
        else:
            info["hit"] = True
        info["landing"] = landing_info
        info["reward"] = reward
        self.collected_reward_info.append(info)
        if len(self.collected_reward_info) >= 100:
            filename = self.collect_file
            with open(filename, "w") as f_obj:
                json.dump(self.collected_reward_info, f_obj)
            assert False, "Data has benn collected!"


    def generateReward(self, action, hit_info, landing_info, distance_info):
        """
            Reward contains the info about:
            1. the distance between sampled action to its valid range (if the sampled action is out of the valid range and therefore may lead danger to robot) 
            2. minimum distance between racket and ball if not hit
            3. landing to opponent's court or not?
            4. the landing position to target landing position
            5. other stuff
        """
        reward = 0.0

        # Process action validity
        action_valid = True
        distance_to_valid_action = 0.0

        if 1 == self.training_mode:
            pass
        elif 2 == self.training_mode:
            distance_to_valid_action = self.distance_delta_t0(action[0]) + self.distance_T(action[1])    
        elif 3 == self.training_mode:
            distance_to_valid_action = self.distance_DOF(action[0]) + self.distance_delta_t0(action[1]) + self.distance_T(action[2])    
        elif 4 == self.training_mode:
            for i in range(7):
                distance_to_valid_action += self.distance_DOF(action[i])             
            distance_to_valid_action += self.distance_delta_t0(action[7]) + self.distance_T(action[8])    
        else:
            assert False, "Wrong training mode!"

        if distance_to_valid_action > 0.001:
            reward += -distance_to_valid_action
            self.out_string += "\n\nReward Info:\n====> action valid: False\n"
            self.out_string += "\n\nReward:\n====>      reward: " + \
            str(reward) + "\n"
            return reward
        else:  # Action is in valid range
            reward += 0                
            self.out_string += "\n\nReward Info:\n====> action valid: True\n"
        
        # Process hitting and minimum distance between ball and racket            
        # hit or not
        if hit_info[0] < 0.0:            
            reward += max([0, 1-distance_info[0]])
            self.out_string += "\n====>         hit: False\n"
            self.out_string += "\n\nReward:\n====>      reward: " + \
            str(reward) + "\n"
            return reward

        else:
            reward += 1            
            self.out_string += "\n====>         hit: True\n"        

        # Process opponent's court?
        land_opponent_court = True
        distance_to_opponent_court = [0.0, 0.0]
        # land on opponent's side or not?            
        if landing_info[0] < self.OPPONENT_COURT_COORDINATE[0]:
            land_opponent_court = False
            distance_to_opponent_court[0] = self.OPPONENT_COURT_COORDINATE[0] - landing_info[0]
        elif landing_info[0] > self.OPPONENT_COURT_COORDINATE[1]:
            land_opponent_court = False
            distance_to_opponent_court[0] = landing_info[0] - self.OPPONENT_COURT_COORDINATE[1]
        else:
            pass
        
        if landing_info[1] < self.OPPONENT_COURT_COORDINATE[2]:
            land_opponent_court = False
            distance_to_opponent_court[1] = self.OPPONENT_COURT_COORDINATE[2] - landing_info[1]
            
        elif landing_info[1] > self.OPPONENT_COURT_COORDINATE[3]:
            land_opponent_court = False
            distance_to_opponent_court[1] = landing_info[1] - self.OPPONENT_COURT_COORDINATE[3]
        else:
            pass
        
        if True == land_opponent_court:
            reward += 2
            self.out_string += "\n====> hit opponent: True\n"      
        else:                
            reward += 2 - min([pow(distance_to_opponent_court[0], 2)+ pow  (distance_to_opponent_court[1], 2), 2.0])      
            self.out_string += "\n====> hit opponent: False\n"      
            self.out_string += "\n\nReward:\n====>      reward: " + \
            str(reward) + "\n"
            return reward                

        # Process landing position        
        # compute Euclidean distance in x and y coordinate space
        distance_to_target = distance.euclidean(
            landing_info[0:2], self.TARGET_COORDINATE[0:2])
        reward += 2.231 - pow(distance_to_target, 2)
                
        self.out_string += "\n\nReward:\n====>      reward: " + \
            str(reward) + "\n"
            
        return reward

    def test_reward_function(self):        

        xs = np.linspace(-2.5, 2.5, 101)
        ys = np.linspace(-5.0, 0.0, 101)
        rs = np.zeros((101, 101))
        
        self.training_mode = 1
        print(self.generateReward([1,2], [1.0], [-0.76, -1.93], [0.1]))
        self.training_mode = 2    
        print(self.generateReward([0.81, 0.41], [1.0], [-0.76, -1.93], [0.1]))
        self.training_mode = 3
        print(self.generateReward([2.6, 1.0, 0.8], [1.0], [-0.76, -1.93], [0.1]))
        self.training_mode = 4        
        print(self.generateReward([-2.6,2.6,2.6,2.6,2.6,2.6,-2.6, -1.2, 0.3], [1.0], [-0.76, -1.93], [0.1]))                

        
        self.training_mode =2
        for i, x in enumerate(xs):
            print(i, x )
            for j, y in enumerate(ys):
                rs[j,i] = self.generateReward([0.81, 0.51], [1.0], [x,y], [0.1])
         
        xs, ys = np.meshgrid(xs, ys)
        levels = np.linspace(1.0, self.MAX_REWARD+0.08, 51)
        """        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(xs, ys, rs,cmap=cm.coolwarm,
                       linewidth=0, antialiased=False,alpha=0.3)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('r')
        
        """
        plt.figure()
        plt.contourf(xs, ys, rs, levels, cmap=cm.plasma)
        plt.xlabel("x")
        plt.ylabel("y")        
        plt.colorbar(format='%.2f')
        
        #plt.show()
        plt.savefig(rel_path("../figure/reward.png"), dpi=300, bbox_inches='tight')
        assert False,"exit test reward function peacefully!"

    def export_ok(self):
        policy_updated_json = {"policy_ready": True}
        self.socket.send_json(policy_updated_json)

    def mainLoop(self):
        external_episode_counter = 0

        rw_list = list()
        en_list = list()
        overall_en_list = list()
        N = self.ccmaes.get_recommand_sample_number()
        plt.figure(figsize=(22, 10), dpi=80)

        while external_episode_counter < MAX_EPS:
            # self.ccmaes.print_policy()
            temp_reward = 0.0
            for counter in range(N):
                internal_episode_counter, internal_update_counter = self.ccmaes.get_counters()
                # get state
                state_info = self.get_ball_state()

                self.out_string = "\n===============================\n\nEpisode: " + \
                    str(internal_episode_counter+1)
                self.out_string += "\n\nBall State: " + str(state_info)

                # get action
                action = self.ccmaes.generate_action(state_info)

                bound_action = np.array(action)
                # bound clip action to make it valid for robot                
                bound_action = self.bound_clip(bound_action)                

                self.export_action(bound_action)

                # receive reward info
                hit_info, landing_info, ball_racket_dist_info = self.get_reward_info()

                # generate reward
                reward = self.generateReward(
                    action, hit_info, landing_info, ball_racket_dist_info)

                if None != self.collect_file:
                    self.collect_reward_info(hit_info, landing_info, reward)

                temp_reward += reward

                # store
                self.ccmaes.store_episode(state_info, action, reward)

                print(self.out_string)

                if (internal_episode_counter + 1) % N != 0:
                    self.export_ok()
                external_episode_counter += 1

            if temp_reward / N > self.STOP_THRESHOLD:
                print("\nTarget achieved in episode:", internal_episode_counter +
                      1, ", update time: ", internal_update_counter+1, " !!!")
                break

            self.ccmaes.learn()

            rw_list = self.ccmaes.get_average_reward_list()
            plt.ion()
            plt.clf()
            ax1 = plt.subplot(1, 2, 1)
            x = np.arange(1, len(rw_list)+1) * N
            ax1.plot(x, rw_list)
            ax1.set_xlabel("Episode number")
            ax1.set_ylabel("Average reward")
            ax1.set_xlim([0, len(rw_list)*N])
            ax2 = ax1.twiny()
            ax2.set_xlabel("Generation number")
            ax2.set_xlim([0, len(rw_list)])

            en_list, overall_en_list = self.ccmaes.get_entropy_list()
            ax3 = plt.subplot(1, 2, 2)
            ax3.plot(x, en_list, x, overall_en_list)
            ax3.legend(["Entropy", "Overall Entropy"])
            ax3.set_xlabel("Episode number")
            ax3.set_ylabel("Entropy of distribution")
            ax3.set_xlim([0, len(rw_list)*N])
            ax4 = ax3.twiny()
            ax4.set_xlabel("Generation number")
            ax4.set_xlim([0, len(rw_list)])
            plt.pause(0.1)
            if 0 == (internal_update_counter+1) % 20:
                plt.savefig(rel_path('../figure/part_promp_May16_update' +
                                     str(internal_update_counter+1)), dpi=100, format='png')

            # When all stuff are done, send ok to continue the simulation
            self.export_ok()

        plt.ioff()
        plt.show()
    
    def dummyMainLoop(self):
        external_episode_counter = 0

        rw_list = list()                

        while external_episode_counter < MAX_EPS:
            temp_reward = 0.0
            
            # get state
            state_info = self.get_ball_state()
            self.out_string = "\nEpisode: " + \
                str(external_episode_counter+1)
            self.out_string += "\n\nBall State: " + str(state_info)
            
            # export dummy action
            self.export_action(None)
            # receive reward info
            hit_info, landing_info, ball_racket_dist_info = self.get_reward_info()
            # generate reward
            reward = self.generateReward(
                None, hit_info, landing_info, ball_racket_dist_info)
            
            if None != self.collect_file:
                self.collect_reward_info(hit_info, landing_info, reward)
            
            temp_reward += reward
            print(self.out_string)
            
            self.export_ok()
            external_episode_counter += 1


if __name__ == "__main__":
    def rel_path(fname):
        return os.path.join(os.path.dirname(__file__), fname)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='training_config.json', help='json file of the config info')
    parser.add_argument('--save_file', default=None, help='file to save training parameters')
    parser.add_argument('--save_update_counter', type=int, default=10, help='save parameters after each M updates.')
    parser.add_argument('--load_file', default=None, help='file to load training parameters')
    parser.add_argument('--collect_file', default=None, help='collect 100 episodes data for analysis.')
    args = parser.parse_args()
    ccmaes_port = TrttPortCCmaEs(args)
