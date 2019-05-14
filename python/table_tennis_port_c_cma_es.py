import zmq
from c_cma_es import ContextualCmaEs
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.spatial import distance
import os
import math
import time
import json
import datetime

plt.rcParams.update({'font.size': 20})

'''
Hyper parameters:
'''
MAX_EPS = 20000      # Max episodes
STATE_DIM = 5       # Dimension of feature function of context s
ACTION_DIM = 37      # Dimension of parameters of action a


# Valid boundary for delta_t0, T, w
ACTION_BOUNDARY = np.array([[0.70, 1.1], [0.45, 0.65]])

SAMPLE_NUMBER = None

class TrttPortCCmaEs:
    def __init__(self, args):
        self.config_file = args.config_file

        self.uri = ""
        self.context_feature_type = ""
        self.baseline_feature_type = ""
        
        self.TARGET_COORDINATE = [0.35, -2.93, -0.99]
        self.COV_SCALE = 1
        self.STOP_THRESHOLD = 3.8
        self.INITIAL_TIME_VALUES = [0.90, 0.5]
        self.INITIAL_TIME_VARIANCE = [0.1, 0.1]

        self.prior_promp_mean = None
        self.prior_promp_cov = None

        self.load_config()

        self.initial_mean, self.initial_cov = self.generate_initial_values()

        self.ccmaes = ContextualCmaEs(state_dim=STATE_DIM, action_dim=ACTION_DIM, initial_action= self.initial_mean, initial_cov=self.initial_cov,sample_number=SAMPLE_NUMBER, cov_scale=self.COV_SCALE, context_feature_type='linear', baseline_feature_type='quadratic')

        self.openSocket()
        self.mainLoop()

        self.out_string = str("")
    def load_config(self):
        config_file = rel_path('../config/' + self.config_file)
        with open(config_file) as j_file:
            j_obj = json.load(j_file)
            self.uri = j_obj["uri"]
            self.context_feature_type = j_obj["context_feature_type"]
            self.baseline_feature_type = j_obj["baseline_feature_type"]
            self.TARGET_COORDINATE = j_obj["target_coordinate"]            
            self.COV_SCALE = j_obj["cov_scale"]
            self.STOP_THRESHOLD = j_obj["stop_threshold"]
            self.INITIAL_TIME_VALUES = j_obj["initial_time_values"]
            self.INITIAL_TIME_VARIANCE = j_obj["initial_time_variance"]
            
        prior_promp_file = rel_path('../config/prior_promp.json')        
        with open(prior_promp_file) as p_file:
            p_obj = json.load(p_file)
            self.prior_promp_mean = np.array(p_obj["model"]["mu_w"])
            self.prior_promp_cov = np.array(p_obj["model"]["Sigma_w"])                

    def generate_initial_values(self):
        initial_mean = np.append(self.prior_promp_mean, self.INITIAL_TIME_VALUES)
        initial_cov = np.block([[self.prior_promp_cov, np.zeros([35, 2])], [np.zeros([2, 35]), np.array([[self.INITIAL_TIME_VARIANCE[0], 0.0],[0.0, self.INITIAL_TIME_VARIANCE[1]]])]])
        return initial_mean, initial_cov


    def openSocket(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(self.uri)

    def get_ball_state(self):
        ball_obs_json = self.socket.recv_json()
        ball_state = ball_obs_json["ball_obs"]
        #print(ball_state)
        ball_state_array = np.array(ball_state)

        # y is constant and therefore be removed
        ball_state_array = np.delete(ball_state_array, 1)

        return ball_state_array

    def bound_clip(self, delta_t0, T):
        self.out_string += "\n\nAction before clip:\n====>    delta_t0: " + str(delta_t0) + "\n====>           T: " + str(T)

        if delta_t0 < ACTION_BOUNDARY[0][0]:
            delta_t0 = ACTION_BOUNDARY[0][0]
        if delta_t0 > ACTION_BOUNDARY[0][1]:
            delta_t0 = ACTION_BOUNDARY[0][1]

        if T < ACTION_BOUNDARY[1][0]:
            T = ACTION_BOUNDARY[1][0]
        if T > ACTION_BOUNDARY[1][1]:
            T = ACTION_BOUNDARY[1][1]

        self.out_string += "\n\nAction after clip:\n====>    delta_t0: " + str(delta_t0) + "\n====>           T: " + str(T) 
        return delta_t0, T

    def export_action(self, promp_mean, delta_t0, T):
        promp_mean = promp_mean.tolist()        
        
        action_json = {"promp_mean":promp_mean, "T": T, "delta_t0": delta_t0}
        self.socket.send_json(action_json)

    def get_reward_info(self):
        reward_info_json = self.socket.recv_json()
        landing_info = reward_info_json["landing_point"]
        ball_racket_dist_info = reward_info_json["min_distance"]
        hit_info = reward_info_json["hit"]
        return hit_info, landing_info, ball_racket_dist_info

    def generateReward(self, action, hit_info, landing_info, distance_info):
        """
            Reward contains the info about:
            1. the distance between sampled action to its valid range (if the sampled action is out of the valid range and therefore may lead danger to robot) 
            2. minimum distance between racket and ball if not hit
            3. the landing position to target landing position
            4. other stuff
        """

        reward = 0

        # Process action validity
        action_valid = True
        distance_to_valid_action = 0.0
        for counter in range(ACTION_BOUNDARY.shape[0]):
            if action[counter] < ACTION_BOUNDARY[counter][0]:
                action_valid = False
                distance_to_valid_action += distance.euclidean(
                    [action[counter]], [ACTION_BOUNDARY[counter][0]])
            elif action[counter] > ACTION_BOUNDARY[counter][1]:
                action_valid = False
                distance_to_valid_action += distance.euclidean(
                    [action[counter]], [ACTION_BOUNDARY[counter][0]])

        if False == action_valid:
            reward += -distance_to_valid_action
        else:  # Action is in valid range
            reward += 0

        # Process hitting and minimum distance between ball and racket
        hit = False
        # hit or not
        if True == action_valid and hit_info[0] < 0.0:
            hit = False
            reward += max([0, 1-distance_info[0]])
        elif True == action_valid and hit_info[0] > 0.0:
            reward += 1
            hit = True
        else:
            reward += 0
            hit = False

        # Process landing position
        if True == hit:
            # compute Euclidean distance in x and y coordinate space
            distance_to_target = distance.euclidean(
                landing_info[0:2], self.TARGET_COORDINATE[0:2])

            if distance_to_target <= 3.0:
                reward += -1 * pow(distance_to_target,1) + 3
            else:
                reward += 0
        else:
            reward += 0
        self.out_string += "\n\nReward:\n====>      reward: " + str(reward) + "\n"
        return reward

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
            #self.ccmaes.print_policy()
            temp_reward = 0.0
            for counter in range(N):
                internal_episode_counter, internal_update_counter = self.ccmaes.get_counters() 
                # get state
                state_info = self.get_ball_state()

                self.out_string = "\nEpisode: " + str(internal_episode_counter+1)                
                self.out_string += "\n\nBall State: " + str(state_info)                
                
                # get action
                action = self.ccmaes.generate_action(state_info)

                # bound clip action to make it valid for robot
                bound_action = self.bound_clip(action[-2], action[-1])

                self.export_action(
                    action[:-2], bound_action[0], bound_action[1])

                # receive reward info
                hit_info, landing_info, ball_racket_dist_info = self.get_reward_info()

                # generate reward
                reward = self.generateReward(
                    action[-2:], hit_info, landing_info, ball_racket_dist_info)
                temp_reward += reward

                # store
                self.ccmaes.store_episode(state_info, action, reward)
                
                print(self.out_string)

                if (internal_episode_counter + 1) % N != 0:
                    self.export_ok()
                external_episode_counter += 1
            
                   
            
            if temp_reward / N > self.STOP_THRESHOLD:
                print("\nTarget achieved in episode:", internal_episode_counter+1, ", update time: ", internal_update_counter+1, " !!!")
                break

            self.ccmaes.learn()

            rw_list = self.ccmaes.get_average_reward_list()
            plt.ion()
            plt.clf()
            ax1 = plt.subplot(1, 2, 1)
            x = np.arange(1,len(rw_list)+1) * N
            ax1.plot(x, rw_list)    
            ax1.set_xlabel("Episode number")
            ax1.set_ylabel("Average reward")
            ax1.set_xlim([0, len(rw_list)*N])
            ax2 = ax1.twiny()
            ax2.set_xlabel("Gereration number")
            ax2.set_xlim([0, len(rw_list)])
            
            en_list, overall_en_list = self.ccmaes.get_entropy_list()
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

            # When all stuff are done, send ok to continue the simulation
            self.export_ok()
            
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    def rel_path(fname):
        return os.path.join(os.path.dirname(__file__), fname)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config_file', help='json file of the config info')
    args = parser.parse_args()
    ccmaes_port = TrttPortCCmaEs(args)
