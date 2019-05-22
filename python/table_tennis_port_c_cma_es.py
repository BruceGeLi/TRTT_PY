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
import datetime

plt.rcParams.update({'font.size': 20})
np.set_printoptions(precision=2)
'''
Hyper parameters:
'''
MAX_EPS = 50000      # Max episodes
STATE_DIM = 5       # Dimension of feature function of context s
ACTION_DIM = 8      # Dimension of parameters of action a


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
        self.OPPONENT_COURT_COORDINATE = None

        self.prior_promp_mean = None
        self.prior_promp_cov = None
        self.load_config()

        self.initial_mean, self.initial_cov = self.generate_initial_values()

        self.ccmaes = ContextualCmaEs(state_dim=STATE_DIM, action_dim=ACTION_DIM, initial_action=self.initial_mean, initial_cov=self.initial_cov,
                                      sample_number=SAMPLE_NUMBER, cov_scale=self.COV_SCALE, context_feature_type='linear', baseline_feature_type='quadratic', save_file=rel_path('../data/temp_save_May_16_update'), save_update_counter=10, load_file=rel_path('../data/temp_save_May_16_update_20.npz'))

        self.out_string = str("")
        self.test_reward_function()        
        
        self.openSocket()
        self.mainLoop()


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
            self.OPPONENT_COURT_COORDINATE = j_obj["opponent_court"]

        prior_promp_file = rel_path('../config/prior_promp.json')
        with open(prior_promp_file) as p_file:
            p_obj = json.load(p_file)
            self.prior_promp_mean = np.array(p_obj["model"]["mu_w"])
            self.prior_promp_cov = np.array(p_obj["model"]["Sigma_w"])

    def generate_initial_values(self):
        initial_mean = np.append(
            self.prior_promp_mean[3::5], self.INITIAL_TIME_VALUES[0])
        print(initial_mean)
        """
        temp_cov = np.zeros((7, 7))+2e-2
        
        index_list = np.arange(3, 35, 5)

        for new_index, old_index in enumerate(index_list, 0):
            temp_cov[new_index][new_index] = self.prior_promp_cov[old_index][old_index]
            for i in range(new_index+1, 7):
                temp_cov[new_index][i] = self.prior_promp_cov[old_index][index_list[i]]
                temp_cov[i][new_index] = self.prior_promp_cov[index_list[i]][old_index]
        
        initial_cov = np.block([[temp_cov, np.zeros([7, 1])], [
                               np.zeros([1, 7]), self.INITIAL_TIME_VARIANCE[0]]])
        """
        initial_cov = np.eye(8) * 0.02
        print(initial_cov)

        return initial_mean, initial_cov

    def openSocket(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(self.uri)

    def get_ball_state(self):
        ball_obs_json = self.socket.recv_json()
        ball_state = ball_obs_json["ball_obs"]
        # print(ball_state)
        ball_state_array = np.array(ball_state)

        # y is constant and therefore be removed
        ball_state_array = np.delete(ball_state_array, 1)

        return ball_state_array

    def bound_clip(self, delta_t0, T=None):
        if T is None:
            self.out_string += "\n\nAction before clip:\n====>    delta_t0: " + \
                str(delta_t0)
        else:
            self.out_string += "\n\nAction before clip:\n====>    delta_t0: " + \
                str(delta_t0) + "\n====>           T: " + str(T)

        if delta_t0 < ACTION_BOUNDARY[0][0]:
            delta_t0 = ACTION_BOUNDARY[0][0]
        if delta_t0 > ACTION_BOUNDARY[0][1]:
            delta_t0 = ACTION_BOUNDARY[0][1]

        if T is not None:
            if T < ACTION_BOUNDARY[1][0]:
                T = ACTION_BOUNDARY[1][0]
            if T > ACTION_BOUNDARY[1][1]:
                T = ACTION_BOUNDARY[1][1]

        if T is None:
            self.out_string += "\n\nAction after clip:\n====>    delta_t0: " + \
                str(delta_t0)
            return delta_t0
        else:
            self.out_string += "\n\nAction after clip:\n====>    delta_t0: " + \
                str(delta_t0) + "\n====>           T: " + str(T)
            return delta_t0, T

    def export_action(self, promp_mean, delta_t0, T=None):
        promp_mean = promp_mean.tolist()
        if T is None:
            action_json = {"promp_mean": promp_mean,
                           "T": 0.45, "delta_t0": delta_t0}
        else:
            action_json = {"promp_mean": promp_mean,
                           "T": T, "delta_t0": delta_t0}
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
            3. landing to opponent's court or not?
            4. the landing position to target landing position
            5. other stuff
        """

        reward = 0.0

        # Process action validity
        action_valid = True
        distance_to_valid_action = 0.0

        #print(action)

        #ACTION_BOUNDARY = np.array([[0.70, 1.1], [0.45, 0.65]])

        """
        for counter in range(ACTION_BOUNDARY.shape[0]):        
            if action[counter] < ACTION_BOUNDARY[counter][0]:
                action_valid = False
                distance_to_valid_action += distance.euclidean(
                    [action[counter]], [ACTION_BOUNDARY[counter][0]])
            elif action[counter] > ACTION_BOUNDARY[counter][1]:
                action_valid = False
                distance_to_valid_action += distance.euclidean(
                    [action[counter]], [ACTION_BOUNDARY[counter][0]])
        """
        if action < 0.70:
            action_valid = False
            distance_to_valid_action = distance.euclidean([action], [0.7])
        elif action > 1.1:
            action_valid = False
            distance_to_valid_action = distance.euclidean([action], [1.1])

        if False == action_valid:
            reward += -distance_to_valid_action
        else:  # Action is in valid range
            reward += 0
        
        # Process hitting and minimum distance between ball and racket
        hit = True
        
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
        
        # Process opponent's court?
        land_opponent_court = True
        distance_to_opponent_court = [0.0, 0.0]
        # land on opponent's side or not?
        if True == hit:
            
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
            else:
                
                reward += 2 - min([pow(distance_to_opponent_court[0], 2)+ pow   (distance_to_opponent_court[1], 2), 2.0])            
                
        else:
            land_opponent_court = False
            reward += 0
        
        
        # Process landing position
        if True == land_opponent_court:
            # compute Euclidean distance in x and y coordinate space
            distance_to_target = distance.euclidean(
                landing_info[0:2], self.TARGET_COORDINATE[0:2])
            reward += 2.231 - pow(distance_to_target, 2)
            
        else:
            reward += 0
        self.out_string += "\n\nReward Info:\n====>         hit: " + \
            str(hit) + "\n"
        self.out_string += "\n====> hit opponent: " + \
            str(land_opponent_court) + "\n"
        self.out_string += "\n\nReward:\n====>      reward: " + \
            str(reward) + "\n"
        
        return reward

    def test_reward_function(self):        

        xs = np.linspace(-2.0, 2.0, 31)
        ys = np.linspace(-5.0, 0.0, 201)
        rs = np.zeros((201, 31))
        
        print(self.generateReward(0.9, [1.0], [-0.76, -1.93], [0.1]))
        asd
        for i, x in enumerate(xs):
            print(i, x )
            for j, y in enumerate(ys):
                rs[j,i] = self.generateReward(0.9, [1.0], [x,y], [0.1])
        """
        xs, ys = np.meshgrid(xs, ys)
                
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(xs, ys, rs,cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('r')
        """
        plt.contourf(xs, ys, rs)
        plt.colorbar()
        plt.show()


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

                self.out_string = "\nEpisode: " + \
                    str(internal_episode_counter+1)
                self.out_string += "\n\nBall State: " + str(state_info)

                # get action
                action = self.ccmaes.generate_action(state_info)

                # bound clip action to make it valid for robot
                #bound_action = self.bound_clip(action[-2], action[-1])
                bound_action = self.bound_clip(action[-1])

                #self.export_action(action[:-2], bound_action[0], bound_action[1])
                self.export_action(action[:-1], bound_action)

                # receive reward info
                hit_info, landing_info, ball_racket_dist_info = self.get_reward_info()

                # generate reward
                # reward = self.generateReward(
                #    action[-2:], hit_info, landing_info, ball_racket_dist_info)
                reward = self.generateReward(
                    action[-1], hit_info, landing_info, ball_racket_dist_info)

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
            if 0 == (internal_update_counter+1) % 20:
                plt.savefig(rel_path('../figure/part_promp_May16_update' +
                                     str(internal_update_counter+1)), dpi=100, format='png')

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
