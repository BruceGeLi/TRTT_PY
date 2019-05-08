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

'''
Hyper parameters:
'''
MAX_EPS = 20000      # Max episodes
STATE_DIM = 5       # Dimension of feature function of context s
ACTION_DIM = 3      # Dimension of parameters of action a
COV_SCALE = 1

# Valid boundary for delta_t0, T, w
ACTION_BOUNDARY = np.array([[0.70, 1.1], [0.45, 0.65], [-2.2, -1.3]])

TARGET_COORDINATE = [0.35, -2.93, -0.99]

INITIAL_ACTION = [0.90, 0.5, -1.75]

STOP_THRESHOLD = 3.8

SAMPLE_NUMBER = None

class TrttPortCCmaEs:
    def __init__(self):
        self.ccmaes = ContextualCmaEs(state_dim=STATE_DIM, action_dim=ACTION_DIM, initial_action= INITIAL_ACTION, sample_number=SAMPLE_NUMBER, cov_scale=COV_SCALE, context_feature_type='linear', baseline_feature_type='quadratic')

        self.openSocket()
        self.mainLoop()

    def openSocket(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:8181")

    def get_ball_state(self):
        ball_obs_json = self.socket.recv_json()
        ball_state = ball_obs_json["ball_obs"]
        #print(ball_state)
        ball_state_array = np.array(ball_state)

        # y is constant and therefore be removed
        ball_state_array = np.delete(ball_state_array, 1)

        return ball_state_array

    def bound_clip(self, delta_t0, T, w):
        print("Before clip:\n====>    delta_t0: ", delta_t0,
              "\n====>           T: ", T, "\n====>           w: ", w)
        if delta_t0 < ACTION_BOUNDARY[0][0]:
            delta_t0 = ACTION_BOUNDARY[0][0]
        if delta_t0 > ACTION_BOUNDARY[0][1]:
            delta_t0 = ACTION_BOUNDARY[0][1]

        if T < ACTION_BOUNDARY[1][0]:
            T = ACTION_BOUNDARY[1][0]
        if T > ACTION_BOUNDARY[1][1]:
            T = ACTION_BOUNDARY[1][1]

        if w < ACTION_BOUNDARY[2][0]:
            w = ACTION_BOUNDARY[2][0]
        if w > ACTION_BOUNDARY[2][1]:
            w = ACTION_BOUNDARY[2][1]

        print("After clip:\n====>    delta_t0: ", delta_t0,
              "\n====>           T: ", T, "\n====>           w: ", w)
        return delta_t0, T, w

    def export_action(self, delta_t0, T, w):
        action_json = {"T": T, "delta_t0": delta_t0, "w": w}
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
                landing_info[0:2], TARGET_COORDINATE[0:2])

            if distance_to_target <= 3.0:
                reward += -1 * pow(distance_to_target,1) + 3
            else:
                reward += 0
        else:
            reward += 0
        print("\n====>      reward: ", reward, "\n")
        return reward

    def export_ok(self):
        policy_updated_json = {"policy_ready": True}
        self.socket.send_json(policy_updated_json)

    def mainLoop(self):
        episode_counter = 0
        update_counter = 1
        rw_list = list()
        en_list = list()
        N = self.ccmaes.get_recommand_sample_number()
        plt.figure(figsize=(18, 8), dpi=80)
        
        while episode_counter < MAX_EPS:
            self.ccmaes.print_policy()
            temp_reward = 0.0
            for counter in range(N):
                
                # get state
                state_info = self.get_ball_state()
                print("\nEpisode: ", episode_counter+1)
                print("\nState: ", state_info)

                # get action
                action = self.ccmaes.generate_action(state_info)

                # bound clip action to make it valid for robot
                bound_action = self.bound_clip(action[0], action[1], action[2])

                self.export_action(
                    bound_action[0], bound_action[1], bound_action[2])

                # receive reward info
                hit_info, landing_info, ball_racket_dist_info = self.get_reward_info()

                # generate reward
                reward = self.generateReward(
                    action, hit_info, landing_info, ball_racket_dist_info)
                temp_reward += reward

                # store
                self.ccmaes.store_episode(state_info, action, reward)
                self.export_ok()
                episode_counter += 1

            average_reward = temp_reward / N            
            rw_list.append(average_reward)
            if temp_reward / N > STOP_THRESHOLD:
                print("\nTarget achieved! in episode ", episode_counter)
                """
                rw_list = rw_list[-500:]
                x = range(len(rw_list))
                plt.plot(x, rw_list)
                plt.show()
                """
                break
            plt.ion()
            plt.cla()
            x = np.arange(1,len(rw_list)+1) * N
            plt.subplot(1,2,1)
            plt.plot(x, rw_list)
            plt.xlabel("Episode number")
            plt.ylabel("Average reward")

            self.ccmaes.learn()
            en_list.append(self.ccmaes.get_entropy())
            plt.subplot(1,2,2)
            plt.plot(en_list)
            plt.xlabel("Generation number")
            plt.ylabel("Entropy of distribution")
            plt.pause(0.1)

            update_counter += 1
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    ccmaes_port = TrttPortCCmaEs()
