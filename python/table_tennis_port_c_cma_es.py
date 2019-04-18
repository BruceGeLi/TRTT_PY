import zmq

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
MAX_EPS = 3000      # Max episodes
STATE_DIM = 6       # Dimension of feature function of context s
ACTION_DIM = 3      # Dimension of parameters of action a

# Number of episodes for each update, should be computed
#N = (int)(4 + 3 * math.log(STATE_DIM + ACTION_DIM) * (1+2*STATE_DIM))
N = 20

# Number of episodes for updating the policy
P = 10


INITIAL_ACTION_MEAN = [0.9, 0.5, -1.75]
INITIAL_ACTION_VARIANCE = 0.03

TARGET_COORDINATE = [0.35, -2.93, -0.99]

RC = 10.0  # Regularization coefficient


class TrttPortCCmaEs:
    def __init__(self):
        # Initialize the Normal Distribution of the Policy
        self.GM_O = np.ones((STATE_DIM, ACTION_DIM))  # Old mean gain matrix
        self.GM = np.ones((STATE_DIM, ACTION_DIM))  # Mean gain matrix
        self.CM = np.ones(ACTION_DIM, ACTION_DIM)   # Covariance matrix
        self.step_size = 1

        # Initialize the update weight matrix
        self.DM = np.zeros((N, N))

        # Initialize the contexts feature matrix
        self.CFM = np.zeros((N, STATE_DIM))

        # Initialize the baseline feature matrix
        self.BFM = np.zeros((N, STATE_DIM))

        # Initialize the sampled action parameter matrix
        self.SAM = np.zeros((N, ACTION_DIM))

        # Initialize the Reward vector
        self.RV = np.zeros((N, 1))

        # Initialize the Baseline weight vector
        self.BWV = np.zeros((N, 1))

        # Initialize the estimated advantage vector
        self.AV_E = np.zeros((N, 1))

        # Initialize the estimated mean contexts feature vector
        self.CFV_E = np.zeros((STATE_DIM, 1))

        # Initialize the XXXXX
        self.y = np.zeros((ACTION_DIM, 1))

        # Initialize the Mu_eff
        self.mu_eff = 0.0

        # Initialize the covariance hyper parameters
        self.c_1 = 0.0          # Rank 1
        self.c_mu = 0.0         # Rank mu
        self.c_c = 0.0

        # Initialize the step size hyper parameters
        self.c_sigma = 0.0
        self.d_sigma = 0.0

        # Initialize the evolution path vector
        self.PV_c = np.array((ACTION_DIM, 1))
        self.PV_sigma = np.array((ACTION_DIM, 1))

        # Expectation of N dim normal distribution N(0, I)
        self.enn = math.sqrt(ACTION_DIM) * (1 - 1 /
                                            (4 * ACTION_DIM) + 1 / (21 * pow(ACTION_DIM, 2)))

    def openSocket(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:8181")

    def generateReward(self, hit_info, landing_info, distance_info):
        reward = 0
        hit = False
        # hit or not
        if hit_info[0] < 0.0:
            reward += 0
            hit = False
        else:
            reward += 1
            hit = True

        """
            Second level of reward represents landing position
        """
        if True == hit:
            # compute Euclidean distance in x and y coordinate space
            distance_to_target = distance.euclidean(
                landing_info[0:2], TARGET_COORDINATE[0:2])

            if distance_to_target <= 3.0:
                reward += -1 * distance_to_target + 3
            else:
                reward += 0
        else:
            reward += 0

        return reward

    def bound_clip(self, delta_t0, T, w):
        if delta_t0 < 0.7:
            delta_t0 = 0.7
        if delta_t0 > 1.2:
            delta_t0 = 1.2

        if T < 0.45:
            T = 0.45
        if T > 0.65:
            T = 0.65

        if w < -2.25:
            w = -2.25
        if w > -1.45:
            w = -1.45

        return delta_t0, T, w

    def learn(self):
        episode_counter = 0
        while episode_counter < 0:

            # Reset
            # Some potential reset work here ...

            for i in range(N):
                # Get ball state
                ball_obs_json = self.socket.recv_json()
                ball_state = ball_obs_json["ball_obs"]
                ball_state = np.array(ball_state)

                # Build feature matrices
                self.CFM[i] = ball_state
                self.BFM[i] = ball_state

                # Compute Action's Gaussain distribution
                action_mean = np.matmul(np.transpose(self.GM), ball_state)
                action_cov = pow(self.step_size, 2) * self.CM

                # Sample an action from Gaussain distribution
                action_sample = np.random.multivariate_normal(
                    action_mean, action_cov)

                # Build sampled action matrix
                self.SAM[i] = action_sample

                # Make action values in valid boundary
                delta_t0 = action_sample[0]
                T = action_sample[1]
                w = action_sample[2]
                delta_t0, T, w = self.bound_clip(delta_t0, T, w)
                print("\n====>    delta_t0: ", delta_t0,
                      "\n====>           T: ", T, "\n====>           w: ", w)

                # Export action to simulation
                action_json = {"T": T, "delta_t0": delta_t0, "w": w}
                self.socket.send_json(action_json)

                # Process reward data
                reward_info_json = self.socket.recv_json()
                landing_info = reward_info_json["landing_point"]
                ball_racket_dist_info = reward_info_json["min_distance"]
                hit_info = reward_info_json["hit"]
                reward = self.generateReward(
                    hit_info, landing_info, ball_racket_dist_info)
                print("\n====>      reward: ", reward, "\n")
                self.RV[i] = reward

                # Send response back
                policy_updated_json = {"policy_ready": True}
                self.socket.send_json(policy_updated_json)
                episode_counter += 1
            # End for loop

            # Update baseline weight vector
            self.BWV = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(
                self.BFM), self.BFM) + RC * np.identity(STATE_DIM)), np.transpose(self.BFM)), self.RV)

            # Get estimated advantage vector
            for j in range(N):
                advantage = self.RV[j] - \
                    np.matmul(np.transpose(self.BWV),
                              self.BFM[j])
                self.AV_E[j] = advantage
            # End for loop

            # Get rank of estimated advantage vector, rank starts from 1
            advantage_rank = np.argsort(np.argsort(self.AV_E)) + 1

            # Get updating weight matrix, should be diagnol
            z = 0.0
            for k in range(N):
                d = max([0, (math.log(P+0.5) - math.log(advantage_rank[k]))])
                self.DM[k, k] = d
                z += d
            self.DM /= z

            # Update Gain matrix
            self.GM_O = self.GM
            self.GM = np.matmul(np.matmul(np.matmul(np.linalg.inv(np.matmul(np.matmul(np.transpose(
                self.CFM), self.DM), self.CFM + RC * np.identity(STATE_DIM)), np.transpose(self.CFM)), self.DM), self.SAM))

            # Update estimated mean context feature vector
            self.CFV_E = np.average(self.CFM, axis=0)

            # Update y
            self.y = (np.matmul(self.GM, self.CFV_E) -
                      np.matmul(self.GM_O, self.CFV_E)) / self.step_size

            # Update Mu_eff
            self.mu_eff = N / np.matmul(self.DM, self.DM)

            # Update covariance hyper parameters
            self.c_1 = 2 / (pow(ACTION_DIM + STATE_DIM + 1.3, 2) + self.mu_eff)

            self.c_mu = min([1 - self.c_1, 2*(self.mu_eff-2+1/self.mu_eff) /
                             (pow(ACTION_DIM + STATE_DIM+2, 2) + self.mu_eff)])

            self.c_c = (4 + self.mu_eff / (ACTION_DIM + STATE_DIM)) / \
                (4+ACTION_DIM+STATE_DIM+2*self.mu_eff/(ACTION_DIM + STATE_DIM))

            self.c_sigma = (self.mu_eff+2) / \
                (ACTION_DIM+STATE_DIM+self.mu_eff + 5)

            self.d_sigma = 1+2*max([0, math.sqrt((self.mu_eff-1)/(ACTION_DIM + STATE_DIM+1))-1]
                                   ) + self.c_sigma + math.log(ACTION_DIM + STATE_DIM + 1)

            # Update evolution path parameters
            self.PV_sigma = (1 - self.c_sigma) * self.PV_sigma + math.sqrt(self.c_sigma*(2 - self.c_sigma)
                                                                           * self.mu_eff) * np.matmul(sp.linalg.fractional_matrix_power(self.CM, -0.5), self.y)
                                                                                                                # uPDATE                                                               


if __name__ == "__main__":
    ccmaes_port = TrttPortCCmaEs()
    ccmaes_port.openSocket()
    ccmaes_port.learn()
