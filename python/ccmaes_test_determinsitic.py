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
#np.random.seed(314)
'''
Hyper parameters:
'''
MAX_EPS = 500000000      # Max episodes
STATE_DIM = 6       # Dimension of feature function of context s
ACTION_DIM = 3      # Dimension of parameters of action a

# Number of episodes for each update, should be computed
#N = (int)(4 + 3 * math.log(STATE_DIM + ACTION_DIM) * (1+2*STATE_DIM))
N = 10

# Number of episodes for updating the policy
P = N/2

# Valid boundary for delta_t0, T, w
ACTION_BOUNDARY = np.array([[0.75, 1.05], [0.45, 0.55], [-2.0, -1.5]])

TARGET_COORDINATE = [0.35, -2.93, -0.99]

GM_INITIAL = np.array([[0.01230781, 0.00684402, -0.02393734],
                       [-0.01277858, -0.0071058,  0.02485294],
                       [-0.05745817, -0.03195083,  0.11174987],
                       [0.16406376, 0.09123111, -0.31908612],
                       [0.04678647, 0.0260166, -0.09099459]])

# Stop threshold value
EPSILON = -1e-10

RC = 0.00000001  # Regularization coefficient


class TrttPortCCmaEs:
    def __init__(self):
        # Initialize the Normal Distribution of the Policy
        self.GM = np.ones((STATE_DIM, ACTION_DIM))  # Mean gain matrix
        #self.GM = GM_INITIAL  # Mean gain matrix
        self.GM_O = np.ones((STATE_DIM, ACTION_DIM))  # Old mean gain matrix
        self.CM = np.identity(ACTION_DIM)    # Covariance matrix

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
        self.RV = np.zeros(N)

        # Initialize the Baseline weight vector
        self.BWV = np.zeros(STATE_DIM)

        # Initialize the estimated advantage vector
        self.AV_E = np.zeros(N)

        # Initialize the estimated mean contexts feature vector
        self.CFV_E = np.zeros(STATE_DIM)

        # Initialize the Y vector
        self.YV = np.zeros(ACTION_DIM)

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
        self.PV_c = np.zeros(ACTION_DIM)
        self.PV_sigma = np.zeros(ACTION_DIM)

        # Expectation of N dim normal distribution N(0, I)
        self.enn = math.sqrt(ACTION_DIM) * (1 - 1 /
                                            (4 * ACTION_DIM) + 1 / (21 * pow(ACTION_DIM, 2)))

        # Sample covariance matrix
        self.SCM = np.zeros((ACTION_DIM, ACTION_DIM))

        # Unknown parameters in algorithm
        self.h_sigma = 0.0
        self.c_1a = 0.0

        # Save reward into list
        self.rw_list = list()

    def openSocket(self):
        pass

    def get_ball_state(self):
        generate_error = np.random.randint(1, 3)
        ball_state = list()
        if generate_error == 1:
            ball_state = [1.0, 0.33820778131485, -0.351144075393677, -1.57890021800995, 4.50832843780518, 1.2856513261795]
        else:
            ball_state = [1.0, 0.33820778131485, -0.351144075393677, -1.57890021800995, 4.20832843780518, 1.2856513261795]
        ball_state = [1.0, 0.33820778131485, -0.351144075393677, -1.57890021800995, 4.50832843780518, 1.2856513261795]
        ball_state_mean = np.array(ball_state)
        #ball_state_covariance = np.ones((5, 5)) * 0.01
        #ball_state = np.random.multivariate_normal(
         #   ball_state_mean, ball_state_covariance)
        print(ball_state_mean)
        # return ball_state        
        return ball_state_mean

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
        pass

    def get_reward_info(self):
        pass

    def generateReward(self, ball_state, action, delta_t0, T, w):

        delta_t0 = math.floor(delta_t0 * 100) / 100
        T = math.floor(T * 100) / 100

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


        target = [0.9, 0.5, -1.75]
        reward += -pow(distance.euclidean(target, [delta_t0, T, w]), 2)
        print("\n====>      reward: ", reward, "\n")
        
        return reward

    def export_ok(self):
        pass

    def learn(self):
        # 0. Initialize counters
        episode_counter = 0
        update_counter = 1

        while episode_counter < MAX_EPS:
            print("\nUpdate counter ", update_counter)
            print("Gain matrix: \n", self.GM)
            print("COV matrix: \n", self.CM)
            for i in range(N):
                print("\nEpisode ", episode_counter + 1)
                # Get ball state
                ball_state = self.get_ball_state()

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

                # Make action values in valid boundary to avoid danger to robot
                delta_t0 = action_sample[0]
                T = action_sample[1]
                w = action_sample[2]
                delta_t0, T, w = self.bound_clip(delta_t0, T, w)
                # Export action to simulation
                self.export_action(delta_t0, T, w)

                # Process reward data
                #hit_info, landing_info, ball_racket_dist_info = self.get_reward_info()
                reward = self.generateReward(ball_state, action_sample,
                                             delta_t0, T, w)

                self.RV[i] = reward
                episode_counter += 1
                if((episode_counter) % N != 0):
                    # Send response back
                    self.export_ok()
            # End for loop
            av_rw = np.average(self.RV)
            self.rw_list.append(av_rw)
            ###################################################
            ###################################################
            #########################################################################################################################################################
            # Check stop
            if np.average(self.RV) > EPSILON:
                print("\nTarget achieved! in episode ", episode_counter)
                self.rw_list = self.rw_list[-50:]
                x = range(len(self.rw_list))
                plt.plot(x, self.rw_list)
                plt.show()
                break

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

            # Get rank of estimated advantage vector, rank starts from 1, descending order
            advantage_rank_ascend = np.argsort(
                np.argsort(self.AV_E))  # Get ascending firstly
            advantage_rank = N - advantage_rank_ascend

            # Get updating weight matrix, should be diagnol
            z = 0.0
            for k in range(N):
                d = max([0, (math.log(P+0.5) - math.log(advantage_rank[k]))])
                self.DM[k, k] = d
                z += d
            self.DM /= z

            # Update Gain matrix
            self.GM_O = self.GM
            self.GM = np.matmul(
                np.matmul(
                    np.matmul(
                        np.linalg.inv(
                            np.matmul(
                                np.matmul(
                                    np.transpose(
                                        self.CFM), self.DM), self.CFM) + RC * np.identity(STATE_DIM)), np.transpose(self.CFM)), self.DM), self.SAM)

            # Update estimated mean context feature vector
            self.CFV_E = np.average(self.CFM, axis=0)

            # Update YV
            self.YV = (np.matmul(np.transpose(self.GM), self.CFV_E) -
                       np.matmul(np.transpose(self.GM_O), self.CFV_E)) / self.step_size

            # Update Mu_eff
            self.mu_eff = 1 / np.trace(np.matmul(self.DM, self.DM))

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
                                                                           * self.mu_eff) * np.matmul(sp.linalg.fractional_matrix_power(self.CM, -0.5), self.YV)
            # Update h_sigma
            if pow(np.linalg.norm(self.PV_sigma), 2) / (ACTION_DIM * math.sqrt(1 - pow(1 - self.c_sigma, 2*update_counter))) < 2 + 4/(ACTION_DIM + 1):
                self.h_sigma = 1
            else:
                self.h_sigma = 0

            # Update evolution path parameters
            self.PV_c = (1 - self.c_c) * self.PV_c + self.h_sigma * \
                math.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * self.YV

            # Update c_1a
            self.c_1a = self.c_1 * \
                (1 - (1 - self.h_sigma) * self.c_c * (2-self.c_c))

            # Update Sample covariance matrix
            self.SCM = np.zeros(self.SCM.shape)
            for l in range(N):
                self.SCM += self.DM[l, l] / pow(self.step_size, 2) * np.matmul(np.expand_dims(self.SAM[l] - np.matmul(np.transpose(
                    self.GM_O), self.CFM[l]), axis=1), np.expand_dims(self.SAM[l] - np.matmul(np.transpose(self.GM_O), self.CFM[l]), axis=0))

            # Update covariance matrix of policy's normal distribution
            self.CM = (1-self.c_1a - self.c_mu) * self.CM + self.c_1 * np.matmul(np.expand_dims(
                self.PV_c, axis=1), np.expand_dims(self.PV_c, axis=0)) + self.c_mu * self.SCM

            # Update step size
            self.step_size = self.step_size * \
                math.exp(self.c_sigma / self.d_sigma *
                         (np.linalg.norm(self.PV_sigma) / self.enn - 1))

            # Update counters
            update_counter += 1

            # After updating, send ok signal to simulation
            self.export_ok()


if __name__ == "__main__":
    ccmaes_port = TrttPortCCmaEs()
    ccmaes_port.openSocket()
    ccmaes_port.learn()
