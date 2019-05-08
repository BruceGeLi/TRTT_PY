"""
Author: Ge Li, ge.li@tuebingen.mpg.de

This scipt implements the Contextual CMA-ES class, which is a state of art multi-task optimization Reinforcement Learning method.

Using numpy, scipy

"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.spatial import distance
from scipy.stats import multivariate_normal
import os
import math
import time
import json
import datetime
from sklearn.preprocessing import PolynomialFeatures


class ContextualCmaEs:
    def __init__(self, state_dim, action_dim, initial_action, sample_number=None, cov_scale=1, context_feature_type='linear', baseline_feature_type='linear'):

        #######################################################################
        # Initialize Hyper parameters
        # Hyper parameters are given in all capital letters
        #######################################################################

        # Dimension of parameters of action a
        self.ACTION_DIM = action_dim

        # Dimension of context s
        self.STATE_DIM = state_dim

        # Dimension of feature function of context, e.g. in linear generalization case, [1 s]
        self.CONTEXT_FEATURE_TYPE, self.CONTEXT_FEATURE_DIM = self.process_feature_type_dimension(
            context_feature_type)

        # Dimension of feature function of baseline, e.g. in quadratic case, [1 s s^2]
        self.BASELINE_FEATURE_TYPE, self.BASELINE_FEATURE_DIM = self.process_feature_type_dimension(
            baseline_feature_type)

        # Number of total samples for each update, depends on the dimension of context and action
        if sample_number == None:
            self.N = (int)(4 + 3 * math.log(state_dim +
                                            action_dim) * (1 + 2 * state_dim))
        else:
            self.N = sample_number

        # Number of selected samples for updating the policy, evolution strategy
        self.SN = self.N / 2

        # Regularization coefficient, to avoid inverse a zeros matrix
        self.RC = 1e-9

        #######################################################################
        # Initialize scalars, vectors and matrices used for the algorithm
        # Scalars: all small letters,
        # Vectors: capital letters, and a 'V' in its name
        # Matrices: capital letters, and a 'M' in its name
        # Estimation: capital letters and a '_E' as suffix
        #######################################################################

        # Initialize the Normal Distribution of the Policy

        # Initialize the policy matrix
        self.PM = np.zeros((self.CONTEXT_FEATURE_DIM, self.ACTION_DIM))
        self.PM[0] = initial_action

        # Initialize the old policy matrix
        self.PM_O = np.zeros((self.CONTEXT_FEATURE_DIM, self.ACTION_DIM))

        # Initialize the policy's covariance matrix
        self.CM = np.identity(self.ACTION_DIM) * cov_scale

        # Initialize step size
        self.step_size = 1

        # Initialize the update weight matrix
        self.DM = np.zeros((self.N, self.N))

        # Initialize the contexts feature matrix
        self.CFM = np.zeros((self.N, self.CONTEXT_FEATURE_DIM))

        # Initialize the baseline feature matrix
        self.BFM = np.zeros((self.N, self.BASELINE_FEATURE_DIM))

        # Initialize the sampled action parameter matrix
        self.SAM = np.zeros((self.N, self.ACTION_DIM))

        # Initialize the reward vector
        self.RV = np.zeros(self.N)

        # Initialize the baseline weight vector
        self.BWV = np.zeros(self.BASELINE_FEATURE_DIM)

        # Initialize the estimated advantage vector
        self.AV_E = np.zeros(self.N)

        # Initialize the estimated mean contexts feature vector
        self.CFV_E = np.zeros(self.CONTEXT_FEATURE_DIM)

        # Initialize the Y vector
        self.YV = np.zeros(self.ACTION_DIM)

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
        self.PV_c = np.zeros(self.ACTION_DIM)
        self.PV_sigma = np.zeros(self.ACTION_DIM)

        # Expectation of N dim normal distribution N(0, I)
        self.enn = math.sqrt(self.ACTION_DIM) * (1 - 1 /
                                                 (4 * self.ACTION_DIM) + 1 / (21 * pow(self.ACTION_DIM, 2)))

        # Initialize sample covariance matrix
        self.SCM = np.zeros((self.ACTION_DIM, self.ACTION_DIM))

        # Initialize heaviside function
        self.h_sigma = 0.0

        # Unknown parameters in algorithm
        self.c_1a = 0.0

        #######################################################################
        # Initialize other stuff
        #######################################################################
        self.entropy = 0.0
        self.episode_counter = 0  # sample counter
        self.update_counter = 1

    def process_feature_type_dimension(self, feature_type):
        POSSIBLE_FEATURE_TYPE = ['linear', 'quadratic']
        assert feature_type in POSSIBLE_FEATURE_TYPE, "Wrong feature type is given!"

        if 'linear' == feature_type:
            return 'linear', self.STATE_DIM + 1
        elif 'quadratic' == feature_type:
            print((int)((self.STATE_DIM * self.STATE_DIM + 3 * self.STATE_DIM + 2)/2))
            return 'quadratic', (int)((self.STATE_DIM * self.STATE_DIM + 3 * self.STATE_DIM + 2)/2)
        else:
            assert False

    def generate_feature_function(self, feature_type, state_info):
        if 'linear' == feature_type:
            poly = PolynomialFeatures(degree=1)
            return np.reshape(poly.fit_transform(np.expand_dims(state_info, axis=0)), -1)
        elif 'quadratic' == feature_type:
            poly = PolynomialFeatures(degree=2)
            return np.reshape(poly.fit_transform(np.expand_dims(state_info, axis=0)), -1)
        else:
            assert False

    def get_recommand_sample_number(self):
        return self.N

    def generate_action(self, state_info):
        # Compute Action's Gaussain distribution
        context_feature = self.generate_feature_function(
            self.CONTEXT_FEATURE_TYPE, state_info)
        action_mean = np.matmul(np.transpose(self.PM), context_feature)
        action_cov = pow(self.step_size, 2) * self.CM
        # Sample an action from Gaussain distribution
        action_sample = np.random.multivariate_normal(
            action_mean, action_cov)
        return action_sample

    def store_episode(self, state_info, action, reward):
        # Get sample coutner
        i = self.episode_counter % self.N

        # Build feature matrices
        self.CFM[i] = self.generate_feature_function(
            self.CONTEXT_FEATURE_TYPE, state_info)
        self.BFM[i] = self.generate_feature_function(
            self.BASELINE_FEATURE_TYPE, state_info)

        # Build sampled action matrix
        self.SAM[i] = action

        # Build reward vector
        self.RV[i] = reward

        # increase episode counter
        self.episode_counter += 1

    def print_policy(self):
        print("\nupdate: ", self.update_counter)
        print("\nPolicy matrix: \n", self.PM)
        print("\nCovariance matrix: \n", self.CM)
        if self.episode_counter > 1:
            print("\nAverage reward: ", np.average(self.RV))

    def get_entropy(self):
        return self.entropy

    def learn(self):
        # Update baseline weight vector
        self.BWV = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(
            self.BFM), self.BFM) + self.RC * np.identity(self.BASELINE_FEATURE_DIM)), np.transpose(self.BFM)), self.RV)

        # Get estimated advantage vector
        for j in range(self.N):
            advantage = self.RV[j] - \
                np.matmul(np.transpose(self.BWV),
                          self.BFM[j])
            self.AV_E[j] = advantage
        # End for loop

        # Get rank of estimated advantage vector, rank starts from 1, descending order
        advantage_rank_ascend = np.argsort(
            np.argsort(self.AV_E))  # Get ascending firstly
        advantage_rank = self.N - advantage_rank_ascend

        # Get updating weight matrix, should be diagnol
        z = 0.0
        for k in range(self.N):
            d = max([0, (math.log(self.SN+0.5) - math.log(advantage_rank[k]))])
            self.DM[k, k] = d
            z += d
        self.DM /= z

        # Update Gain matrix
        self.PM_O = self.PM
        self.PM = np.matmul(
            np.matmul(
                np.matmul(
                    np.linalg.inv(
                        np.matmul(
                            np.matmul(
                                np.transpose(
                                    self.CFM), self.DM), self.CFM) + self.RC * np.identity(self.CONTEXT_FEATURE_DIM)), np.transpose(self.CFM)), self.DM), self.SAM)

        # Update estimated mean context feature vector
        self.CFV_E = np.average(self.CFM, axis=0)

        # Update YV
        self.YV = (np.matmul(np.transpose(self.PM), self.CFV_E) -
                   np.matmul(np.transpose(self.PM_O), self.CFV_E)) / self.step_size

        # Update Mu_eff
        self.mu_eff = 1 / np.trace(np.matmul(self.DM, self.DM))

        # Update covariance hyper parameters
        self.c_1 = 2 / \
            (pow(self.ACTION_DIM + self.STATE_DIM + 1.3, 2) + self.mu_eff)

        self.c_mu = min([1 - self.c_1, 2*(self.mu_eff-2+1/self.mu_eff) /
                         (pow(self.ACTION_DIM + self.STATE_DIM+2, 2) + self.mu_eff)])

        self.c_c = (4 + self.mu_eff / (self.ACTION_DIM + self.STATE_DIM)) / \
            (4+self.ACTION_DIM+self.STATE_DIM+2 *
             self.mu_eff/(self.ACTION_DIM + self.STATE_DIM))

        self.c_sigma = (self.mu_eff+2) / \
            (self.ACTION_DIM+self.STATE_DIM+self.mu_eff + 5)

        self.d_sigma = 1+2*max([0, math.sqrt((self.mu_eff-1)/(self.ACTION_DIM + self.STATE_DIM+1))-1]
                               ) + self.c_sigma + math.log(self.ACTION_DIM + self.STATE_DIM + 1)

        # Update evolution path parameters
        self.PV_sigma = (1 - self.c_sigma) * self.PV_sigma + math.sqrt(self.c_sigma*(2 - self.c_sigma)
                                                                       * self.mu_eff) * np.matmul(sp.linalg.fractional_matrix_power(self.CM, -0.5), self.YV)
        # Update h_sigma
        if pow(np.linalg.norm(self.PV_sigma), 2) / (self.ACTION_DIM * math.sqrt(1 - pow(1 - self.c_sigma, 2*self.update_counter))) < 2 + 4/(self.ACTION_DIM + 1):
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
        for l in range(self.N):
            self.SCM += self.DM[l, l] / pow(self.step_size, 2) * np.matmul(np.expand_dims(self.SAM[l] - np.matmul(np.transpose(
                self.PM_O), self.CFM[l]), axis=1), np.expand_dims(self.SAM[l] - np.matmul(np.transpose(self.PM_O), self.CFM[l]), axis=0))

        # Update covariance matrix of policy's normal distribution
        self.CM = (1-self.c_1a - self.c_mu) * self.CM + self.c_1 * np.matmul(np.expand_dims(
            self.PV_c, axis=1), np.expand_dims(self.PV_c, axis=0)) + self.c_mu * self.SCM

        # Compute entropy of the distribution
        self.entropy = 0.5 * math.log(np.linalg.det(2*math.pi*math.e*self.CM))

        # Update step size
        self.step_size = self.step_size * \
            math.exp(self.c_sigma / self.d_sigma *
                     (np.linalg.norm(self.PV_sigma) / self.enn - 1))

        # Update counters
        self.update_counter += 1
