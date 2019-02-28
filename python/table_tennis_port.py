"""
Author: Ge Li, ge.li@tuebingen.mpg.de

This script shall work with the policy gradient script, offering input states and rewards to policy gradient and receive output actions.

This script will link to the vrep simulation, where the ball observation and landing info are generated and the actions should be executed by the robot.

Using:
zmq to communicate with C++ code.

"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import os
import math
import zmq
import time
import json
from policy_gradient import PolicyGradient
from termcolor import colored


class TrttPort:
    def __init__(self, args):
        self.host = args.host
        self.port = args.port
        self.ep_num = args.ep_num

        self.policyGradient = None

        self.current_ball_state = None
        self.current_action = None
        self.current_reward = None

        self.current_landing_info = None

        self.context = None
        self.socket = None

        self.openSocket()
        self.mainLoop()
        self.closeSocket()

    def openSocket(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:8181")

    def closeSocket(self):
        pass

    def generateReward(self, landing_info):
        target_coordinate = [0.35, -3.13, -0.99]
        # compute Euclidean distance
        # print(target_coordinate)
        # print(landing_info)

        dist = distance.euclidean(landing_info, target_coordinate)
        # inverse it
        dist_reverse = dist ** (-1)
        reward = np.tanh(dist_reverse)

        if not (0.3 < self.current_action[0] < 0.5):
            reward = -0.3
        if not (0.8 < self.current_action[1] < 0.9):
            reward = -0.3

        return reward

    def recordTrainingData(self):
        pass

    def mainLoop(self):

        self.policyGradient = PolicyGradient(
            ball_state_dimension=6, action_dimension=2, hidden_layer_dimension=20, learning_rate=0.01, output_graph=False)

        for episode_counter in range(self.ep_num):
            print("\n\n------episode: {}".format(episode_counter+1))
            """
                Get ball observation from Vrep sim
            """

            print("\nWaiting for ball observation...")
            ball_obs_json = self.socket.recv_json()
            print("Ball observation received!\n\n")
            self.current_ball_state = ball_obs_json["ball_obs"]

            # print(self.current_ball_state)

            """
                Get action parameters from Policy Gradient
            """
            print("\nComputing hitting parameters...")
            self.current_action = self.policyGradient.generate_action(
                self.current_ball_state).tolist()
            print("T: {:.2f}, delta_t0: {:.2f}".format(self.current_action[0], self.current_action[1]))

            # Export action to c++
            action_json = {
                "T": self.current_action[0], "delta_t0": self.current_action[1]}
            self.socket.send_json(action_json)
            print("Hitting parameters computed!\n\n")

            """
                Get ball landing info from Vrep sim
            """
            print("\nWaiting for ball landing info...")
            landing_info_json = self.socket.recv_json()
            print("Ball landing info received!\n\n")

            self.current_landing_info = landing_info_json["landing_info"]

            """
                Generate reward
            """
            #print("Transfer landing info into reward.")
            self.current_reward = self.generateReward(
                self.current_landing_info)
            self.policyGradient.store_transition(
                self.current_ball_state, self.current_action, self.current_reward)
            """
                Update Policy
            """
            print("\nUpdating hitting policy...")
            self.policyGradient.learn()

            policy_updated_json = {"policy_ready": True}
            self.socket.send_json(policy_updated_json)
            print("Policy updated!\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    def rel_path(fname):
        return os.path.join(os.path.dirname(__file__), fname)

    parser.add_argument('--host', default='localhost',
                        help="Name of the host to connect to")

    parser.add_argument('--port', default=8181,
                        help="Port of the host to connect to")

    parser.add_argument('--ep_num', type=int, default=1000,
                        help="Number of episode to train the policy.")

    # parser.add_argument('file', help='File name where the training data should be stored.')

    # to do add more parameters of the NN to parameters control

    args = parser.parse_args()

    pg = TrttPort(args)
