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
import datetime


class TrttPort:
    def __init__(self, args):
        self.host = args.host
        self.port = args.port
        self.ep_num = args.ep_num
        self.save_num = args.save_num
        self.save_dir_file = args.save_dir_file
        self.restore_dir_file = args.restore_dir_file

        self.policyGradient = None
        self.save_iterator = 0
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
        dist = distance.euclidean(landing_info, target_coordinate)
        if 0.6 < dist:
            reward = -0.1
        elif 0.3 < dist <= 0.6:
            reward = np.cos(dist * 5 * np.pi / 6)
        else:
            reward = 5 * np.cos(dist * 5 * np.pi / 6)
        return reward

    def recordTrainingData(self):
        pass

    def mainLoop(self):
        self.policyGradient = PolicyGradient(
            ball_state_dimension=6, action_dimension=2, hidden_layer_dimension=20, learning_rate=0.01, output_graph=False, restore_dir_file=self.restore_dir_file)

        """
        If Tensorflow-gpu is used, the first action will take quite long time to compute
        In case of Tensorflow-gpu, do one dummy action generation to activate the NN
        """
        self.dummy_ball_state = np.zeros((6)).tolist()
        dummy_action = self.policyGradient.generate_action(self.dummy_ball_state)


        for episode_counter in range(self.ep_num):
            """
                Start of new episode
            """
            print("\n\n\n============================================")
            print("\n------ Episode: {}\n".format(episode_counter+1))

            print("------------------------------------")

            """
                Get ball observation from Vrep sim
            """
            print("\n--- Waiting for ball observation...")
            ball_obs_json = self.socket.recv_json()

            print("--- Ball observation received!\n")
            self.current_ball_state = ball_obs_json["ball_obs"]

            print("------------------------------------")
            """
                Get action parameters from Policy Gradient
            """
            print("\n--- Computing hitting parameters...")
            #t2 = datetime.datetime.now()
            self.current_action = self.policyGradient.generate_action(
                self.current_ball_state).tolist()
            print("--- Hitting parameters computed!\n\n")

            print("====>           T: {:.3f}\n====>    delta_t0: {:.3f}".format(
                self.current_action[0], self.current_action[1]))

            #t3 = datetime.datetime.now()
            # print(t3-t2)
            # print("\n\n")
            action_json = {
                "T": self.current_action[0], "delta_t0": self.current_action[1]}

            # Try a fixed action
            #action_json = {"T": 0.39, "delta_t0": 0.83}
            self.socket.send_json(action_json)
            print("\n")
            print("--- Action exported!\n")
            print("------------------------------------")

            """
                Get ball landing info from Vrep sim
            """
            print("\n--- Waiting for ball landing info...")
            landing_info_json = self.socket.recv_json()
            print("--- Ball landing info received!\n")

            self.current_landing_info = landing_info_json["landing_info"]

            """
                Generate reward
            """
            # print("Transfer landing info into reward.")
            self.current_reward = self.generateReward(
                self.current_landing_info)
            print("====>      Reward: {:.3f}\n".format(self.current_reward))
            self.policyGradient.store_transition(
                self.current_ball_state, self.current_action, self.current_reward)
            print("------------------------------------")
            """
                Update Policy
            """
            print("\n--- Updating hitting policy...")
            #t5 = datetime.datetime.now()

            if self.save_num is not 0:
                self.save_iterator += 1
                self.save_iterator%=self.save_num
                if self.save_iterator is 0:
                    self.policyGradient.learn(save=True, save_dir_file=self.save_dir_file)                    
                else:
                    self.policyGradient.learn()
            else:
                self.policyGradient.learn()
            #t6 = datetime.datetime.now()
            # print(t6-t5)
            policy_updated_json = {"policy_ready": True}
            self.socket.send_json(policy_updated_json)
            print("--- Policy updated!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    def rel_path(fname):
        return os.path.join(os.path.dirname(__file__), fname)

    parser.add_argument('--host', default='localhost',
                        help="Name of the host to connect to")

    parser.add_argument('--port', default=8181,
                        help="Port of the host to connect to")

    parser.add_argument('--ep_num', type=int, default=1000000,
                        help="Number of episode to train the policy.")
    parser.add_argument('--save_num', type=int, default=0,
                        help="Save the Neural network parameters for every N episode.")
    parser.add_argument('--save_dir_file', default='/tmp/RL_NN_parameters.ckpt',
                        help="Dir and file name where the parameters shall be stored.")

    parser.add_argument('--restore_dir_file', default=None,
                        help="Dir and file where the parameters shall be load.")
    args = parser.parse_args()

    pg = TrttPort(args)
