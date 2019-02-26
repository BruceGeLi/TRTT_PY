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
import os
import math
import zmq
import time
import json
from policy_gradient import PolicyGradient


class TrttPort:
    def __init__(self, args):
        self.host = args.host
        self.port = args.port
        self.ep_num = args.ep_num

        self.policyGradient = None

        self.current_state = None
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
        reward = list()
        return reward

    def recordTrainingData(self):
        pass

    def mainLoop(self):
        '''
        self.policyGradient = PolicyGradient(
            input_layer_size=6, output_layer_size=4, hidden_layer_size=20, learning_rate=0.01)
        '''

        for episode_counter in range(self.ep_num):
            print("\n\n------episode: {}".format(episode_counter+1))
            """
                Get ball observation from Vrep sim
            """
            print("\nWaiting for ball observation...")
            ball_obs_json = self.socket.recv_json()
            print("Ball observation received!")
            
            self.current_state = ball_obs_json["ball_obs"]
            
            
            """
                Get action parameters from Policy Gradient     
            """
            print("\nComputing hitting parameters...")
            #self.current_action = self.policyGradient.choose_action(
            #    self.current_state)
            self.current_action = [0.35, 0.80]
            action_json = {"T": 0.25, "delta_t0": 0.8}

            # Export action to c++
            self.socket.send_json(action_json)
            print("Hitting parameters computed!")


            """
                Get ball landing info from Vrep sim
            """
            print("\nWaiting for ball landing info...")
            landing_info_json = self.socket.recv_json()
            print("Ball landing info received!")
            print("Transfer landing info into reward.")
            self.current_landing_info = landing_info_json["landing_info"]

            self.current_reward = self.generateReward(
                self.current_landing_info)

            #self.policyGradient.store_transition(
            #    self.current_state, self.current_action, self.current_reward)

            """
                Update Policy
            """
            print("\nUpdating hitting policy...")
            #self.policyGradient.learn()
            # todo export updated to c++
            policy_updated_json = {"policy_updated": True}
            self.socket.send_json(policy_updated_json)
            print("Policy updated!")

            


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

    #parser.add_argument('file', help='File name where the training data should be stored.')

    # to do add more parameters of the NN to parameters control

    args = parser.parse_args()

    pg = TrttPort(args)
