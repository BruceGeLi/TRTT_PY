"""
Author: Ge Li, ge.li@tuebingen.mpg.de

This script is a dummy port to send manual action to robot 

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

class TrttDummyPort:
    def __init__(self):
        self.context = None
        self.socket = None

        self.openSocket()
        self.sampling_from_json()

    def openSocket(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        try:
            self.socket.bind("tcp://*:8181")            

        except:
            self.socket.bind("tcp://*:8182")
    
    def closeSocket(self):
        pass
            
    def generateReward(self, landing_info, distance_info):
        reward = 0

        """
            Second level of reward represents landing position
        """
        target_coordinate = [0.35, -2.93, -0.99]
        # compute Euclidean distance in x and y coordinate space
        distance_to_target = distance.euclidean(
            landing_info[0:2], target_coordinate[0:2])
        #print("\ndistance to target: ", distance_to_target)
        # print("\n")
        if distance_to_target <= 3.0:
            reward += -1 * distance_to_target + 3
        else:
            reward += 0

        return reward            

    def sampling_from_json(self):
        counter = 0
        while True:
            sampling_file = rel_path('../config/sampling.json')
            with open(sampling_file) as j_file:
                j_obj = json.load(j_file)
                T = j_obj["T"]
                delta_t0 = j_obj["delta_t0"]
                w = j_obj["w"]

                counter += 1
            print("\n====>    delta_t0: ", delta_t0,
                  "\n====>           T: ", T, "\n====>           w: ", w,)
            ball_obs_json = self.socket.recv_json()

            action_json = {"T": T, "delta_t0": delta_t0, "w": w}
            self.socket.send_json(action_json)

            reward_info_json = self.socket.recv_json()
            print(reward_info_json)
            current_landing_info = reward_info_json["landing_point"]
            print(current_landing_info)
            current_ball_racket_dist_info = reward_info_json["min_distance"]
            print(current_ball_racket_dist_info)
            current_hit_info = reward_info_json["hit"]
            print(current_hit_info)
            policy_updated_json = {"policy_ready": True}
            self.socket.send_json(policy_updated_json)

    
if __name__ == "__main__":
    def rel_path(fname):
        return os.path.join(os.path.dirname(__file__), fname)
    dp = TrttDummyPort()