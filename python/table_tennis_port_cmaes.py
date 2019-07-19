import cma
import zmq

import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import os
import math
import time
import json
import datetime


class TrttPortCmaEs:
    def __init__(self):
        pass

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

    def bound_clip(self):
        pass

    def optimize(self):
        counter = 0
        es = cma.CMAEvolutionStrategy([0.9, 0.5, -1.75], 0.03)
        while counter < 1000:
            es_samples = es.ask()
            minus_rewards = list()
            for sample in es_samples:
                ball_obs_json = self.socket.recv_json()

                delta_t0 = sample[0]
                T = sample[1]
                if T < 0.45:
                    T = 0.45
                w = sample[2]
                print("\n====>    delta_t0: ", delta_t0,
                      "\n====>           T: ", T, "\n====>           w: ", w,)

                action_json = {"T": T, "delta_t0": delta_t0, "w": w}
                self.socket.send_json(action_json)

                reward_info_json = self.socket.recv_json()

                landing_info = reward_info_json["landing_point"]
                ball_racket_dist_info = reward_info_json["min_distance"]
                hit_info = reward_info_json["hit"]

                reward = self.generateReward(
                    hit_info, landing_info, ball_racket_dist_info)

                print("\n====>      reward: ", reward,"\n")
                minus_rewards.append(-reward)
                policy_updated_json = {"policy_ready": True}
                self.socket.send_json(policy_updated_json)
                counter += 1

            es.tell(es_samples, minus_rewards)
            es.logger.add()
            es.disp()

        es.result_pretty()


if __name__ == "__main__":
    cmaes_port = TrttPortCmaEs()
    cmaes_port.openSocket()
    cmaes_port.optimize()
